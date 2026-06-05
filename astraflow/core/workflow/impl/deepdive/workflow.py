"""DeepDive recursive-agent workflow (action-based format, TextCraft-style).

Each turn the model emits ONE action block:
    <action type="NAME">{JSON args}</action>

Action types (the only valid ones):
  - search   {"query": str, "n_docs": int = 5}
  - spawn    {"goal": str}
  - finish   {"answer": str}

The workflow parses the block, dispatches to the matching env method, and
returns the observation as the next user message. No Python sandbox — the
agent cannot "forget to print" search results.

reward_mode controls which reward each agent's tokens train on:
  - team_credit:     every agent trains on root's LLM-judged reward
  - per_agent_judge: each agent trains on its own LLM-judged reward
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiofiles
import aiofiles.os
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest, ModelResponse
from astraflow.core.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id
from astraflow.core.workflow.utils.dynamic_import import import_from_string
from astraflow.core.workflow.impl.deepdive.env import (
    DEFAULT_PASSAGE_TRUNCATE,
    DeepDiveEnv,
)
from astraflow.core.workflow.impl.deepdive.tasks import Task, get_task

logger = logging.getLogger("DeepDiveRecursive")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MAIN_SYSTEM_PROMPT = """You are a research agent answering a factual question using a knowledge corpus.

You act by emitting EXACTLY ONE action block per turn, in this format:
  <action type="ACTION_NAME">{JSON args}</action>

After each action you will receive an observation as the next user message. Use it to decide your next action.

Available actions:

- search  — Query the knowledge corpus. Returns up to n_docs passages, each with text + source + score.
    <action type="search">{"query": "first president of the united states", "n_docs": 5}</action>

- spawn   — Delegate a focused sub-question to a child researcher. Returns the child's final answer string.
    <action type="spawn">{"goal": "find the birth year of the actor who played Robin Hood in Disney's 1973 animated film"}</action>

- finish  — Submit your final answer and end the episode.
    <action type="finish">{"answer": "George Washington"}</action>

Strategy:
- ALWAYS call `search` at least once before `finish`. Do NOT answer from memory alone.
- If a search returns nothing useful, REPHRASE the query (different keywords) and try again.
- Cross-check key claims across multiple passages when possible.
- Only `spawn` for clearly independent sub-questions; do not just forward the whole task.
- Keep your final answer concise and direct; semantic accuracy matters, formatting does not.

Do not output any text outside the <action ...>...</action> block.
"""

SUB_SYSTEM_PROMPT = """You are a sub-research-agent dispatched by a parent agent to answer a focused sub-question.

You act by emitting EXACTLY ONE action block per turn (same format as the parent):
  <action type="ACTION_NAME">{JSON args}</action>

Available actions:

- search  <action type="search">{"query": "...", "n_docs": 5}</action>
- spawn   <action type="spawn">{"goal": "..."}</action>   (you MAY recurse; depth-limited)
- finish  <action type="finish">{"answer": "..."}</action>

Strategy:
- ALWAYS search at least once before finishing.
- Your `finish` answer is the ONLY thing the parent will see — be concise but complete.
- You have a tight step budget; do not over-search.

Do not output any text outside the <action ...>...</action> block.
"""


# ---------------------------------------------------------------------------
# Action parsing  (cloned from TextCraft's parse_action)
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"<action\s+type=\"(\w+)\"\s*>\s*(\{.*?\})\s*</action>",
    re.DOTALL,
)
_VALID_TYPES = frozenset({"search", "spawn", "finish"})


@dataclass
class ParsedAction:
    type: str
    args: dict[str, Any]
    error: str | None = None
    raw_text: str = ""


def parse_action(text: str) -> ParsedAction:
    """Extract the first <action type="...">{json}</action> block."""
    m = _ACTION_RE.search(text)
    if not m:
        return ParsedAction(
            type="__noaction__",
            args={},
            error='no <action type="...">{...}</action> block found in response',
            raw_text=text,
        )
    action_type = m.group(1)
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError as e:
        return ParsedAction(
            type=action_type,
            args={},
            error=f"action JSON decode failed: {e}",
            raw_text=m.group(0),
        )
    if not isinstance(args, dict):
        return ParsedAction(
            type=action_type,
            args={},
            error=f"action args must be a JSON object, got {type(args).__name__}",
            raw_text=m.group(0),
        )
    return ParsedAction(type=action_type, args=args, raw_text=m.group(0))


def validate_action(action: ParsedAction) -> str | None:
    """Return None if valid, else an error string the model can learn from."""
    if action.error is not None:
        return action.error
    if action.type not in _VALID_TYPES:
        return f"unknown action type: {action.type!r}; valid: search / spawn / finish"
    if action.type == "search":
        q = action.args.get("query")
        if not isinstance(q, str) or not q.strip():
            return "search: 'query' must be a non-empty string"
        n = action.args.get("n_docs", 5)
        if not isinstance(n, int) or n < 1 or n > 20:
            return "search: 'n_docs' must be an int in [1, 20]"
    elif action.type == "spawn":
        g = action.args.get("goal")
        if not isinstance(g, str) or not g.strip():
            return "spawn: 'goal' must be a non-empty string"
    elif action.type == "finish":
        # answer can be any JSON-serializable scalar; coerce to string later
        if "answer" not in action.args:
            return "finish: 'answer' is required"
    return None


# ---------------------------------------------------------------------------
# Step-budget tracker (shared across the whole trajectory tree)
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    total: int
    used: int = 0
    reserved: int = 0

    def remaining(self) -> int:
        return max(0, self.total - self.used - self.reserved)

    def consume(self, n: int = 1) -> bool:
        if self.remaining() < n:
            return False
        self.used += n
        return True

    def reserve(self, n: int) -> bool:
        if self.remaining() < n:
            return False
        self.reserved += n
        return True

    def release(self, n: int) -> None:
        self.reserved = max(0, self.reserved - n)


# ---------------------------------------------------------------------------
# Per-agent trajectory record
# ---------------------------------------------------------------------------


@dataclass
class AgentTrajectory:
    traj_id: str
    parent_id: str | None
    depth: int
    task: Task
    is_root: bool
    turns: list[tuple[list[int], "ModelResponse"]] = field(default_factory=list)
    finish_payload: Any = None
    finish_message: str = ""
    error_message: str | None = None
    reward: float = 0.0
    bonus: float = 0.0
    messages: list[dict[str, str]] = field(default_factory=list)
    # Judge output captured at evaluate() time — contains "reason" and
    # (for sub-agents) the checklist details. Surfaced in rollout dumps
    # so we can debug why a reward was 0 vs 1 without re-running the
    # judge by hand.
    eval_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@register_workflow("deepdive_recursive")
class DeepDiveRecursiveWorkflow(RolloutWorkflow):
    """Recursive web-research workflow over the DeepDive Q&A benchmark."""

    def __init__(
        self,
        reward_fn: str | Any = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
        gconfig: GenerationHyperparameters | None = None,
        max_depth: int = 4,
        max_steps_per_episode: int = 25,
        max_concurrent_subagents: int = 4,
        delegation_lambda: float = 0.0,
        sub_max_steps: int = 15,
        passage_truncate: int = DEFAULT_PASSAGE_TRUNCATE,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        dump_prob: float = 0.0,
        **kwargs: Any,
    ):
        self.reward_fn = reward_fn
        self.async_reward_fn: Any = None
        if isinstance(reward_fn, str):
            try:
                self.reward_fn = import_from_string(reward_fn)
                self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)
            except Exception as e:
                logger.warning("could not import reward_fn=%r: %s", reward_fn, e)
                self.reward_fn = None

        self.tokenizer = tokenizer
        self.gconfig = gconfig
        self.max_depth = max_depth
        # max_breadth intentionally absent: DeepDive's spawn action is
        # one-goal-per-turn (unlike TextCraft's batched spawn), so the
        # bound that matters is total budget + max_depth + the
        # max_concurrent_subagents semaphore. A per-agent breadth cap is
        # not wired up — dropped from config rather than left as dead code.
        self.max_steps_per_episode = max_steps_per_episode
        self.max_concurrent_subagents = max_concurrent_subagents
        self.delegation_lambda = float(delegation_lambda)
        self.sub_max_steps = sub_max_steps
        self.passage_truncate = passage_truncate
        self.enable_thinking = bool(enable_thinking)
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.dump_prob = float(dump_prob)
        self.reward_mode: str = str(kwargs.pop("reward_mode", "team_credit"))
        if self.reward_mode not in ("team_credit", "per_agent_judge"):
            raise ValueError(
                f"reward_mode must be 'team_credit' or 'per_agent_judge', "
                f"got {self.reward_mode!r}"
            )
        self.judge_model = kwargs.pop("judge_model", None)

    # ------------------------------------------------------------------ utils

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("DeepDiveRecursiveWorkflow has no tokenizer attached")
        kwargs: dict[str, Any] = dict(
            add_generation_prompt=add_generation_prompt,
            return_tensors=None,
        )
        try:
            out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            out = self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        # transformers>=5 returns a BatchEncoding (a Mapping, not a list) when
        # tokenize=True; older versions return a flat list[int].
        if hasattr(out, "keys"):
            out = out["input_ids"]
        return list(out)

    def _build_initial_messages(self, task: Task, is_root: bool) -> list[dict]:
        system = MAIN_SYSTEM_PROMPT if is_root else SUB_SYSTEM_PROMPT
        if is_root:
            user = f"Question: {task.goal}\n"
        else:
            user = f"Sub-task assigned by parent: {task.goal}\n"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------------ episode

    async def _run_episode(
        self,
        engine: InferenceEngine,
        env: DeepDiveEnv,
        task: Task,
        budget: BudgetTracker,
        sem: asyncio.Semaphore,
        all_trajs: list[AgentTrajectory],
        parent_id: str | None,
        depth: int,
        is_root: bool,
    ) -> AgentTrajectory:
        traj_id = uuid.uuid4().hex
        traj = AgentTrajectory(
            traj_id=traj_id, parent_id=parent_id, depth=depth, task=task, is_root=is_root,
        )
        all_trajs.append(traj)

        messages = self._build_initial_messages(task, is_root)
        steps_taken = 0
        max_local_steps = (
            task.max_steps if task.max_steps is not None
            else (self.max_steps_per_episode if is_root else self.sub_max_steps)
        )

        while not env.finished and steps_taken < max_local_steps:
            if not budget.consume(1):
                traj.error_message = (
                    f"[budget exhausted at local step {steps_taken}; "
                    f"total {budget.used}/{budget.total}]"
                )
                break

            input_ids = self._apply_chat_template(messages, add_generation_prompt=True)
            resp = await engine.agenerate(
                ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
            )
            traj.turns.append((input_ids, resp))
            response_text = self.tokenizer.decode(resp.output_tokens)

            parsed = parse_action(response_text)
            err = validate_action(parsed)
            messages.append({"role": "assistant", "content": response_text})

            if err is not None:
                obs = f"ERROR: {err}. Reminder: emit ONE <action type=\"...\">{{...}}</action> block per turn."
                stats_tracker.get(self.rollout_stat_scope).scalar(parse_errors=1)
                messages.append({"role": "user", "content": obs})
                steps_taken += 1
                continue

            # Dispatch to env action.
            if parsed.type == "search":
                obs = await env.search(
                    query=parsed.args["query"],
                    n_docs=int(parsed.args.get("n_docs", 5)),
                )
            elif parsed.type == "spawn":
                obs = await env.spawn(goal=parsed.args["goal"])
            elif parsed.type == "finish":
                env.finish(parsed.args["answer"])
                traj.finish_payload = env.finish_payload
                traj.finish_message = "" if env.finish_payload is None else str(env.finish_payload)
                score, info = await env.evaluate()
                traj.reward = float(score)
                traj.eval_info = info  # B3: preserve judge reason for dumps
                steps_taken += 1
                break
            else:
                obs = f"ERROR: unhandled action type {parsed.type!r}"

            messages.append({"role": "user", "content": obs})
            steps_taken += 1

        traj.messages = messages
        return traj

    # ------------------------------------------------------------------ spawn

    def _make_spawn_callback(
        self,
        engine: InferenceEngine,
        budget: BudgetTracker,
        sem: asyncio.Semaphore,
        all_trajs: list[AgentTrajectory],
        parent_id: str,
        parent_depth: int,
    ):
        """Async fn the env's spawn() action calls. Single-goal per call."""
        async def _launch(goal: str) -> str:
            if parent_depth >= self.max_depth:
                return f"ERROR: max recursion depth ({self.max_depth}) reached; cannot spawn"
            child_task = Task(
                goal=str(goal),
                id=f"{parent_id}/sub_{uuid.uuid4().hex[:8]}",
                max_steps=self.sub_max_steps,
                misc={},
            )
            child_env = DeepDiveEnv(
                task=child_task,
                spawn_callback=self._make_spawn_callback(
                    engine, budget, sem, all_trajs, child_task.id, parent_depth + 1
                ),
                passage_truncate=self.passage_truncate,
                judge_model=self.judge_model,
            )
            async with sem:
                child_traj = await self._run_episode(
                    engine=engine,
                    env=child_env,
                    task=child_task,
                    budget=budget,
                    sem=sem,
                    all_trajs=all_trajs,
                    parent_id=parent_id,
                    depth=parent_depth + 1,
                    is_root=False,
                )
            return child_traj.finish_message or (child_traj.error_message or "")

        return _launch

    # ------------------------------------------------------------------ reward

    def _apply_delegation_bonus(self, all_trajs: list[AgentTrajectory]) -> None:
        if self.delegation_lambda <= 0:
            return
        children: dict[str, list[AgentTrajectory]] = {}
        for t in all_trajs:
            if t.parent_id is not None:
                children.setdefault(t.parent_id, []).append(t)
        for t in all_trajs:
            kids = children.get(t.traj_id, [])
            if not kids:
                continue
            mean_child_success = sum(k.reward for k in kids) / len(kids)
            bonus = self.delegation_lambda * mean_child_success
            t.bonus = bonus
            t.reward = t.reward + bonus

    # --------------------------------------------------------- sequence packing

    def _build_sequences_for_agent(self, ag: AgentTrajectory, reward: float) -> list[dict[str, torch.Tensor]]:
        out: list[dict[str, torch.Tensor]] = []
        for input_ids, resp in ag.turns:
            seq_ids = list(input_ids) + list(resp.output_tokens)
            seq_mask = [0] * len(input_ids) + [1] * len(resp.output_tokens)
            seq_logprobs = [0.0] * len(input_ids) + list(resp.output_logprobs)
            seq_versions = [-1] * len(input_ids) + list(resp.output_versions)
            n = len(seq_ids)
            seq = {
                "input_ids": torch.tensor(seq_ids, dtype=torch.int32),
                "loss_mask": torch.tensor(seq_mask, dtype=torch.int32),
                "logprobs": torch.tensor(seq_logprobs, dtype=torch.float32),
                "versions": torch.tensor(seq_versions, dtype=torch.int32),
                "attention_mask": torch.ones(n, dtype=torch.bool),
                "rewards": torch.tensor(reward, dtype=torch.float32),
            }
            out.append({k: v.unsqueeze(0) for k, v in seq.items()})
        return out

    # ------------------------------------------------------------------ entry

    async def _run_one_rollout(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
        rollout_idx: int,
    ) -> dict[str, Any]:
        task = self._task_from_data(data)
        budget = BudgetTracker(total=task.max_steps or self.max_steps_per_episode)
        sem = asyncio.Semaphore(self.max_concurrent_subagents)
        all_trajs: list[AgentTrajectory] = []

        root_id = task.id
        env = DeepDiveEnv(
            task=task,
            spawn_callback=self._make_spawn_callback(
                engine, budget, sem, all_trajs, root_id, parent_depth=0
            ),
            passage_truncate=self.passage_truncate,
            judge_model=self.judge_model,
        )

        await self._run_episode(
            engine=engine, env=env, task=task, budget=budget, sem=sem,
            all_trajs=all_trajs, parent_id=None, depth=0, is_root=True,
        )

        self._apply_delegation_bonus(all_trajs)

        root_traj = all_trajs[0] if all_trajs else None
        root_reward = float(root_traj.reward) if root_traj else 0.0

        per_agent: list[dict[str, Any]] = []
        for ag in all_trajs:
            if not ag.turns:
                continue
            agent_reward = self._reward_for_agent(ag, root_reward)
            seqs = self._build_sequences_for_agent(ag, agent_reward)
            per_agent.append({
                "reward": agent_reward,
                "sequences": seqs,
                "depth": ag.depth,
                "is_root": ag.is_root,
            })

        return {
            "per_agent": per_agent,
            "all_trajs": all_trajs,
            "task": task,
            "root_reward": root_reward,
            "n_agents": len(all_trajs),
            "subagent_launched": int(env.subagent_launched),
            "subagent_succeeded": float(env.subagent_succeeded),
            "search_calls": int(env.search_calls),
        }

    def _reward_for_agent(self, ag: AgentTrajectory, root_reward: float) -> float:
        if self.reward_mode == "team_credit":
            return root_reward
        if ag.reward is None:
            return 0.0
        return float(ag.reward)

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        n_samples = max(1, int(self.gconfig.n_samples))

        rollouts = await asyncio.gather(*[
            self._run_one_rollout(engine, data, i) for i in range(n_samples)
        ])

        root_rewards = [r["root_reward"] for r in rollouts]
        n_agents_list = [r["n_agents"] for r in rollouts]
        launched_total = sum(r["subagent_launched"] for r in rollouts)
        search_total = sum(r["search_calls"] for r in rollouts)

        for rw in root_rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=rw)
            stats_tracker.get(self.rollout_stat_scope).scalar(env_score=rw)
        for na in n_agents_list:
            stats_tracker.get(self.rollout_stat_scope).scalar(n_agents=na)
        if launched_total > 0:
            stats_tracker.get(self.rollout_stat_scope).scalar(
                subagent_launched_per_rollout=launched_total / max(1, len(rollouts)),
            )
        if search_total > 0:
            stats_tracker.get(self.rollout_stat_scope).scalar(
                search_calls_per_rollout=search_total / max(1, len(rollouts)),
            )

        if self.dump_dir is not None and self.dump_prob > 0 and random.random() < self.dump_prob:
            try:
                await self._dump_trajectory(rollouts[0], data)
            except Exception as e:
                logger.warning("dump_trajectory failed: %s", e)

        # Compute sample-weighted group mean over the n_samples ROOT rewards
        # (one value per sample, not per sequence) — fixes the sequence-count
        # weighting bias the producer would otherwise introduce.
        #
        # We stamp std = 1.0 to match platoon's mean-only centering at the
        # group level. The trainer's reward_norm formula is
        #     (reward - g_mean) / (g_std + eps)
        # with g_std=1.0 this reduces to mean-centering only — equivalent to
        # platoon's `train_data["rewards"] -= mean(task_reward)`. We could
        # also achieve this by changing reward_norm config to std_level=None,
        # but stamping std=1.0 keeps the trainer math centralized and lets
        # us flip back to std-normalization without touching the config.
        if len(root_rewards) >= 2:
            root_rewards_t = torch.tensor(root_rewards, dtype=torch.float32)
            g_mean = float(root_rewards_t.mean().item())
        elif len(root_rewards) == 1:
            g_mean = float(root_rewards[0])
        else:
            g_mean = 0.0
        g_std = 1.0  # disabled — match platoon's mean-only centering

        trajectories: list[dict[str, Any]] = []
        rewards_kept: list[float] = []
        for r in rollouts:
            for pa in r["per_agent"]:
                if not pa["sequences"]:
                    continue
                for seq in pa["sequences"]:
                    seq["group_reward_mean"] = torch.tensor([g_mean])
                    seq["group_reward_std"] = torch.tensor([g_std])
                trajectories.append({"sequences": pa["sequences"]})
                rewards_kept.append(pa["reward"])

        prompt_id = resolve_prompt_id(data) or (rollouts[0]["task"].id if rollouts else None)

        if not trajectories:
            return {
                "n_trajs": 0,
                "rewards": torch.tensor([], dtype=torch.float32),
                "trajectories": [],
                "prompt_id": prompt_id,
            }
        return {
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards_kept, dtype=torch.float32),
            "trajectories": trajectories,
            "prompt_id": prompt_id,
        }

    # ------------------------------------------------------------------ helpers

    def _task_from_data(self, data: dict[str, Any]) -> Task:
        if "task_id" in data:
            return get_task(data["task_id"])
        return Task(
            goal=str(data.get("question") or data.get("goal") or ""),
            id=str(data.get("id") or data.get("query_id") or uuid.uuid4().hex),
            max_steps=int(data.get("max_steps", self.max_steps_per_episode)),
            misc=dict(data),
        )

    async def _dump_trajectory(self, rollout: dict[str, Any], data: dict[str, Any]) -> None:
        if self.dump_dir is None:
            return
        await aiofiles.os.makedirs(self.dump_dir, exist_ok=True)
        task = rollout["task"]
        sub = "0"
        try:
            v = rollout["all_trajs"][0].turns[0][1].output_versions[0]
            sub = str(int(v))
        except Exception:
            pass
        out_dir = f"{self.dump_dir}/{sub}"
        await aiofiles.os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/deepdive-{abs(hash(task.id)) % 100_000_000:08d}.txt"
        ground_truth = str(task.misc.get("ground_truth", "")).strip()
        async with aiofiles.open(out_path, "w") as f:
            await f.write(
                f"=== Episode reward={rollout['root_reward']:.3f} "
                f"n_agents={rollout['n_agents']} "
                f"searches={rollout['search_calls']} ===\n"
                f"question: {task.goal}\n"
                f"ground_truth: {ground_truth}\n\n"
            )
            for ag in rollout["all_trajs"]:
                tag = "ROOT" if ag.is_root else f"SUB depth={ag.depth}"
                await f.write(
                    f"--- {tag} traj_id={ag.traj_id} task='{ag.task.goal[:120]}' "
                    f"per_agent_reward={ag.reward:.3f} bonus={ag.bonus:.3f} steps={len(ag.turns)} ---\n"
                )
                if ag.error_message:
                    await f.write(f"  (error: {ag.error_message})\n")
                if ag.finish_message:
                    await f.write(f"  finish_message: {ag.finish_message[:400]}\n")
                # B3: surface judge reasoning so dumps explain WHY a reward
                # was 0 vs 1 without needing to re-run the judge.
                if ag.eval_info:
                    reason = str(ag.eval_info.get("reason", ""))[:600]
                    if reason:
                        await f.write(f"  judge_reason: {reason}\n")
                if ag.messages:
                    for j, msg in enumerate(ag.messages):
                        body = msg["content"]
                        if len(body) > 2000:
                            body = body[:2000] + f"\n[...truncated, total {len(msg['content'])} chars...]"
                        await f.write(f"  [{j}] {msg['role']}:\n    {body}\n")
                await f.write("\n")
