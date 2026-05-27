"""Oolong recursive-agent workflow.

Design (per Gandhi et al. 2026, "Recursive Agent Optimization",
arxiv 2605.06639):

- Multi-turn agent loop.  Each turn the model emits one
  ``<thought>...</thought><python>...</python>`` block.  The Python code
  is executed in a stateful sandbox (per-agent ``exec()`` namespace).
- Tools exposed via the sandbox:
    - ``context`` (str): pre-populated with the full input text
    - ``finish(answer)`` -> raises FinishSignal, ending the agent
    - ``await launch_subagent(goal, context)`` -> spawns a child agent
    - ``asyncio`` module: for ``asyncio.gather(...)`` parallel spawns
- Aggregation: parent's ``launch_subagent`` returns the child's
  ``finish(...)`` payload as a string.  No inventory aliasing; sub-agents
  process whatever chunk the parent passes.
- Reward (delegation-bonus):
      R(X) = success(X) + lambda * mean(success(children))
  Default lambda = 0.4 (paper's choice for OOLONG-REAL).  Computed in a
  post-pass over the trajectory tree before per-agent rewards are emitted
  to the buffer.
"""

from __future__ import annotations

import asyncio
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
from astraflow.core.workflow.impl.oolong.env import (
    DEFAULT_STDOUT_TRUNCATE,
    OolongEnv,
)
from astraflow.core.workflow.impl.oolong.tasks import Task, get_task

logger = logging.getLogger("OolongRecursive")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
# Both root and sub use the same recursive-agent prompt format.  The prompt
# is taken from platoon's OolongRecursivePromptBuilder; we drop the
# "include_reasoning" branch since we always require <thought>...</thought>.

MAIN_SYSTEM_PROMPT = """You are tasked with answering a query that requires analyzing and aggregating information from a large context.

You have access to a REPL environment with the following pre-loaded variable:
- `context` (str): The full text context to analyze (may be very large)

<TIPS>
CONTEXT ANALYSIS:
- First check if the length of the context is very large (>32K characters) using `len(context)`.
- For very large contexts (>32K characters), work with chunks rather than the entire context at once.
- Use subagents to process chunks and then aggregate the results to produce a final answer. Try not to split the context into too many chunks (32K characters per chunk is a good rule of thumb).
- If the context <= 32K characters, prefer to process your context by printing out and reading it rather than using programmatic heuristics.
- **IMPORTANT: DO NOT USE regex, string matching, etc. types of programmatic heuristics. Read the context with `print(context)` to be accurate in your answer.**

SUBAGENT DELEGATION:
- **Do not use subagents if the context you need to process is <= 32K characters.** Just print out the context to observe it directly and answer the question by reading the context.
- You have the ability to spawn subagents (other instantiations of yourself), by providing them with their own `context`/chunk to process and a goal/instruction for what result it should return.
- You can use `await asyncio.gather(...)` to process multiple chunks simultaneously.
- Be specific about the format and type in which you expect subagents to return their results.
- Do not provide the context/chunk as part of the goal. Instead, pass it explicitly as the `context` argument to the `launch_subagent` function.

ANSWER SUBMISSION:
- You can submit your answer using the `finish` function in the format requested in the user-provided goal.
</TIPS>

You can perform printing out, peeking into the context, or launching subagents using Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your recursive strategy in <thought> </thought> tags, then output your code in <python> </python> tags.
Your code will be executed in a Jupyter-like environment and the output will be shown to you. The python code block should be formatted as follows: <python>code block</python> without any other tags.
Do not output anything else except for <thought>...</thought>\n<python>...</python>
"""

SUB_SYSTEM_PROMPT = MAIN_SYSTEM_PROMPT  # same prompt for now; differs by initial user message


# ---------------------------------------------------------------------------
# Code-block parsing
# ---------------------------------------------------------------------------

# DOTALL so code can span lines.  Lazy match on the body to stop at the
# first matching </python> tag.
_PY_RE = re.compile(r"<python>(.*?)</python>", re.DOTALL)
_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)


@dataclass
class ParsedCode:
    code: str
    thought: str = ""
    error: str | None = None
    raw_text: str = ""


def parse_code(text: str) -> ParsedCode:
    m = _PY_RE.search(text)
    if not m:
        return ParsedCode(
            code="",
            error="no <python>...</python> block found in response",
            raw_text=text,
        )
    code = m.group(1)
    tm = _THOUGHT_RE.search(text)
    thought = tm.group(1).strip() if tm else ""
    return ParsedCode(code=code, thought=thought, raw_text=m.group(0))


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


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@register_workflow("oolong_recursive")
class OolongRecursiveWorkflow(RolloutWorkflow):
    """Recursive-agent workflow for Oolong long-context aggregation."""

    def __init__(
        self,
        reward_fn: str | Any = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
        gconfig: GenerationHyperparameters | None = None,
        # Task-specific knobs (matches platoon defaults for OOLONG-REAL).
        max_depth: int = 2,                       # 0-indexed; 3 levels incl. root
        max_breadth: int = 8,                     # safety cap on children per spawn
        max_steps_per_episode: int = 50,
        max_concurrent_subagents: int = 8,        # RaaS queue bound
        delegation_lambda: float = 0.4,           # paper default for OOLONG-REAL
        sub_max_steps: int = 25,                  # sub-agent's own step cap
        stdout_truncate: int = DEFAULT_STDOUT_TRUNCATE,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        dump_prob: float = 0.0,
        **kwargs: Any,
    ):
        # Resolve reward_fn for API parity with other workflows (we
        # actually use env.evaluate() for the reward signal, not this fn).
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
        self.max_breadth = max_breadth
        self.max_steps_per_episode = max_steps_per_episode
        self.max_concurrent_subagents = max_concurrent_subagents
        self.delegation_lambda = float(delegation_lambda)
        self.sub_max_steps = sub_max_steps
        self.stdout_truncate = stdout_truncate
        self.enable_thinking = bool(enable_thinking)
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.dump_prob = float(dump_prob)

    # ------------------------------------------------------------------ utils

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("OolongRecursiveWorkflow has no tokenizer attached")
        kwargs: dict[str, Any] = dict(
            add_generation_prompt=add_generation_prompt,
            return_tensors=None,
        )
        # Qwen3-Instruct supports a thinking-mode toggle; default off for parity.
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)

    def _build_initial_messages(self, task: Task, is_root: bool) -> list[dict]:
        system = MAIN_SYSTEM_PROMPT if is_root else SUB_SYSTEM_PROMPT
        # Show the agent only the goal; the context is pre-loaded into the
        # sandbox as `context` and can be inspected via `print(context[:N])`
        # or `len(context)`.
        user = f"Goal: {task.goal}\n"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _format_observation(self, code_result, parse_error: str | None) -> str:
        if parse_error is not None:
            return f"ERROR: {parse_error}"
        parts = []
        if code_result.stdout:
            parts.append(code_result.stdout)
        if code_result.error:
            parts.append(f"\n[error]\n{code_result.error}")
        elif code_result.stderr:
            parts.append(f"\n[stderr]\n{code_result.stderr}")
        if not parts:
            parts.append("(no output)")
        return "".join(parts)

    # ------------------------------------------------------------------ episode

    async def _run_episode(
        self,
        engine: InferenceEngine,
        env: OolongEnv,
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

            parsed = parse_code(response_text)
            if parsed.error is not None:
                obs = f"ERROR: {parsed.error}"
                stats_tracker.get(self.rollout_stat_scope).scalar(parse_errors=1)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": obs})
                steps_taken += 1
                continue

            code_result = await env.run_code(parsed.code)
            obs = self._format_observation(code_result, parse_error=None)
            messages.append({"role": "assistant", "content": response_text})

            if env.finished:
                # No observation after finish — the episode terminates.
                traj.finish_payload = env.finish_payload
                traj.finish_message = "" if env.finish_payload is None else str(env.finish_payload)
                score, _info = env.evaluate()
                traj.reward = float(score)
                steps_taken += 1
                break

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
        """Returns an async fn the env can call to launch one sub-agent.

        Each call spawns exactly one child and returns its finish_message
        as a string.  The agent can wrap multiple calls in asyncio.gather()
        to parallelize.
        """
        async def _launch(goal: str, context: str) -> str:
            if parent_depth >= self.max_depth:
                return f"ERROR: max recursion depth ({self.max_depth}) reached; cannot spawn"
            # Build a fresh child task.
            child_task = Task(
                goal=str(goal),
                id=f"{parent_id}/sub_{uuid.uuid4().hex[:8]}",
                max_steps=self.sub_max_steps,
                misc={"context": str(context)},
            )
            child_env = OolongEnv(
                task=child_task,
                spawn_callback=self._make_spawn_callback(
                    engine, budget, sem, all_trajs, child_task.id, parent_depth + 1
                ),
                stdout_truncate=self.stdout_truncate,
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
        """Add lambda * mean(children's success) to each agent's reward.

        Mutates each AgentTrajectory.reward in place. Eq. 1 of arxiv
        2605.06639. lambda=0 disables (pure per-agent reward).
        """
        if self.delegation_lambda <= 0:
            return
        # Build children index by parent_id.
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
        """Emit one training sequence per turn of this agent.

        Per-turn layout: [turn_input_ids (loss_mask=0)] [response_tokens (loss_mask=1)].
        Matches TextCraft's approach (see textcraft/workflow.py:_build_sequences_for_agent
        for the explanation of why per-turn rather than concatenated).
        """
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

        root_id = task.id  # used as parent_id for top-level children
        env = OolongEnv(
            task=task,
            spawn_callback=self._make_spawn_callback(
                engine, budget, sem, all_trajs, root_id, parent_depth=0
            ),
            stdout_truncate=self.stdout_truncate,
        )

        await self._run_episode(
            engine=engine, env=env, task=task, budget=budget, sem=sem,
            all_trajs=all_trajs, parent_id=None, depth=0, is_root=True,
        )

        # Apply delegation bonus across the tree.
        self._apply_delegation_bonus(all_trajs)

        # Root-only training: emit ONLY the root agent's sequences for
        # PPO. v2 (team-credit, bs=64) and v3 (team-credit, bs=256) both
        # showed pre_filter degrading from 0.6-0.8 → 0.2-0.4 in <10 steps.
        # Root tokens already include `await launch_subagent(...)` calls,
        # so the model still learns to spawn from the root's gradient.
        # The sub-agent's own generations don't get a gradient — they need
        # a per-segment verifier (LLM judge) to train usefully, which we
        # defer per [[spawn-subagent-credit]].
        root_traj = all_trajs[0] if all_trajs else None
        team_reward = float(root_traj.reward) if root_traj else 0.0

        per_agent: list[dict[str, Any]] = []
        for ag in all_trajs:
            if not ag.turns or not ag.is_root:
                continue
            seqs = self._build_sequences_for_agent(ag, team_reward)
            per_agent.append({
                "reward": team_reward,
                "sequences": seqs,
                "depth": ag.depth,
                "is_root": ag.is_root,
            })

        return {
            "per_agent": per_agent,
            "all_trajs": all_trajs,
            "task": task,
            "root_reward": team_reward,
            "n_agents": len(all_trajs),
            "subagent_launched": int(env.subagent_launched),
            "subagent_succeeded": float(env.subagent_succeeded),
        }

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        n_samples = max(1, int(self.gconfig.n_samples))

        rollouts = await asyncio.gather(*[
            self._run_one_rollout(engine, data, i) for i in range(n_samples)
        ])

        # Aggregate metrics.
        root_rewards = [r["root_reward"] for r in rollouts]
        n_agents_list = [r["n_agents"] for r in rollouts]
        launched_total = sum(r["subagent_launched"] for r in rollouts)

        for rw in root_rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=rw)
            stats_tracker.get(self.rollout_stat_scope).scalar(env_score=rw)
        for na in n_agents_list:
            stats_tracker.get(self.rollout_stat_scope).scalar(n_agents=na)
        if launched_total > 0:
            stats_tracker.get(self.rollout_stat_scope).scalar(
                subagent_launched_per_rollout=launched_total / max(1, len(rollouts)),
            )

        # Optional dump (one rollout per group call).
        if self.dump_dir is not None and self.dump_prob > 0 and random.random() < self.dump_prob:
            try:
                await self._dump_trajectory(rollouts[0], data)
            except Exception as e:
                logger.warning("dump_trajectory failed: %s", e)

        # Flatten to trajectories.
        trajectories: list[dict[str, Any]] = []
        rewards_kept: list[float] = []
        for r in rollouts:
            for pa in r["per_agent"]:
                if not pa["sequences"]:
                    continue
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
        """Build a Task from a dataset row.  Two supported shapes:
        (1) `{"task_id": "oolong.synth.validation.42"}` -> loads via tasks.get_task
        (2) inline: `{"context": ..., "question": ..., "answer": ..., ...}` (rare).
        """
        if "task_id" in data:
            return get_task(data["task_id"])
        return Task(
            goal=str(data.get("question") or data.get("goal") or ""),
            id=str(data.get("id") or data.get("query_id") or uuid.uuid4().hex),
            max_steps=int(data.get("max_steps", self.max_steps_per_episode)),
            misc=dict(data),
        )

    async def _dump_trajectory(self, rollout: dict[str, Any], data: dict[str, Any]) -> None:
        """Append a human-readable dump of one rollout for offline inspection."""
        if self.dump_dir is None:
            return
        await aiofiles.os.makedirs(self.dump_dir, exist_ok=True)
        task = rollout["task"]
        # Subdir per weight version when present in agentic dumps.
        sub = "0"
        try:
            v = rollout["all_trajs"][0].turns[0][1].output_versions[0]
            sub = str(int(v))
        except Exception:
            pass
        out_dir = f"{self.dump_dir}/{sub}"
        await aiofiles.os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/oolong-{abs(hash(task.id)) % 100_000_000:08d}.txt"
        async with aiofiles.open(out_path, "w") as f:
            await f.write(
                f"=== Episode reward={rollout['root_reward']:.3f} n_agents={rollout['n_agents']} ===\n\n"
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
                if ag.messages:
                    for j, msg in enumerate(ag.messages):
                        body = msg["content"]
                        if len(body) > 2000:
                            body = body[:2000] + f"\n[...truncated, total {len(msg['content'])} chars...]"
                        await f.write(f"  [{j}] {msg['role']}:\n    {body}\n")
                await f.write("\n")
