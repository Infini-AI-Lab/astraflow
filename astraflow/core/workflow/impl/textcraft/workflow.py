"""Recursive agent workflow for TextCraft.

Design summary (see ``claude-doc/recursive-agent-textcraft-plan.md``):

- Multi-turn agent loop.  Each turn the model emits one ``<action type="...">{JSON}</action>``
  block.  Workflow parses, dispatches, returns an observation, appends to chat
  history, calls the model again — until ``finish`` or budget exhausted.
- Recursion: ``<action type="spawn">`` fans out up to ``max_breadth`` sub-agents
  in parallel via ``asyncio.gather``.  Each child is a full episode (its own
  agent + forked env that shares parent's inventory by reference).
- Aggregation: parent sees only each child's ``finish_message`` wrapped in a
  ``<spawn_result><sub_agent_i task="...">{msg}</sub_agent_i>…</spawn_result>``
  block (Option A — platoon-faithful, bounded context).
- Trajectory tree → flat list of sequences inside one trajectory.  Each agent
  (root + every descendant) contributes one sequence with loss_mask=1 on its
  own response tokens and 0 on its observations.  Team reward broadcast.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from collections.abc import Callable
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
from astraflow.core.workflow.impl.textcraft.env import TextCraftEnv, get_default_recipe_db
from astraflow.core.workflow.impl.textcraft.task import Task
from astraflow.core.workflow.impl.textcraft.tasks import get_task

logger = logging.getLogger("RecursiveAgent")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MAIN_SYSTEM_PROMPT = """You are a TextCraft agent. Your goal is to craft target items using a shared inventory and a recipe database.

You act by emitting EXACTLY ONE action block per turn, in this format:
  <action type="ACTION_NAME">{JSON args}</action>

After each action you will receive an observation as the next user message. Use it to decide your next action.

Available actions:

- get_info  — Query recipe database for one or more items.
    <action type="get_info">{"items": ["stick", "oak_planks"]}</action>

- view_inventory  — Read current inventory.
    <action type="view_inventory">{}</action>

- craft  — Consume ingredients, produce target. target must be [item_name, total_count]; total_count must be divisible by the recipe's result_count. Ingredients must match the recipe exactly (no extras).
    <action type="craft">{"ingredients": {"oak_log": 1}, "target": ["oak_planks", 4]}</action>

- spawn  — Dispatch 1-4 sub-agents in PARALLEL. Each shares your inventory by reference (their crafts affect you). Use this to delegate independent sub-goals (e.g. crafting different intermediates).
    <action type="spawn">{"subtasks": [
      {"targets": {"oak_planks": 16}, "max_steps": 8},
      {"targets": {"stick": 8}, "max_steps": 5}
    ]}</action>

- finish  — End your episode with a brief summary. After finish, no more actions can be taken.
    <action type="finish">{"message": "crafted 4 wooden_pickaxe"}</action>

You share a step budget with any sub-agents you spawn. Be efficient.
"""

SUB_SYSTEM_PROMPT = """You are a TextCraft sub-agent dispatched by a parent agent to complete a focused sub-task. You share the parent's inventory: items you craft become available to the parent.

You act by emitting EXACTLY ONE action block per turn (same format as the parent):
  <action type="ACTION_NAME">{JSON args}</action>

After each action you will receive an observation as the next user message. Use it to decide your next action.

Available actions:

- get_info  — Query recipe database for one or more items.
    <action type="get_info">{"items": ["stick", "oak_planks"]}</action>

- view_inventory  — Read current inventory.
    <action type="view_inventory">{}</action>

- craft  — Consume ingredients, produce target. target must be [item_name, total_count]; total_count must be divisible by the recipe's result_count. Ingredients must match the recipe exactly (no extras).
    <action type="craft">{"ingredients": {"oak_log": 1}, "target": ["oak_planks", 4]}</action>

- spawn  — You MAY recurse and spawn your own sub-agents (depth-limited). Same syntax as the parent.

- finish  — End your episode with a CONCISE summary message; this message is the only thing the parent will see about your work. Be informative but brief.
    <action type="finish">{"message": "crafted 16 oak_planks"}</action>

You have a tight step budget. Solve your sub-task directly when possible.
"""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

# Match the first <action type="X">{...}</action> block. DOTALL so JSON
# args may span lines.  Lazy match on the JSON body to stop at the first
# matching </action> (handles trailing model text after the block).
_ACTION_RE = re.compile(
    r"<action\s+type=\"(\w+)\"\s*>\s*(\{.*?\})\s*</action>",
    re.DOTALL,
)


@dataclass
class ParsedAction:
    type: str
    args: dict[str, Any]
    error: str | None = None
    raw_text: str = ""


def parse_action(text: str) -> ParsedAction:
    """Extract the first <action type="..."]>{json}</action> block."""
    m = _ACTION_RE.search(text)
    if not m:
        return ParsedAction(
            type="__noaction__",
            args={},
            error="no <action type=\"...\">{...}</action> block found in response",
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
    """Return None if valid, else an error string."""
    if action.error is not None:
        return action.error
    if action.type not in {"get_info", "view_inventory", "craft", "spawn", "finish"}:
        return f"unknown action type: {action.type!r}"
    if action.type == "get_info":
        items = action.args.get("items")
        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
            return "get_info: 'items' must be a list of strings"
    elif action.type == "craft":
        ing = action.args.get("ingredients")
        tgt = action.args.get("target")
        if not isinstance(ing, dict):
            return "craft: 'ingredients' must be a JSON object"
        if not (isinstance(tgt, list) and len(tgt) == 2 and isinstance(tgt[0], str) and isinstance(tgt[1], int)):
            return "craft: 'target' must be [item_name: str, count: int]"
    elif action.type == "spawn":
        subs = action.args.get("subtasks")
        if not isinstance(subs, list) or not subs:
            return "spawn: 'subtasks' must be a non-empty list"
        for i, s in enumerate(subs):
            if not isinstance(s, dict) or not isinstance(s.get("targets"), dict):
                return f"spawn: subtasks[{i}] needs a 'targets' dict"
            if not isinstance(s.get("max_steps", 0), int):
                return f"spawn: subtasks[{i}].max_steps must be int"
    elif action.type == "finish":
        if not isinstance(action.args.get("message", ""), str):
            return "finish: 'message' must be a string"
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
    # Sequence of (turn input_ids, ModelResponse) pairs for this agent's
    # own steps. We rebuild the chat history each turn so we don't have to
    # track per-turn input growth here.
    turns: list[tuple[list[int], ModelResponse]] = field(default_factory=list)
    finish_message: str | None = None
    error_message: str | None = None
    # Final per-agent reward (set after env.evaluate at end of episode).
    reward: float = 0.0
    # Final chat history (system + user + per-turn assistant/observation). Used
    # for rollout dumps. Stored as a list of {"role": str, "content": str}.
    messages: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@register_workflow("recursive_agent")
class RecursiveAgentWorkflow(RolloutWorkflow):
    """Multi-turn recursive agent on a stateful in-process env.

    For TextCraft: see ``astraflow/core/workflow/impl/textcraft/env.py``.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        reward_fn: Callable[..., Any] | str | None = None,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        max_depth: int = 3,
        max_breadth: int = 4,
        max_steps_per_episode: int = 50,
        max_concurrent_subagents: int = 8,
        delegation_reward_cap: float = 0.0,
        depth_level_weighting: bool = True,
        dump_prob: float = 1 / 128,
        parse_error_observation: str = (
            "ERROR: could not parse your response. Reply with a single "
            "<action type=\"...\">{...}</action> block."
        ),
    ):
        # reward_fn is OPTIONAL — recursive_agent computes reward from
        # env.evaluate() directly (rule-based, no LLM verifier).  We accept
        # the kwarg for API parity with other workflows but don't require it.
        self.reward_fn = reward_fn
        if isinstance(tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.async_reward_fn = None
        if reward_fn is not None and not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        self.max_depth = int(max_depth)
        self.max_breadth = int(max_breadth)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.max_concurrent_subagents = int(max_concurrent_subagents)
        self.delegation_reward_cap = float(delegation_reward_cap)
        self.depth_level_weighting = bool(depth_level_weighting)
        self.dump_prob = float(dump_prob)
        self.parse_error_observation = parse_error_observation

        # Eager-load the recipe DB so the first episode doesn't pay the
        # 860-recipe parse cost.
        get_default_recipe_db()

    # ------------------------------------------------------------------ chat

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool) -> list[int]:
        try:
            return list(self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking,
            ))
        except TypeError:
            return list(self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            ))

    def _build_initial_messages(
        self, task: Task, env: TextCraftEnv, is_root: bool
    ) -> list[dict]:
        system = MAIN_SYSTEM_PROMPT if is_root else SUB_SYSTEM_PROMPT
        target_str = ", ".join(f"{c}x {it}" for it, c in (task.misc.get("target_items") or {}).items())
        user = (
            f"Task: {task.goal or 'craft target items'}\n"
            f"Targets: {target_str or '(none)'}\n"
            f"Initial inventory: {env.view_inventory()}\n"
            f"Step budget: {task.max_steps}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------------ episode

    async def _run_episode(
        self,
        engine: InferenceEngine,
        env: TextCraftEnv,
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

        messages = self._build_initial_messages(task, env, is_root)
        steps_taken = 0
        max_local_steps = task.max_steps if task.max_steps is not None else self.max_steps_per_episode

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

            action = parse_action(response_text)
            err = validate_action(action)
            if err is not None:
                obs = f"ERROR: {err}"
                stats_tracker.get(self.rollout_stat_scope).scalar(parse_errors=1)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": obs})
                steps_taken += 1
                continue

            # Dispatch.
            if action.type == "get_info":
                obs = env.get_info(action.args["items"])
            elif action.type == "view_inventory":
                obs = env.view_inventory()
            elif action.type == "craft":
                obs = env.craft(action.args["ingredients"], action.args["target"])
            elif action.type == "finish":
                env.finish(action.args.get("message", ""))
                traj.finish_message = action.args.get("message", "")
                # Per-agent reward: evaluate THIS agent's env against ITS
                # own target_items at the moment of finish(). Inventory is
                # shared with the parent by reference, so this captures the
                # snapshot at this finish — matches platoon's per-agent
                # reward semantics (each agent scored on its own subtask).
                score, _info = env.evaluate()
                traj.reward = float(score)
                messages.append({"role": "assistant", "content": response_text})
                # No observation appended — episode terminates.
                steps_taken += 1
                break
            elif action.type == "spawn":
                if depth >= self.max_depth:
                    obs = f"ERROR: max recursion depth ({self.max_depth}) reached; cannot spawn"
                else:
                    subs = action.args["subtasks"][: self.max_breadth]
                    obs = await self._dispatch_spawn(
                        engine, env, budget, sem, all_trajs, traj_id, depth, subs,
                    )
            else:
                obs = f"ERROR: unhandled action type {action.type!r}"

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": obs})
            steps_taken += 1

        # Save full chat history (system + initial user + per-turn pairs) for
        # rollout dumps. May or may not end with a trailing user observation
        # depending on whether the agent finished cleanly.
        traj.messages = messages
        return traj

    async def _dispatch_spawn(
        self,
        engine: InferenceEngine,
        parent_env: TextCraftEnv,
        budget: BudgetTracker,
        sem: asyncio.Semaphore,
        all_trajs: list[AgentTrajectory],
        parent_id: str,
        parent_depth: int,
        subs: list[dict[str, Any]],
    ) -> str:
        """Run N children in parallel; return finish-message-only spawn_result."""
        parent_env.subagent_launched += len(subs)
        child_tasks: list[Task] = []
        for i, s in enumerate(subs):
            targets = s.get("targets", {})
            max_steps = int(s.get("max_steps", 10))
            context = s.get("context", "")
            target_str = ", ".join(f"{c}x {it}" for it, c in targets.items())
            goal = f"Craft the following items: {target_str}"
            if context:
                goal += f"\n\nContext from parent: {context}"
            child_tasks.append(Task(
                goal=goal,
                id=f"{parent_id}/sub_{i}",
                max_steps=max_steps,
                misc={
                    "target_items": targets,
                    "initial_inventory": parent_env.inventory,  # ALIASED
                },
            ))

        async def _one(child_task: Task) -> AgentTrajectory:
            async with sem:
                child_env = parent_env.fork(child_task)
                return await self._run_episode(
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

        children = await asyncio.gather(*[_one(t) for t in child_tasks])

        # Telemetry: count how many children produced a finish_message (proxy
        # for "succeeded at terminating cleanly").  Real per-child task
        # success is checked in evaluate() at episode end.
        for child in children:
            if child.finish_message:
                # Per-child success on its own targets.
                child_targets: dict[str, int] = child.task.misc.get("target_items") or {}
                ok = all(parent_env.inventory.get(it, 0) >= c for it, c in child_targets.items())
                parent_env.subagent_succeeded += 1.0 if ok else 0.0

        # Format observation: finish_message-only (Option A).
        parts = ["<spawn_result>"]
        for i, (child, sub) in enumerate(zip(children, subs)):
            target_str = ", ".join(f"{c}x {it}" for it, c in sub["targets"].items())
            msg = child.finish_message or (child.error_message or "[no finish_message]")
            parts.append(f"<sub_agent_{i} task=\"craft {target_str}\">{msg}</sub_agent_{i}>")
        parts.append("</spawn_result>")
        return "\n".join(parts)

    # ------------------------------------------------------------------ entry

    async def _run_one_rollout(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
        rollout_idx: int,
    ) -> dict[str, Any]:
        """Run one independent rollout (one root + its full spawn tree).

        Returns per-agent training data. Each agent (root or sub-agent)
        carries its OWN reward (computed at finish() time in _run_episode
        against THAT agent's own target_items). Agents that never finished
        (budget exhausted) keep reward=0.0.

        ``per_agent`` is a list of ``{"reward": float, "sequences": [...]}
        — one entry per agent that produced training data.
        ``root_score`` is the root agent's reward (for stats).
        """
        task = self._task_from_data(data)
        env = TextCraftEnv(task=task)
        budget = BudgetTracker(total=task.max_steps or self.max_steps_per_episode)
        sem = asyncio.Semaphore(self.max_concurrent_subagents)
        all_trajs: list[AgentTrajectory] = []

        await self._run_episode(
            engine=engine, env=env, task=task, budget=budget, sem=sem,
            all_trajs=all_trajs, parent_id=None, depth=0, is_root=True,
        )

        # Per-agent rewards are already stored on each AgentTrajectory by
        # _run_episode at finish() time. Build per-agent sequences here.
        per_agent: list[dict[str, Any]] = []
        for ag in all_trajs:
            if not ag.turns:
                continue
            seqs = self._build_sequences_for_agent(ag, ag.reward)
            per_agent.append({"reward": ag.reward, "sequences": seqs, "depth": ag.depth, "is_root": ag.is_root})

        root_traj = all_trajs[0] if all_trajs else None
        return {
            "per_agent": per_agent,
            "all_trajs": all_trajs,
            "task": task,
            "root_reward": float(root_traj.reward) if root_traj else 0.0,
            "n_agents": len(all_trajs),
            "subagent_launched": int(env.subagent_launched),
            "subagent_succeeded": float(env.subagent_succeeded),
        }

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run K=n_samples rollouts of one prompt in parallel.

        Returns ``n_trajs=K`` with one trajectory per rollout. Each
        trajectory's sequences share that rollout's root reward. This shape
        lets ``filter_zero_adv`` see the across-sibling variance and lets
        GRPO compute per-prompt advantages — matching the pattern used by
        ``rlvr`` and other workflows.
        """
        # Optional dynamic import for reward_fn — kept for API parity even
        # though we don't call it (reward is env.evaluate()-based).
        if isinstance(self.reward_fn, str):
            try:
                self.reward_fn = import_from_string(self.reward_fn)
                self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)
            except Exception as e:
                logger.warning("could not import reward_fn=%r: %s — proceeding (env reward only)", self.reward_fn, e)
                self.reward_fn = None

        n_samples = max(1, int(self.gconfig.n_samples))

        # Run all K rollouts in parallel. Each rollout is independent: its
        # own env, budget, semaphore, trajectory tree.
        rollouts = await asyncio.gather(*[
            self._run_one_rollout(engine, data, i) for i in range(n_samples)
        ])

        # Aggregate stats so wandb/StatsLogger see the group.
        root_rewards = [r["root_reward"] for r in rollouts]
        n_agents_list = [r["n_agents"] for r in rollouts]
        launched_total = sum(r["subagent_launched"] for r in rollouts)
        succeeded_total = sum(r["subagent_succeeded"] for r in rollouts)
        # Per-agent rewards across all agents in all rollouts (for diagnostics).
        all_agent_rewards = [
            pa["reward"] for r in rollouts for pa in r["per_agent"]
        ]
        sub_agent_rewards = [
            pa["reward"] for r in rollouts for pa in r["per_agent"] if not pa["is_root"]
        ]

        for rw in root_rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=rw)
            stats_tracker.get(self.rollout_stat_scope).scalar(env_score=rw)
        for na in n_agents_list:
            stats_tracker.get(self.rollout_stat_scope).scalar(n_agents=na)
        for sr in sub_agent_rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(subagent_reward=sr)
        if launched_total > 0:
            stats_tracker.get(self.rollout_stat_scope).scalar(
                subagent_success_rate=succeeded_total / launched_total,
            )

        # Optional dump: one rollout sampled per group call (avoids K× volume).
        if self.dump_dir is not None and random.random() < self.dump_prob:
            r0 = rollouts[0]
            await self._dump_trajectory(
                r0["all_trajs"], r0["root_reward"], r0["root_reward"],
                {"subagent_launched": r0["subagent_launched"], "subagent_succeeded": r0["subagent_succeeded"]},
                data,
            )

        # Build the structured result. Each AGENT (root or sub-agent) across
        # ALL K rollouts becomes its own trajectory with its OWN reward —
        # platoon's per-agent reward semantics. AstraFlow's GRPO then
        # normalizes advantages across all trajectories sharing this prompt_id.
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
        """Materialize a Task object from a dataset row.

        Supported shapes:
          1. ``{"task_id": "textcraft.train.42"}``  → loads from disk via tasks.py
          2. inline: ``{"target_items": {...}, "initial_inventory": {...}, "max_steps": N, "id": ...}``
        """
        if "task_id" in data:
            return get_task(data["task_id"])
        return Task(
            goal=data.get("goal") or "Craft the target items",
            id=data.get("id") or data.get("query_id") or uuid.uuid4().hex,
            max_steps=int(data.get("max_steps", self.max_steps_per_episode)),
            misc={
                "target_items": data.get("target_items", {}),
                "initial_inventory": data.get("initial_inventory", {}),
            },
        )

    def _build_sequences_for_agent(
        self, ag: AgentTrajectory, reward: float
    ) -> list[dict[str, torch.Tensor]]:
        """Emit one training sequence per turn of this agent.

        Per-turn layout::

            [turn_input_ids (loss_mask=0)] [response_tokens (loss_mask=1)]

        We deliberately do NOT try to concatenate turns into one long
        sequence.  Doing so requires reconciling each turn's ``input_ids``
        (built by re-applying the chat template after appending the prior
        assistant message + new observation) with the cumulative
        prior_input + prior_response_tokens — but SGLang's actual
        ``resp.output_tokens`` is one of many tokenizations whose decoded
        text matches the assistant message; the chat-template re-encode
        often picks a different tokenization due to BPE non-uniqueness +
        special-token handling.  That breaks any "next turn starts with
        cumulative-so-far" invariant.

        Treating each turn as an independent sequence sidesteps this:
        the loss is computed against the actual context the model saw
        (turn_input_ids) and the actual tokens it generated
        (resp.output_tokens).  All per-turn sequences for this agent
        share the same trajectory reward (with depth-level weighting if
        enabled).
        """
        # Per-agent reward weighting.
        weight = 1.0 / (ag.depth + 1) if self.depth_level_weighting else 1.0
        per_seq_reward = reward * weight

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
                "rewards": torch.tensor(per_seq_reward, dtype=torch.float32),
            }
            out.append({k: v.unsqueeze(0) for k, v in seq.items()})
        return out

    async def _dump_trajectory(
        self,
        all_trajs: list[AgentTrajectory],
        reward: float,
        score: float,
        info: dict,
        data: dict[str, Any],
    ) -> None:
        if self.dump_dir is None:
            return
        # Pick the engine version from any trajectory's first turn (best-effort).
        version = 0
        if all_trajs and all_trajs[0].turns:
            v = all_trajs[0].turns[0][1].output_versions
            if v:
                version = max(v)
        dump_path = os.path.join(self.dump_dir, str(version))
        await aiofiles.os.makedirs(dump_path, exist_ok=True)
        qid = resolve_prompt_id(data) or (all_trajs[0].task.id if all_trajs else uuid.uuid4().hex)
        file_path = os.path.join(dump_path, f"{qid}.txt")
        async with aiofiles.open(file_path, "a") as f:
            await f.write(f"=== Episode reward={reward} env_score={score} n_agents={len(all_trajs)} info={info} ===\n\n")
            for ag in all_trajs:
                kind = "ROOT" if ag.is_root else f"SUB depth={ag.depth} parent={ag.parent_id}"
                await f.write(f"--- {kind} traj_id={ag.traj_id} task={ag.task.goal!r} target_items={ag.task.misc.get('target_items')} per_agent_reward={ag.reward} steps={len(ag.turns)} ---\n")
                if ag.finish_message:
                    await f.write(f"FINISH: {ag.finish_message}\n")
                if ag.error_message:
                    await f.write(f"ERROR:  {ag.error_message}\n")
                # Full chat history: system prompt + initial user (task+targets+inventory) + per-turn (assistant + user observation).
                # Falls back to legacy per-turn output dump if messages wasn't populated (e.g., agent never ran).
                if ag.messages:
                    for j, msg in enumerate(ag.messages):
                        await f.write(f"  [{j}] {msg['role']}:\n    {msg['content']}\n")
                else:
                    for i, (_input_ids, resp) in enumerate(ag.turns):
                        await f.write(f"  turn {i} output:\n    {self.tokenizer.decode(resp.output_tokens)[:600]}\n")
                await f.write("\n")
