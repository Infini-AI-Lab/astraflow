"""Spawn-sub-agents math workflow.

The main agent generates as usual, but may emit a single tool call of the
form ``<spawn>{"tasks": ["...", "..."]}</spawn>`` mid-generation. When that
happens the workflow:

  1. Halts the main agent at ``</spawn>`` (using a string-level stop).
  2. Parses the JSON payload, caps the task list at ``max_sub_agents``.
  3. Fans out ``len(tasks)`` independent sub-agent generations against the
     same RaaS pool via ``asyncio.gather`` — each sub-agent sees a fixed
     system prompt plus ``{original_problem}\\n\\n{task}``.
  4. Concatenates the sub-agent outputs into a ``<spawn_result>...`` block,
     appends it to the main agent's context, and continues main generation.
  5. Computes ``reward_fn`` on the main agent's final answer.

Training scheme (shared-reward, multi-sequence trajectory):
  - One trajectory per episode-sample, containing 1 main + N sub-agent
    sequences. All inherit the trajectory reward via
    ``_ingest_structured_result``'s fallback.
  - Single-model regime: no ``model_ids`` tagging — every sequence routes
    to the same trainer. Main and sub-agents are the same policy.
  - GRPO/M2PO advantage normalization runs over n_samples × (1 + sub-count)
    sequences per prompt — credit-assignment is noisy by design (team
    reward), but every contributor's tokens get gradient.  See
    ``solve_and_check.py`` for the multi-sequence-shared-reward precedent.

Behavior when the main agent does not spawn (or emits a malformed payload):
  the workflow degrades to vanilla single-turn RLVR (1 main sequence, no
  sub-agents) — so the recipe stays valid for prompts that don't benefit.

Usage in YAML config::

    workflow_spec:
      workflow_cls: spawn_rlvr
      reward_fn: "math_verify"
      tokenizer: "Qwen/Qwen3-8B"
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from collections.abc import Callable
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

logger = logging.getLogger("Spawn workflow")


SPAWN_OPEN = "<spawn>"
SPAWN_CLOSE = "</spawn>"
SPAWN_RESULT_OPEN = "<spawn_result>"
SPAWN_RESULT_CLOSE = "</spawn_result>"

# Regex that matches a complete spawn block in the decoded phase-1 text.
# We do NOT use string-level SGLang stops here: SGLang in this repo runs
# with ``--skip-tokenizer-init`` and its scheduler has ``tokenizer=None``,
# so any ``stop=[...]`` argument crashes the scheduler in
# ``schedule_batch.py``'s ``_check_str_based_finish``.  Instead we let the
# main agent generate freely and detect the close tag after decode.
# DOTALL so the JSON payload may span lines.
_SPAWN_RE = re.compile(r"<spawn>\s*(\{.*?\})\s*</spawn>", re.DOTALL)

# System prompt prepended to the main agent so a fresh base model knows how
# to invoke the tool. Plain text — no chat-template surgery.
MAIN_SYSTEM_PROMPT = (
    "You are solving a math problem. Reason step by step and put your final "
    "answer in \\boxed{}.\n\n"
    "You have access to one tool: <spawn>. You may call it AT MOST ONCE in "
    "your response, to dispatch up to 4 sub-agents in parallel for "
    "independent sub-tasks (e.g. verifying a step, exploring an alternative "
    "approach, computing a hard sub-expression).\n\n"
    "To call the tool, emit exactly one block of the form:\n"
    "<spawn>{\"tasks\": [\"<task 1>\", \"<task 2>\", ...]}</spawn>\n"
    "After the </spawn> tag the system will pause your generation, run the "
    "sub-agents in parallel, and inject their outputs back into your "
    "context inside <spawn_result>...</spawn_result>. You then continue and "
    "produce the final answer.\n\n"
    "Use the tool only when sub-tasks would genuinely help. If not, just "
    "solve the problem directly."
)

# Fixed system prompt for every sub-agent.
SUB_SYSTEM_PROMPT = (
    "You are a sub-agent dispatched by a main reasoning agent to solve a "
    "focused sub-task that is part of a larger math problem. Solve the "
    "sub-task concisely. Return your reasoning followed by your answer in "
    "\\boxed{} when applicable. Do not call any tools."
)


def _extract_problem(messages: list[dict]) -> str:
    """Return the first user-turn content, used as the sub-agent's context."""
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


@register_workflow("spawn_rlvr")
class SpawnWorkflow(RolloutWorkflow):
    """RLVR-style workflow with a single ``<spawn>`` tool call per trajectory.

    Parameters
    ----------
    reward_fn : callable or str
        Reward function applied to the main agent's final answer.
    gconfig : GenerationHyperparameters
        Generation config for the main agent.  Sub-agents inherit it with
        n_samples=1 and ``sub_agent_max_new_tokens`` (default: main // 2).
    tokenizer : str or PreTrainedTokenizerFast
        Tokenizer.
    enable_thinking : bool
        Forwarded to the chat template (Qwen3 ``<think>`` tokens).
    max_sub_agents : int
        Hard cap on sub-agents per spawn call (extras dropped).
    sub_agent_max_new_tokens : int | None
        Override for sub-agent budget.  None → main's ``max_new_tokens // 2``.
    dump_prob : float
        Probability of dumping a fully-decoded trajectory (with phase-1
        output, every sub-agent task + output, and phase-2 output) to
        ``dump_dir/{version}/{qid}.txt`` for sanity-checking.  Default
        1/128 (matching rlvr); bump to 1.0 for smoke tests.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        max_sub_agents: int = 4,
        sub_agent_max_new_tokens: int | None = None,
        dump_prob: float = 1 / 128,
    ):
        self.reward_fn = reward_fn
        if isinstance(tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

        self.max_sub_agents = max_sub_agents
        self.dump_prob = float(dump_prob)
        if sub_agent_max_new_tokens is not None:
            self.sub_agent_max_new_tokens = sub_agent_max_new_tokens
        else:
            self.sub_agent_max_new_tokens = max(64, self.gconfig.max_new_tokens // 2)

    # ------------------------------------------------------------------ helpers

    def _apply_chat_template(self, messages, add_generation_prompt: bool = True):
        try:
            return list(self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking,
            ))
        except TypeError:
            return list(self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            ))

    def _build_main_input_ids(self, data: dict[str, Any]) -> list[int]:
        """Main agent sees the spawn-tool system prompt + the original messages."""
        messages = [{"role": "system", "content": MAIN_SYSTEM_PROMPT}]
        for m in data["messages"]:
            messages.append(m)
        return self._apply_chat_template(messages, add_generation_prompt=True)

    def _build_sub_input_ids(self, problem: str, task: str) -> list[int]:
        messages = [
            {"role": "system", "content": SUB_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original problem:\n{problem}\n\n"
                    f"Your sub-task:\n{task}\n\n"
                    "Solve the sub-task and return your reasoning."
                ),
            },
        ]
        return self._apply_chat_template(messages, add_generation_prompt=True)

    def _parse_spawn_payload(self, completion_text: str) -> list[str]:
        """Return the list of task strings, or [] if no usable payload."""
        m = _SPAWN_RE.search(completion_text)
        if not m:
            return []
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            logger.warning("spawn payload JSON decode failed; treating as no-spawn")
            return []
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            return []
        tasks = [str(t) for t in tasks if isinstance(t, (str, int, float))]
        if len(tasks) > self.max_sub_agents:
            logger.warning(
                "spawn payload had %d tasks; truncating to %d",
                len(tasks), self.max_sub_agents,
            )
            tasks = tasks[: self.max_sub_agents]
        return tasks

    def _format_spawn_result(self, sub_outputs: list[str]) -> str:
        parts = [SPAWN_RESULT_OPEN, ""]
        for i, out in enumerate(sub_outputs):
            parts.append(f"<sub_agent_{i}>")
            parts.append(out)
            parts.append(f"</sub_agent_{i}>")
        parts.append(SPAWN_RESULT_CLOSE)
        parts.append("")
        return "\n".join(parts)

    # ------------------------------------------------------------------ sub-agent

    async def _run_sub_agent(
        self,
        engine: InferenceEngine,
        problem: str,
        task: str,
    ) -> tuple[list[int], ModelResponse]:
        sub_input_ids = self._build_sub_input_ids(problem, task)
        sub_gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.sub_agent_max_new_tokens,
        )
        resp = await engine.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=sub_input_ids,
                gconfig=sub_gconfig,
                tokenizer=self.tokenizer,
            )
        )
        return sub_input_ids, resp

    # ------------------------------------------------------------------ episode

    async def _run_one_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
        """Returns (sequences, reward, debug_info).

        ``sequences`` is the list of per-sequence tensor dicts for this
        episode's trajectory (1 main + N sub-agents).  ``reward`` is the
        scalar math_verify reward on the main agent's final answer.
        """
        problem = _extract_problem(data["messages"])
        main_prompt_ids = self._build_main_input_ids(data)
        prompt_str = self.tokenizer.decode(main_prompt_ids)

        # ── Phase 1: main agent generates freely ──
        # No string-level ``stop`` (SGLang's scheduler runs with
        # tokenizer=None and crashes on stop-string matching).  We let
        # the model run to its natural end and detect <spawn>...</spawn>
        # post-hoc.  If a spawn is detected mid-generation, the tokens
        # after </spawn> are discarded for the trajectory.
        phase1_gconfig = self.gconfig.new(n_samples=1)
        resp1 = await engine.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=main_prompt_ids,
                gconfig=phase1_gconfig,
                tokenizer=self.tokenizer,
            )
        )
        phase1_text = self.tokenizer.decode(resp1.output_tokens)
        spawn_match = _SPAWN_RE.search(phase1_text)
        tasks = self._parse_spawn_payload(phase1_text) if spawn_match else []

        # ── No-spawn branch: degrade to vanilla single-turn RLVR ──
        if not tasks:
            reward, _ = await self._compute_reward(resp1, prompt_str, data)
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
            stats_tracker.get(self.rollout_stat_scope).scalar(num_sub_agents=0)
            main_seq = self._build_sequence(
                prompt_ids=main_prompt_ids,
                model_segments=[resp1],
                env_segments_after_prompt=[],
            )
            return [main_seq], reward, {
                "phase1_text": phase1_text, "spawned": False,
                "n_sub": 0, "phase2_text": "", "sub_outputs": [],
            }

        # ── Spawn branch: truncate phase-1 tokens at the close tag, then
        # fan out sub-agents in parallel ──
        # Re-encode the text up through </spawn> to find the token cutoff.
        # BPE boundary effects can shift the cutoff by ±1 token; that's
        # fine because the prefix text is identical and lands in
        # loss_mask=1 either way.
        phase1_prefix_text = phase1_text[: spawn_match.end()]
        phase1_prefix_ids = list(
            self.tokenizer.encode(phase1_prefix_text, add_special_tokens=False)
        )
        cutoff = min(len(phase1_prefix_ids), len(resp1.output_tokens))
        truncated_phase1_tokens = list(resp1.output_tokens[:cutoff])
        truncated_phase1_logprobs = list(resp1.output_logprobs[:cutoff])
        truncated_phase1_versions = list(resp1.output_versions[:cutoff])

        sub_results = await asyncio.gather(
            *[self._run_sub_agent(engine, problem, t) for t in tasks]
        )
        sub_outputs_text = [self.tokenizer.decode(r[1].output_tokens) for r in sub_results]

        # Build aggregated <spawn_result> block.  Tokenize without
        # special tokens — it's a continuation of the main agent's
        # context.
        agg_text = self._format_spawn_result(sub_outputs_text)
        agg_ids = list(self.tokenizer.encode(agg_text, add_special_tokens=False))

        # ── Phase 2: main agent continues with aggregated result appended ──
        phase2_prefix = main_prompt_ids + truncated_phase1_tokens + agg_ids
        phase2_gconfig = self.gconfig.new(n_samples=1)
        resp2 = await engine.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=phase2_prefix,
                gconfig=phase2_gconfig,
                tokenizer=self.tokenizer,
            )
        )
        phase2_text = self.tokenizer.decode(resp2.output_tokens)

        reward, _ = await self._compute_reward(resp2, prompt_str, data)
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
        stats_tracker.get(self.rollout_stat_scope).scalar(num_sub_agents=len(tasks))

        # ── Build main sequence (multi-segment) ──
        main_seq = self._build_main_sequence_with_spawn(
            prompt_ids=main_prompt_ids,
            phase1_tokens=truncated_phase1_tokens,
            phase1_logprobs=truncated_phase1_logprobs,
            phase1_versions=truncated_phase1_versions,
            agg_ids=agg_ids,
            resp2=resp2,
        )

        # ── Build sub-agent sequences (vanilla input/output layout) ──
        sub_seqs = []
        for sub_input_ids, sub_resp in sub_results:
            sub_seqs.append(
                self._build_sequence(
                    prompt_ids=sub_input_ids,
                    model_segments=[sub_resp],
                    env_segments_after_prompt=[],
                )
            )

        debug = {
            "phase1_text": phase1_text,
            "spawned": True,
            "n_sub": len(tasks),
            "tasks": tasks,
            "sub_outputs": sub_outputs_text,
            "phase2_text": phase2_text,
        }
        return [main_seq] + sub_seqs, reward, debug

    # ------------------------------------------------------------------ tensors

    def _build_sequence(
        self,
        prompt_ids: list[int],
        model_segments: list[ModelResponse],
        env_segments_after_prompt: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        """Single prompt + model output(s). Used for sub-agents and no-spawn main."""
        # For both the vanilla rlvr case and sub-agents, there's one model
        # segment immediately after the prompt and no env segments.
        assert (
            len(model_segments) == 1 and not env_segments_after_prompt
        ), "_build_sequence is the simple path; use _build_main_sequence_with_spawn for the spawn branch"
        resp = model_segments[0]
        seq = list(prompt_ids) + list(resp.output_tokens)
        loss_mask = [0] * len(prompt_ids) + [1] * len(resp.output_tokens)
        logprobs = [0.0] * len(prompt_ids) + list(resp.output_logprobs)
        versions = [-1] * len(prompt_ids) + list(resp.output_versions)
        return self._pack(seq, loss_mask, logprobs, versions)

    def _build_main_sequence_with_spawn(
        self,
        prompt_ids: list[int],
        phase1_tokens: list[int],
        phase1_logprobs: list[float],
        phase1_versions: list[int],
        agg_ids: list[int],
        resp2: ModelResponse,
    ) -> dict[str, torch.Tensor]:
        """Layout:
            [prompt | phase1_out (incl. </spawn>) | <spawn_result>...| phase2_out]
              0                  1                          0                1
        ``phase1_tokens`` is already truncated to end at </spawn>.
        """
        seq = (
            list(prompt_ids)
            + list(phase1_tokens)
            + list(agg_ids)
            + list(resp2.output_tokens)
        )
        p_len = len(prompt_ids)
        p1_len = len(phase1_tokens)
        agg_len = len(agg_ids)
        p2_len = len(resp2.output_tokens)

        loss_mask = (
            [0] * p_len
            + [1] * p1_len        # main agent's pre-spawn reasoning + tool call
            + [0] * agg_len       # env output (sub-agent results)
            + [1] * p2_len        # main agent's post-spawn answer
        )
        logprobs = (
            [0.0] * p_len
            + list(phase1_logprobs)
            + [0.0] * agg_len
            + list(resp2.output_logprobs)
        )
        versions = (
            [-1] * p_len
            + list(phase1_versions)
            + [-1] * agg_len
            + list(resp2.output_versions)
        )
        return self._pack(seq, loss_mask, logprobs, versions)

    @staticmethod
    def _pack(seq, loss_mask, logprobs, versions) -> dict[str, torch.Tensor]:
        n = len(seq)
        out = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(n, dtype=torch.bool),
        }
        return {k: v.unsqueeze(0) for k, v in out.items()}

    # ------------------------------------------------------------------ reward

    async def _compute_reward(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[float, str]:
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )
        if not isinstance(reward, (int, float)):
            reward = float(reward)
        return reward, completions_str

    # ------------------------------------------------------------------ entry

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        n_samples = self.gconfig.n_samples
        version = engine.get_version()

        # Fan out n_samples spawn-episodes in parallel for the same prompt.
        raw = await asyncio.gather(
            *[self._run_one_episode(engine, data) for _ in range(n_samples)]
        )

        trajectories: list[dict[str, Any]] = []
        traj_rewards: list[float] = []
        debugs: list[dict[str, Any]] = []
        for sequences, reward, debug in raw:
            trajectories.append({"sequences": sequences})
            traj_rewards.append(reward)
            debugs.append(debug)

        # Optional debug dump.  Set dump_prob=1.0 in the YAML to dump
        # every rollout (useful for sanity-checking the spawn protocol).
        if self.dump_dir is not None and random.random() < self.dump_prob:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            qid = resolve_prompt_id(data) or uuid.uuid4().hex
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, (debug, reward) in enumerate(zip(debugs, traj_rewards)):
                    await f.write(
                        f"=== Sample {i + 1}/{n_samples} reward={reward} "
                        f"spawned={debug['spawned']} n_sub={debug['n_sub']} ===\n"
                        f"--- phase 1 (main, until </spawn>) ---\n{debug['phase1_text']}\n\n"
                    )
                    if debug["spawned"]:
                        for j, out in enumerate(debug["sub_outputs"]):
                            await f.write(
                                f"--- sub-agent {j} task: {debug['tasks'][j]!r} ---\n{out}\n\n"
                            )
                        await f.write(
                            f"--- phase 2 (main, after spawn_result) ---\n{debug['phase2_text']}\n\n"
                        )

        return {
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(traj_rewards, dtype=torch.float32),
            "trajectories": trajectories,
            "prompt_id": resolve_prompt_id(data),
        }
