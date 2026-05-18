"""Plan-and-Solve math workflow: two-model collaborative reasoning.

**model0** (planner) generates a step-by-step plan for the math problem.
**model1** (solver) produces the final answer following the plan.

Both models' tokens are tracked via the ``model_ids`` tensor so each
trainer only computes loss on its own model's outputs.  Rewards are
computed on the solver's final answer and shared across both models.

Usage in YAML config::

    workflow_spec:
      workflow_cls: plan_and_solve
      reward_fn: "math_verify"
      tokenizer: "Qwen/Qwen3-1.7B"
"""

import asyncio
import os
import random
import uuid
from collections.abc import Callable
from typing import Any

import aiofiles
import aiofiles.os
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.workflow.api.cli_args import GenerationHyperparameters
from astraflow.workflow.api.engine_api import EngineGroup, InferenceEngine
from astraflow.workflow.api.io_struct import ModelRequest
from astraflow.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.workflow.api.workflow_api import RolloutWorkflow
from astraflow.workflow.registry import register_workflow
from astraflow.workflow.utils import logging, stats_tracker
from astraflow.workflow.utils.data import resolve_prompt_id, results_to_structured
from astraflow.workflow.utils.dynamic_import import import_from_string

logger = logging.getLogger("PlanAndSolve workflow")

# model_ids constants
MODEL_ID_PROMPT = -1
MODEL_ID_PLANNER = 0  # model0
MODEL_ID_SOLVER = 1  # model1

# Dataset suffix that conflicts with multi-model workflow instructions.
# Stripped from user messages so each model gets its own tailored instructions.
_DATASET_SUFFIXES = [
    "\nLet's think step by step. Please put your final answer within \\boxed{}.",
    "\nPlease put your final answer within \\boxed{}.",
]


def _strip_dataset_suffix(messages: list[dict]) -> list[dict]:
    """Remove dataset-injected suffixes from user messages."""
    cleaned = []
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "user":
            for suffix in _DATASET_SUFFIXES:
                if content.endswith(suffix):
                    content = content[: -len(suffix)]
                    break
        cleaned.append({**msg, "content": content})
    return cleaned


# ── Prompt templates ──
PLAN_SYSTEM = (
    "You are a math problem planner. Given a math problem, produce a clear, "
    "concise step-by-step plan to solve it. Do NOT compute the final answer — "
    "only outline the reasoning steps."
)
PLAN_SUFFIX = (
    "Write a step-by-step plan to solve this problem. "
    "List each reasoning step clearly. Do NOT give the final numerical answer."
)

# Solver instruction is appended as a follow-up user turn in the same
# conversation, so the solver sees the full planner context.
SOLVE_FOLLOWUP = (
    "Now follow the plan above and solve the problem. "
    "Show your work and put the final answer in \\boxed{}."
)


@register_workflow("plan_and_solve")
class PlanAndSolveWorkflow(RolloutWorkflow):
    """Two-model plan-and-solve workflow for math problems.

    Parameters
    ----------
    reward_fn : callable or str
        Reward function applied to the solver's final answer.
    gconfig : GenerationHyperparameters
        Generation config (n_samples generates n plan-solve pairs).
    tokenizer : str or PreTrainedTokenizerFast
        Tokenizer for the solver (model1). Also used as fallback for planner.
    planner_tokenizer : str or PreTrainedTokenizerFast or None
        Optional separate tokenizer for the planner (model0). If the planner
        uses a different model family with a different chat template, set this
        so ``apply_chat_template`` produces correct input_ids. Must share the
        same vocabulary as the solver tokenizer.
    enable_thinking : bool
        Whether to enable thinking tokens in chat template.
    dump_dir : str | None
        If set, dump trajectories for debugging.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        planner_tokenizer: PreTrainedTokenizerFast | str | None = None,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        if isinstance(tokenizer, str):
            from astraflow.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        # Separate tokenizer for planner if specified (different chat template)
        if planner_tokenizer is not None:
            if isinstance(planner_tokenizer, str):
                from astraflow.workflow.utils.hf_utils import load_hf_tokenizer

                planner_tokenizer = load_hf_tokenizer(planner_tokenizer)
            self.planner_tokenizer = planner_tokenizer
        else:
            self.planner_tokenizer = self.tokenizer

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.dump_dir = dump_dir
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    def _apply_chat_template(self, messages, tokenizer=None, **kwargs):
        """Apply chat template with optional enable_thinking support."""
        tok = tokenizer or self.tokenizer
        try:
            return list(tok.apply_chat_template(
                messages, **kwargs, enable_thinking=self.enable_thinking,
            ))
        except TypeError:
            return list(tok.apply_chat_template(messages, **kwargs))

    def _compute_transition_ids(self) -> list[int]:
        """Compute the transition tokens between planner output and solver input.

        Uses chat template diffing: apply template to messages with and without
        the solver follow-up turn, then take the difference.  This produces
        the exact structural tokens (user turn markers, generation prompt)
        regardless of the chat template format.

        Uses token-level common prefix detection and EOS-based stripping to
        handle BPE boundary effects where the tokenizer produces different
        tokens at the junction depending on context.
        """
        # Messages up to and including planner's assistant response
        prefix_msgs = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": "X"},
            {"role": "user", "content": PLAN_SUFFIX},
            {"role": "assistant", "content": "X"},  # placeholder
        ]
        # Same, plus the solver follow-up user turn
        full_msgs = prefix_msgs + [
            {"role": "user", "content": SOLVE_FOLLOWUP},
        ]
        prefix_ids = self._apply_chat_template(
            prefix_msgs, tokenize=True, add_generation_prompt=False,
        )
        full_ids = self._apply_chat_template(
            full_msgs, tokenize=True, add_generation_prompt=True,
        )
        # Find actual common prefix length (handles BPE boundary effects).
        # The naive full_ids[len(prefix_ids):] breaks when the tokenizer
        # produces different tokens at the boundary depending on context.
        common_len = 0
        for i in range(min(len(prefix_ids), len(full_ids))):
            if prefix_ids[i] != full_ids[i]:
                break
            common_len = i + 1

        transition = full_ids[common_len:]

        # If BPE mismatch occurred, the transition includes leftover
        # placeholder tokens (e.g., "X<|im_end|>"). Strip everything up to
        # and including the first EOS to remove them — the real transition
        # starts after the placeholder's end-of-turn marker.
        if common_len < len(prefix_ids):
            eos_id = self.tokenizer.eos_token_id
            if eos_id in transition:
                eos_pos = transition.index(eos_id)
                transition = transition[eos_pos + 1:]
        return transition

    async def _plan_and_solve_one(
        self,
        engine0: InferenceEngine,
        engine1: InferenceEngine,
        messages: list[dict],
        task_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Run one plan-and-solve sample and return result tensors.

        Builds a single continuous sequence so the solver's generation
        context matches its training context exactly::

            [plan_prompt | plan_output | transition | solve_output]

        ``transition`` contains the structural tokens (EOS, user turn
        with solver instruction, generation prompt) derived from the
        chat template.
        """

        # --- Step 1: Planner (model0) ---
        clean_msgs = _strip_dataset_suffix(messages)
        plan_messages = [
            {"role": "system", "content": PLAN_SYSTEM},
        ] + clean_msgs + [
            {"role": "user", "content": PLAN_SUFFIX},
        ]

        plan_input_ids = self._apply_chat_template(
            plan_messages, tokenizer=self.planner_tokenizer,
            tokenize=True, add_generation_prompt=True,
        )

        plan_resp = await engine0.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=plan_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.planner_tokenizer,
            )
        )
        plan_text = self.planner_tokenizer.decode(plan_resp.output_tokens)
        plan_output_ids = list(plan_resp.output_tokens)
        plan_logprobs = list(plan_resp.output_logprobs)

        # Ensure planner output ends with EOS for proper turn termination
        eos_id = self.tokenizer.eos_token_id
        if not plan_output_ids or plan_output_ids[-1] != eos_id:
            plan_output_ids.append(eos_id)
            plan_logprobs.append(0.0)

        # --- Step 2: Build transition tokens ---
        # Transition = structural tokens between planner EOS and solver
        # generation start (user turn with SOLVE_FOLLOWUP + gen prompt).
        transition_ids = self._compute_transition_ids()

        # --- Step 3: Solver (model1) ---
        # Solver sees the full continuous prefix:
        #   [plan_prompt | plan_output (with EOS) | transition]
        solver_prefix_ids = plan_input_ids + plan_output_ids + transition_ids

        solve_resp = await engine1.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=solver_prefix_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        solve_text = self.tokenizer.decode(solve_resp.output_tokens)
        solve_output_ids = list(solve_resp.output_tokens)
        solve_logprobs = list(solve_resp.output_logprobs)

        # --- Compute reward on the final answer ---
        prompt_str = self.tokenizer.decode(plan_input_ids)
        reward = await self.async_reward_fn(
            prompt_str,
            solve_text,
            solver_prefix_ids,
            solve_output_ids,
            **task_data,
        )
        if not isinstance(reward, (int, float)):
            reward = float(reward)

        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        # --- Build single continuous trajectory ---
        # Layout: [plan_input | plan_output | transition | solve_output]
        full_ids = (
            plan_input_ids + plan_output_ids + transition_ids + solve_output_ids
        )
        full_logprobs = (
            [0.0] * len(plan_input_ids)
            + plan_logprobs
            + [0.0] * len(transition_ids)
            + solve_logprobs
        )

        total_len = len(full_ids)
        p_in = len(plan_input_ids)
        p_out = len(plan_output_ids)
        t_len = len(transition_ids)
        s_out = len(solve_output_ids)

        # input_ids
        input_ids_t = torch.tensor(full_ids, dtype=torch.int32)

        # logprobs
        logprobs_t = torch.tensor(full_logprobs, dtype=torch.float32)

        # loss_mask: 1 for model outputs, 0 for prompts/transition
        loss_mask = torch.zeros(total_len, dtype=torch.int32)
        loss_mask[p_in : p_in + p_out] = 1  # planner output
        loss_mask[p_in + p_out + t_len :] = 1  # solver output

        # model_ids: -1 for prompts/transition, 0 for planner, 1 for solver
        model_ids = torch.full((total_len,), MODEL_ID_PROMPT, dtype=torch.long)
        model_ids[p_in : p_in + p_out] = MODEL_ID_PLANNER
        model_ids[p_in + p_out + t_len :] = MODEL_ID_SOLVER

        # versions: use engine version for output tokens, -1 for input/transition
        plan_versions = (
            plan_resp.output_versions
            if hasattr(plan_resp, "output_versions") and plan_resp.output_versions
            else [engine0.get_version()] * p_out
        )
        solve_versions = (
            solve_resp.output_versions
            if hasattr(solve_resp, "output_versions") and solve_resp.output_versions
            else [engine1.get_version()] * s_out
        )
        versions = (
            [-1] * p_in
            + list(plan_versions)
            + [-1] * t_len
            + list(solve_versions)
        )
        versions_t = torch.tensor(versions, dtype=torch.int32)

        # attention_mask
        attention_mask = torch.ones(total_len, dtype=torch.bool)

        # rewards
        rewards_t = torch.tensor(reward, dtype=torch.float32)

        result = {
            "input_ids": input_ids_t.unsqueeze(0),
            "logprobs": logprobs_t.unsqueeze(0),
            "loss_mask": loss_mask.unsqueeze(0),
            "model_ids": model_ids.unsqueeze(0),
            "versions": versions_t.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "rewards": rewards_t.unsqueeze(0),
        }

        # Decode all segments for debugging
        prompt_text = self.tokenizer.decode(plan_input_ids)
        transition_text = self.tokenizer.decode(transition_ids)

        trajectory_info = {
            "prompt": prompt_text,
            "plan": plan_text,
            "transition": transition_text,
            "solve": solve_text,
            "prompt_len": p_in,
            "plan_len": p_out,
            "transition_len": t_len,
            "solve_len": s_out,
            "total_len": total_len,
        }

        return result, trajectory_info, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # Resolve reward function if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # Resolve engines
        if isinstance(engine, EngineGroup):
            engine0 = engine["model0"]
            engine1 = engine["model1"]
        else:
            engine0 = engine
            engine1 = engine

        messages = data["messages"]
        n_samples = self.gconfig.n_samples
        version = engine0.get_version()

        # Generate n_samples plan-solve pairs in parallel
        sample_coros = [
            self._plan_and_solve_one(engine0, engine1, messages, data)
            for _ in range(n_samples)
        ]
        raw_results = await asyncio.gather(*sample_coros)

        # Collect successful results
        results = []
        trajectory_infos = []
        rewards = []
        for r in raw_results:
            if r is not None:
                result, traj_info, reward = r
                results.append(result)
                trajectory_infos.append(traj_info)
                rewards.append(reward)

        # Debug dump (1 in 128 chance)
        if self.dump_dir is not None and random.random() < 1 / 128:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            qid = resolve_prompt_id(data) or uuid.uuid4().hex

            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, (traj, rew) in enumerate(
                    zip(trajectory_infos, rewards)
                ):
                    await f.write(
                        f"=== Sample {i + 1}/{n_samples} (reward={rew}) ===\n"
                        f"--- Lengths: prompt={traj['prompt_len']}, "
                        f"plan={traj['plan_len']}, "
                        f"transition={traj['transition_len']}, "
                        f"solve={traj['solve_len']}, "
                        f"total={traj['total_len']} ---\n\n"
                        f"--- Prompt ---\n{traj['prompt']}\n\n"
                        f"--- Plan (model0) ---\n{traj['plan']}\n\n"
                        f"--- Transition ---\n{traj['transition']}\n\n"
                        f"--- Answer (model1) ---\n{traj['solve']}\n\n"
                    )

        return results_to_structured(results, prompt_id=resolve_prompt_id(data))
