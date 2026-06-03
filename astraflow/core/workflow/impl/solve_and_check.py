"""Solve-and-Check math workflow: two-model collaborative reasoning.

**model0** (solver) generates a full solution for the math problem.
**model1** (checker) reviews the solution, catches errors, and produces
the final answer.

Both models' tokens are tracked via the ``model_ids`` tensor so each
trainer only computes loss on its own model's outputs.  Rewards are
computed on the checker's final answer and shared across both models.

Usage in YAML config::

    workflow_spec:
      workflow_cls: solve_and_check
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

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import EngineGroup, InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest
from astraflow.core.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id, results_to_structured
from astraflow.core.workflow.utils.dynamic_import import import_from_string

logger = logging.getLogger("SolveAndCheck workflow")

# model_ids constants
MODEL_ID_PROMPT = -1
MODEL_ID_SOLVER = 0  # model0
MODEL_ID_CHECKER = 1  # model1

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
SOLVE_SYSTEM = (
    "You are a math problem solver. Given a math problem, solve it step by step "
    "and put the final answer in \\boxed{}."
)

# Checker instruction is appended as a follow-up user turn in the same
# conversation, so the checker sees the full solver context.
CHECK_FOLLOWUP = (
    "Review your solution above step by step. Check for any errors in reasoning "
    "or computation. If correct, confirm the answer. If there are errors, fix "
    "them. Give the final correct answer in \\boxed{}."
)


@register_workflow("solve_and_check")
class SolveAndCheckWorkflow(RolloutWorkflow):
    """Two-model solve-and-check workflow for math problems.

    Parameters
    ----------
    reward_fn : callable or str
        Reward function applied to the checker's final answer.
    gconfig : GenerationHyperparameters
        Generation config (n_samples generates n solve-check pairs).
    tokenizer : str or PreTrainedTokenizerFast
        Tokenizer (shared by both models).
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
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        if isinstance(tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.dump_dir = dump_dir
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    def _apply_chat_template(self, messages, **kwargs):
        """Apply chat template with optional enable_thinking support."""
        from astraflow.core.workflow.utils.hf_utils import apply_chat_template_to_ids
        return apply_chat_template_to_ids(
            self.tokenizer, messages, enable_thinking=self.enable_thinking, **kwargs
        )

    def _compute_transition_ids(self, problem_text: str) -> list[int]:
        """Compute the transition tokens between solver output and checker input.

        Uses chat template diffing: apply template to messages with and without
        the checker follow-up turn, then take the difference.  This produces
        the exact structural tokens (EOS, user turn markers, generation prompt)
        regardless of the chat template format.

        Uses token-level common prefix detection instead of naive length-based
        slicing, which breaks when BPE produces different token boundaries
        at the junction (e.g. missing ``<|im_start|>user`` markers).
        """
        # Messages up to and including solver's assistant response
        prefix_msgs = [
            {"role": "system", "content": SOLVE_SYSTEM},
            {"role": "user", "content": problem_text},
            {"role": "assistant", "content": "X"},  # placeholder
        ]
        # Same, plus the checker follow-up user turn
        full_msgs = prefix_msgs + [
            {"role": "user", "content": CHECK_FOLLOWUP},
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

    async def _solve_and_check_one(
        self,
        engine0: InferenceEngine,
        engine1: InferenceEngine,
        messages: list[dict],
        task_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Run one solve-and-check sample and return result tensors.

        Builds a single continuous sequence so the checker's generation
        context matches its training context exactly::

            [solve_prompt | solve_output | transition | check_output]

        ``transition`` contains the structural tokens (EOS, user turn
        with checker instruction, generation prompt) derived from the
        chat template.
        """

        # Extract problem text
        problem_text = ""
        for msg in messages:
            if msg["role"] == "user":
                problem_text = msg["content"]
                break

        # --- Step 1: Solver (model0) ---
        clean_msgs = _strip_dataset_suffix(messages)
        solve_messages = [
            {"role": "system", "content": SOLVE_SYSTEM},
        ] + clean_msgs

        solve_input_ids = self._apply_chat_template(
            solve_messages, tokenize=True, add_generation_prompt=True,
        )

        solve_resp = await engine0.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=solve_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        solve_text = self.tokenizer.decode(solve_resp.output_tokens)
        solve_output_ids = list(solve_resp.output_tokens)
        solve_logprobs = list(solve_resp.output_logprobs)

        # Ensure solver output ends with EOS for proper turn termination
        eos_id = self.tokenizer.eos_token_id
        if not solve_output_ids or solve_output_ids[-1] != eos_id:
            solve_output_ids.append(eos_id)
            solve_logprobs.append(0.0)

        # --- Step 2: Build transition tokens ---
        # Transition = structural tokens between solver EOS and checker
        # generation start (user turn with CHECK_FOLLOWUP + gen prompt).
        transition_ids = self._compute_transition_ids(problem_text)

        # --- Step 3: Checker (model1) ---
        # Checker sees the full continuous prefix:
        #   [solve_prompt | solve_output (with EOS) | transition]
        checker_prefix_ids = solve_input_ids + solve_output_ids + transition_ids

        check_resp = await engine1.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=checker_prefix_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        check_text = self.tokenizer.decode(check_resp.output_tokens)
        check_output_ids = list(check_resp.output_tokens)
        check_logprobs = list(check_resp.output_logprobs)

        # --- Compute reward on the checker's final answer ---
        prompt_str = self.tokenizer.decode(solve_input_ids)
        reward = await self.async_reward_fn(
            prompt_str,
            check_text,
            checker_prefix_ids,
            check_output_ids,
            **task_data,
        )
        if not isinstance(reward, (int, float)):
            reward = float(reward)

        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        # --- Build single continuous trajectory ---
        # Layout: [solve_input | solve_output | transition | check_output]
        full_ids = (
            solve_input_ids + solve_output_ids + transition_ids + check_output_ids
        )
        full_logprobs = (
            [0.0] * len(solve_input_ids)
            + solve_logprobs
            + [0.0] * len(transition_ids)
            + check_logprobs
        )

        total_len = len(full_ids)
        s_in = len(solve_input_ids)
        s_out = len(solve_output_ids)
        t_len = len(transition_ids)
        c_out = len(check_output_ids)

        # input_ids
        input_ids_t = torch.tensor(full_ids, dtype=torch.int32)

        # logprobs
        logprobs_t = torch.tensor(full_logprobs, dtype=torch.float32)

        # loss_mask: 1 for model outputs, 0 for prompts/transition
        loss_mask = torch.zeros(total_len, dtype=torch.int32)
        loss_mask[s_in : s_in + s_out] = 1  # solver output
        loss_mask[s_in + s_out + t_len :] = 1  # checker output

        # model_ids: -1 for prompts/transition, 0 for solver, 1 for checker
        model_ids = torch.full((total_len,), MODEL_ID_PROMPT, dtype=torch.long)
        model_ids[s_in : s_in + s_out] = MODEL_ID_SOLVER
        model_ids[s_in + s_out + t_len :] = MODEL_ID_CHECKER

        # versions: use engine version for output tokens, -1 for input/transition
        solve_versions = (
            solve_resp.output_versions
            if hasattr(solve_resp, "output_versions") and solve_resp.output_versions
            else [engine0.get_version()] * s_out
        )
        check_versions = (
            check_resp.output_versions
            if hasattr(check_resp, "output_versions") and check_resp.output_versions
            else [engine1.get_version()] * c_out
        )
        versions = (
            [-1] * s_in
            + list(solve_versions)
            + [-1] * t_len
            + list(check_versions)
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
        prompt_text = self.tokenizer.decode(solve_input_ids)
        transition_text = self.tokenizer.decode(transition_ids)

        trajectory_info = {
            "prompt": prompt_text,
            "solve": solve_text,
            "transition": transition_text,
            "check": check_text,
            "prompt_len": s_in,
            "solve_len": s_out,
            "transition_len": t_len,
            "check_len": c_out,
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

        # Generate n_samples solve-check pairs in parallel
        sample_coros = [
            self._solve_and_check_one(engine0, engine1, messages, data)
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
                        f"solve={traj['solve_len']}, "
                        f"transition={traj['transition_len']}, "
                        f"check={traj['check_len']}, "
                        f"total={traj['total_len']} ---\n\n"
                        f"--- Prompt ---\n{traj['prompt']}\n\n"
                        f"--- Solution (model0) ---\n{traj['solve']}\n\n"
                        f"--- Transition ---\n{traj['transition']}\n\n"
                        f"--- Check (model1) ---\n{traj['check']}\n\n"
                    )

        return results_to_structured(results, prompt_id=resolve_prompt_id(data))
