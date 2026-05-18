"""Solve-and-Check v2: two-model workflow with independent sequences.

**model0** (solver) generates a full solution for the math problem.
**model1** (checker) reviews the solution in a **separate sequence** and
produces the final answer.

Unlike v1 (``solve_and_check``), each model gets its own independent
sequence.  The checker's prompt is a fresh chat-template application
containing the solver's decoded text — not a raw token continuation.
This eliminates transition-token hacks, shortens forward passes, and
ensures the checker sees exactly the same context during training and
inference.

Returns the ASearcher-style structured format::

    {
        "n_trajs": int,
        "rewards": Tensor[n_trajs],
        "trajectories": [{"sequences": [solver_seq, checker_seq]}, ...],
    }

Usage in YAML config::

    workflow_spec:
      workflow_cls: sep_solve_and_check
      reward_fn: "math_verify"
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
from astraflow.workflow.utils.data import resolve_prompt_id
from astraflow.workflow.utils.dynamic_import import import_from_string

logger = logging.getLogger("SolveAndCheckV2 workflow")

# model_ids constants
MODEL_ID_PROMPT = -1
MODEL_ID_SOLVER = 0  # model0
MODEL_ID_CHECKER = 1  # model1

# Dataset suffix that conflicts with multi-model workflow instructions.
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

CHECK_SYSTEM = (
    "You are a math solution checker. You will be given a math problem and a "
    "proposed solution. Review the solution step by step, check for errors in "
    "reasoning or computation. If correct, confirm the answer. If there are "
    "errors, fix them. Give the final correct answer in \\boxed{}."
)


def _build_seq_dict(
    input_ids: list[int],
    output_ids: list[int],
    output_logprobs: list[float],
    output_versions: list[int],
    model_id: int,
    is_first: bool,
) -> dict[str, Any]:
    """Build a self-contained sequence tensor dict."""
    full_ids = input_ids + output_ids
    total_len = len(full_ids)
    p_len = len(input_ids)
    o_len = len(output_ids)

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * p_len + list(output_logprobs), dtype=torch.float32
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [0] * p_len + [1] * o_len, dtype=torch.int32
        ).unsqueeze(0),
        "model_ids": torch.cat([
            torch.full((p_len,), MODEL_ID_PROMPT, dtype=torch.long),
            torch.full((o_len,), model_id, dtype=torch.long),
        ]).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * p_len + list(output_versions), dtype=torch.int32
        ).unsqueeze(0),
        "attention_mask": torch.ones(total_len, dtype=torch.bool).unsqueeze(0),
        "begin_of_trajectory": torch.tensor([int(is_first)]),
    }


@register_workflow("sep_solve_and_check")
class SepSolveAndCheckWorkflow(RolloutWorkflow):
    """Two-model solve-and-check workflow with independent sequences.

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
            from astraflow.workflow.utils.hf_utils import load_hf_tokenizer

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
        try:
            return list(self.tokenizer.apply_chat_template(
                messages, **kwargs, enable_thinking=self.enable_thinking,
            ))
        except TypeError:
            return list(self.tokenizer.apply_chat_template(messages, **kwargs))

    async def _solve_and_check_one(
        self,
        engine0: InferenceEngine,
        engine1: InferenceEngine,
        messages: list[dict],
        task_data: dict[str, Any],
    ) -> tuple[list[dict], float, dict] | None:
        """Run one solve-and-check sample.

        Returns two independent sequences (solver + checker), the reward,
        and trajectory info for debugging.
        """

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

        # --- Step 2: Build checker prompt (fresh template) ---
        # Extract problem text from the original messages
        problem_text = ""
        for msg in clean_msgs:
            if msg["role"] == "user":
                problem_text = msg["content"]
                break

        check_messages = [
            {"role": "system", "content": CHECK_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Problem:\n{problem_text}\n\n"
                    f"Proposed solution:\n{solve_text}"
                ),
            },
        ]
        check_input_ids = self._apply_chat_template(
            check_messages, tokenize=True, add_generation_prompt=True,
        )

        # --- Step 3: Checker (model1) ---
        check_resp = await engine1.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=check_input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        check_text = self.tokenizer.decode(check_resp.output_tokens)
        check_output_ids = list(check_resp.output_tokens)
        check_logprobs = list(check_resp.output_logprobs)

        # --- Step 4: Reward on checker's output ---
        prompt_str = self.tokenizer.decode(solve_input_ids)
        reward = await self.async_reward_fn(
            prompt_str,
            check_text,
            check_input_ids,
            check_output_ids,
            **task_data,
        )
        if not isinstance(reward, (int, float)):
            reward = float(reward)

        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        # --- Step 5: Build two independent sequence dicts ---
        solve_versions = (
            solve_resp.output_versions
            if hasattr(solve_resp, "output_versions") and solve_resp.output_versions
            else [engine0.get_version()] * len(solve_output_ids)
        )
        check_versions = (
            check_resp.output_versions
            if hasattr(check_resp, "output_versions") and check_resp.output_versions
            else [engine1.get_version()] * len(check_output_ids)
        )

        solver_seq = _build_seq_dict(
            input_ids=solve_input_ids,
            output_ids=solve_output_ids,
            output_logprobs=solve_logprobs,
            output_versions=list(solve_versions),
            model_id=MODEL_ID_SOLVER,
            is_first=True,
        )
        checker_seq = _build_seq_dict(
            input_ids=check_input_ids,
            output_ids=check_output_ids,
            output_logprobs=check_logprobs,
            output_versions=list(check_versions),
            model_id=MODEL_ID_CHECKER,
            is_first=False,
        )

        solve_prompt_text = self.tokenizer.decode(solve_input_ids)
        check_prompt_text = self.tokenizer.decode(check_input_ids)
        trajectory_info = {
            "prompt": problem_text,
            "solve_prompt": solve_prompt_text,
            "solve": solve_text,
            "check_prompt": check_prompt_text,
            "check": check_text,
            "solve_prompt_len": len(solve_input_ids),
            "solve_output_len": len(solve_output_ids),
            "check_prompt_len": len(check_input_ids),
            "check_output_len": len(check_output_ids),
        }

        return [solver_seq, checker_seq], reward, trajectory_info

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
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
        trajectories = []
        trajectory_infos = []
        rewards = []
        for r in raw_results:
            if r is not None:
                sequences, reward, traj_info = r
                trajectories.append({"sequences": sequences})
                trajectory_infos.append(traj_info)
                rewards.append(reward)

        # Extract prompt identifier (canonical helper — same id the
        # curator gate saw on this prompt).
        qid = resolve_prompt_id(data)

        # Debug dump (1 in 128 chance)
        if self.dump_dir is not None and random.random() < 1 / 128:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            dump_qid = qid or uuid.uuid4().hex
            file_path = os.path.join(dump_path, f"{dump_qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, (traj_info, rew) in enumerate(
                    zip(trajectory_infos, rewards)
                ):
                    await f.write(
                        f"=== Sample {i + 1}/{n_samples} (reward={rew}) ===\n"
                        f"--- Lengths: solve_prompt={traj_info['solve_prompt_len']}, "
                        f"solve_output={traj_info['solve_output_len']}, "
                        f"check_prompt={traj_info['check_prompt_len']}, "
                        f"check_output={traj_info['check_output_len']} ---\n\n"
                        f"--- Solve Prompt (full) ---\n{traj_info['solve_prompt']}\n\n"
                        f"--- Solve Output (model0) ---\n{traj_info['solve']}\n\n"
                        f"--- Check Prompt (full) ---\n{traj_info['check_prompt']}\n\n"
                        f"--- Check Output (model1) ---\n{traj_info['check']}\n\n"
                    )

        return {
            "prompt_id": qid,
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "trajectories": trajectories,
        }
