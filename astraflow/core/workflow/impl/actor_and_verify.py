"""Actor-and-Verify: two-model multi-agent workflow.

**model0** (solver) generates a solution for the math problem.
**model1** (verifier) checks the solution and replies with
``<verify>approve</verify>`` or ``<verify>reject</verify>``.

If the verifier rejects, the solver retries **once** with the verifier's
full output as team context.  There is no second verification — the retry
solution is final.

Each generation (solver attempt or verifier check) produces an independent
sequence dict with its own ``rewards``:

- Solver reward: 1.0 if ``\\boxed{}`` matches groundtruth, else 0.0
- Verifier reward: 1.0 if the approve/reject decision matches actual
  correctness, else 0.0

Returns the ASearcher-style structured format::

    {
        "n_trajs": int,
        "rewards": Tensor[n_trajs],
        "trajectories": [{"sequences": [seq1, seq2, ...]}, ...],
    }

Usage in YAML config::

    workflow_spec:
      workflow_cls: actor_and_verify
      reward_fn: "math_verify"
"""

import asyncio
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
from astraflow.core.workflow.api.engine_api import EngineGroup, InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest
from astraflow.core.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id
from astraflow.core.workflow.utils.dynamic_import import import_from_string

logger = logging.getLogger("ActorAndVerify workflow")

# model_ids constants
MODEL_ID_PROMPT = -1
MODEL_ID_SOLVER = 0  # model0
MODEL_ID_VERIFIER = 1  # model1

# Dataset suffix that conflicts with multi-model workflow instructions.
_DATASET_SUFFIXES = [
    "\nLet's think step by step. Please put your final answer within \\boxed{}.",
    "\nPlease put your final answer within \\boxed{}.",
]

ENV_PROMPT_TEMPLATE = (
    "You are a member of an expert multi-agent team tasked with solving the "
    "math problem. The team's math problem is:\n{task_description}"
)

SOLVER_PROMPT_TEMPLATE = (
    "# Task Introduction\n{env_prompt}\n"
    "# Your Teammates' Outputs\n{team_context}\n"
    "# Your Role\n"
    "You are a \"Solver Agent\". Your job is to carefully reason through the "
    "math problem step by step and derive the correct answer. When reasoning, "
    "consider your teammates' outputs if available. Put the final answer in "
    "\\boxed{{}}."
)

VERIFIER_PROMPT_TEMPLATE = (
    "# Task Introduction\n{env_prompt}\n"
    "# Your Teammates' Outputs\n{team_context}\n"
    "# Your Role\n"
    "You are a \"Verifier Agent\". Your responsibility is to critically review "
    "the most recent solution provided by the \"Solver Agent\". First, carefully "
    "examine each reasoning step, formula, and conclusion for accuracy, "
    "completeness, and logical consistency. Explain your analysis in detail. "
    "After completing your analysis, at the very end of your output, you MUST "
    "provide your final verdict within <verify> </verify> using exactly one of:\n"
    "(1) <verify>approve</verify> if all steps and the final answer are correct.\n"
    "(2) <verify>reject</verify> if you detect any issue.\n\n"
    "Important: Do NOT output your verdict until you have finished your full analysis."
)


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


def _parse_verify_tag(text: str) -> tuple[bool, bool]:
    """Parse <verify>approve</verify> or <verify>reject</verify>.

    Returns (approved, parsed_ok).  ``parsed_ok`` is False when
    no valid ``<verify>`` tag is found.
    """
    match = re.search(r"<verify>\s*(approve|reject)\s*</verify>", text, re.IGNORECASE)
    if match is not None:
        return match.group(1).lower() == "approve", True

    # Fallback: look for keywords outside tags
    text_lower = text.lower()
    has_approve = "approve" in text_lower
    has_reject = "reject" in text_lower
    if has_approve and not has_reject:
        return True, True
    elif has_reject and not has_approve:
        return False, True

    # Cannot determine — default approve to not waste solver retry
    return True, False


def _build_seq_dict(
    input_ids: list[int],
    output_ids: list[int],
    output_logprobs: list[float],
    output_versions: list[int],
    model_id: int,
    reward: float,
    is_first: bool,
) -> dict[str, Any]:
    """Build a self-contained sequence tensor dict with per-sequence reward."""
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
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "begin_of_trajectory": torch.tensor([int(is_first)]),
    }


@register_workflow("actor_and_verify")
class ActorAndVerifyWorkflow(RolloutWorkflow):
    """Two-model actor-and-verify workflow with at most one verification.

    Parameters
    ----------
    reward_fn : callable or str
        Reward function for checking correctness of solver's answer.
    gconfig : GenerationHyperparameters
        Generation config (n_samples generates n independent sample chains).
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
        gconfigs: dict[str, GenerationHyperparameters] | None = None,
    ):
        self.reward_fn = reward_fn
        if isinstance(tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        # Per-model gconfigs from RaaS (model0=solver, model1=verifier)
        if gconfigs is not None:
            self.solver_gconfig = gconfigs.get(
                "model0", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
            self.verifier_gconfig = gconfigs.get(
                "model1", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
        else:
            self.solver_gconfig = self.gconfig
            self.verifier_gconfig = self.gconfig
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

    async def _actor_and_verify_one(
        self,
        engine0: InferenceEngine,
        engine1: InferenceEngine,
        messages: list[dict],
        task_data: dict[str, Any],
    ) -> tuple[list[dict], float, dict] | None:
        """Run one actor-and-verify chain (at most 1 verification).

        Returns a list of sequence dicts, the final trajectory reward,
        and debug info.
        """
        clean_msgs = _strip_dataset_suffix(messages)
        problem_text = ""
        for msg in clean_msgs:
            if msg["role"] == "user":
                problem_text = msg["content"]
                break

        env_prompt = ENV_PROMPT_TEMPLATE.format(task_description=problem_text)

        sequences = []
        round_infos = []

        # ---- Round 1: Solver (no team context) ----
        solver_prompt_1 = SOLVER_PROMPT_TEMPLATE.format(
            env_prompt=env_prompt, team_context="",
        )
        solve_messages_1 = [{"role": "user", "content": solver_prompt_1}]
        solve_input_ids_1 = self._apply_chat_template(
            solve_messages_1, tokenize=True, add_generation_prompt=True,
        )
        solve_resp_1 = await engine0.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=solve_input_ids_1,
                gconfig=self.solver_gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        solve_text_1 = self.tokenizer.decode(solve_resp_1.output_tokens)
        solve_output_ids_1 = list(solve_resp_1.output_tokens)
        solve_logprobs_1 = list(solve_resp_1.output_logprobs)
        solve_versions_1 = (
            solve_resp_1.output_versions
            if hasattr(solve_resp_1, "output_versions") and solve_resp_1.output_versions
            else [engine0.get_version()] * len(solve_output_ids_1)
        )

        # Solver reward (correctness)
        prompt_str_1 = self.tokenizer.decode(solve_input_ids_1)
        solver_reward_1 = await self.async_reward_fn(
            prompt_str_1,
            solve_text_1,
            solve_input_ids_1,
            solve_output_ids_1,
            **task_data,
        )
        if not isinstance(solver_reward_1, (int, float)):
            solver_reward_1 = float(solver_reward_1)

        sequences.append(_build_seq_dict(
            input_ids=solve_input_ids_1,
            output_ids=solve_output_ids_1,
            output_logprobs=solve_logprobs_1,
            output_versions=list(solve_versions_1),
            model_id=MODEL_ID_SOLVER,
            reward=solver_reward_1,
            is_first=True,
        ))

        round_info_1 = {
            "round": 1,
            "role": "solver",
            "solve_prompt": prompt_str_1,
            "solve": solve_text_1,
            "solver_reward": solver_reward_1,
            "solve_prompt_len": len(solve_input_ids_1),
            "solve_output_len": len(solve_output_ids_1),
        }
        round_infos.append(round_info_1)

        # ---- Verifier ----
        verifier_prompt = VERIFIER_PROMPT_TEMPLATE.format(
            env_prompt=env_prompt, team_context=solve_text_1,
        )
        verify_messages = [{"role": "user", "content": verifier_prompt}]
        verify_input_ids = self._apply_chat_template(
            verify_messages, tokenize=True, add_generation_prompt=True,
        )
        verify_resp = await engine1.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=verify_input_ids,
                gconfig=self.verifier_gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        verify_text = self.tokenizer.decode(verify_resp.output_tokens)
        verify_output_ids = list(verify_resp.output_tokens)
        verify_logprobs = list(verify_resp.output_logprobs)
        verify_versions = (
            verify_resp.output_versions
            if hasattr(verify_resp, "output_versions") and verify_resp.output_versions
            else [engine1.get_version()] * len(verify_output_ids)
        )

        approved, parsed_ok = _parse_verify_tag(verify_text)
        solution_is_correct = solver_reward_1 > 0.5

        if not parsed_ok:
            verifier_reward = 0.0
        else:
            verifier_reward = 1.0 if (approved == solution_is_correct) else 0.0

        sequences.append(_build_seq_dict(
            input_ids=verify_input_ids,
            output_ids=verify_output_ids,
            output_logprobs=verify_logprobs,
            output_versions=list(verify_versions),
            model_id=MODEL_ID_VERIFIER,
            reward=verifier_reward,
            is_first=False,
        ))

        verify_prompt_text = self.tokenizer.decode(verify_input_ids)
        round_info_v = {
            "round": 1,
            "role": "verifier",
            "verify_prompt": verify_prompt_text,
            "verify": verify_text,
            "verdict": "APPROVE" if approved else "REJECT",
            "parsed_ok": parsed_ok,
            "verifier_reward": verifier_reward,
            "verify_prompt_len": len(verify_input_ids),
            "verify_output_len": len(verify_output_ids),
        }
        round_infos.append(round_info_v)

        stats_tracker.get(self.rollout_stat_scope).scalar(
            solver_reward=solver_reward_1,
            verifier_reward=verifier_reward,
        )

        # ---- Round 2: Solver retry (only if rejected) ----
        if not approved:
            solver_prompt_2 = SOLVER_PROMPT_TEMPLATE.format(
                env_prompt=env_prompt, team_context=verify_text,
            )
            solve_messages_2 = [{"role": "user", "content": solver_prompt_2}]
            solve_input_ids_2 = self._apply_chat_template(
                solve_messages_2, tokenize=True, add_generation_prompt=True,
            )
            solve_resp_2 = await engine0.agenerate(
                ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=solve_input_ids_2,
                    gconfig=self.solver_gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
            )
            solve_text_2 = self.tokenizer.decode(solve_resp_2.output_tokens)
            solve_output_ids_2 = list(solve_resp_2.output_tokens)
            solve_logprobs_2 = list(solve_resp_2.output_logprobs)
            solve_versions_2 = (
                solve_resp_2.output_versions
                if hasattr(solve_resp_2, "output_versions") and solve_resp_2.output_versions
                else [engine0.get_version()] * len(solve_output_ids_2)
            )

            prompt_str_2 = self.tokenizer.decode(solve_input_ids_2)
            solver_reward_2 = await self.async_reward_fn(
                prompt_str_2,
                solve_text_2,
                solve_input_ids_2,
                solve_output_ids_2,
                **task_data,
            )
            if not isinstance(solver_reward_2, (int, float)):
                solver_reward_2 = float(solver_reward_2)

            sequences.append(_build_seq_dict(
                input_ids=solve_input_ids_2,
                output_ids=solve_output_ids_2,
                output_logprobs=solve_logprobs_2,
                output_versions=list(solve_versions_2),
                model_id=MODEL_ID_SOLVER,
                reward=solver_reward_2,
                is_first=False,
            ))

            round_info_2 = {
                "round": 2,
                "role": "solver",
                "solve_prompt": prompt_str_2,
                "solve": solve_text_2,
                "solver_reward": solver_reward_2,
                "solve_prompt_len": len(solve_input_ids_2),
                "solve_output_len": len(solve_output_ids_2),
            }
            round_infos.append(round_info_2)
            final_reward = solver_reward_2
            final_solve_text = solve_text_2
        else:
            final_reward = solver_reward_1
            final_solve_text = solve_text_1

        trajectory_info = {
            "prompt": problem_text,
            "n_rounds": len(round_infos),
            "final_solve": final_solve_text,
            "final_reward": final_reward,
            "approved": approved,
            "rounds": round_infos,
        }

        return sequences, final_reward, trajectory_info

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

        # Generate n_samples actor-verify chains in parallel
        sample_coros = [
            self._actor_and_verify_one(engine0, engine1, messages, data)
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

        # Debug dump
        if self.dump_dir is not None and random.random() < 1 / 32:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            dump_qid = qid or uuid.uuid4().hex
            file_path = os.path.join(dump_path, f"{dump_qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, (traj_info, rew) in enumerate(
                    zip(trajectory_infos, rewards)
                ):
                    await f.write(
                        f"=== Sample {i + 1}/{n_samples} "
                        f"(final_reward={rew}, "
                        f"n_rounds={traj_info['n_rounds']}, "
                        f"approved={traj_info['approved']}) ===\n\n"
                    )
                    for ri in traj_info["rounds"]:
                        if ri["role"] == "solver":
                            await f.write(
                                f"--- Round {ri['round']} Solver "
                                f"(reward={ri['solver_reward']}) ---\n"
                                f"--- Solver Prompt ---\n"
                                f"{ri['solve_prompt']}\n\n"
                                f"--- Solver Output ---\n"
                                f"{ri['solve']}\n\n"
                            )
                        else:
                            await f.write(
                                f"--- Round {ri['round']} Verifier "
                                f"[{ri['verdict']}, "
                                f"reward={ri['verifier_reward']}] ---\n"
                                f"--- Verifier Prompt ---\n"
                                f"{ri['verify_prompt']}\n\n"
                                f"--- Verifier Output ---\n"
                                f"{ri['verify']}\n\n"
                            )
                    await f.write("\n")

        # Compute agent-scoped metrics for wandb logging.
        agent_metrics: dict[str, float] = {}
        total_approves = 0
        total_verdicts = 0
        total_retries = 0
        for traj_info in trajectory_infos:
            if not traj_info["approved"]:
                total_retries += 1
            total_verdicts += 1
            if traj_info["approved"]:
                total_approves += 1
        if total_verdicts > 0:
            agent_metrics["verifier_approve_rate"] = total_approves / total_verdicts
            agent_metrics["retry_rate"] = total_retries / total_verdicts

        return {
            "prompt_id": qid,
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "trajectories": trajectories,
            "agent_metrics": agent_metrics,
        }
