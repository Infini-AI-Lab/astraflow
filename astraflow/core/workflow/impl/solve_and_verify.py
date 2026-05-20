"""Solve-and-Verify: two-model iterative workflow.

**model0** (solver) generates a solution for the math problem.
**model1** (verifier) checks the solution and replies ACCEPT or REJECT
with concise feedback.

If the verifier rejects, the solver retries with the feedback (up to
``max_rounds`` total solver attempts).  The verifier sees only the latest
attempt each round.  The solver sees only the latest feedback (not its
previous attempts).

Each generation (solver attempt or verifier check) produces an independent
sequence dict with its own ``rewards``:

- Solver reward: 1.0 if ``\\boxed{}`` matches groundtruth, else 0.0
- Verifier reward: 1.0 if the ACCEPT/REJECT decision matches actual
  correctness, else 0.0

Returns the ASearcher-style structured format::

    {
        "n_trajs": int,
        "rewards": Tensor[n_trajs],
        "trajectories": [{"sequences": [seq1, seq2, ...]}, ...],
    }

Usage in YAML config::

    workflow_spec:
      workflow_cls: solve_and_verify
      reward_fn: "math_verify"
      max_rounds: 5
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

logger = logging.getLogger("SolveAndVerify workflow")

# model_ids constants
MODEL_ID_PROMPT = -1
MODEL_ID_SOLVER = 0  # model0
MODEL_ID_VERIFIER = 1  # model1

# Dataset suffix that conflicts with multi-model workflow instructions.
_DATASET_SUFFIXES = [
    "\nLet's think step by step. Please put your final answer within \\boxed{}.",
    "\nPlease put your final answer within \\boxed{}.",
]

SOLVE_SYSTEM = (
    "You are a math problem solver. Given a math problem, solve it step by step "
    "and put the final answer in \\boxed{}."
)

SOLVE_RETRY_SYSTEM = (
    "You are a math problem solver. A previous solution attempt was rejected "
    "with the feedback below. Solve the problem again carefully, addressing "
    "the feedback. Put the final answer in \\boxed{}."
)

VERIFY_SYSTEM = (
    "You are a math solution verifier. Given a problem and a proposed solution, "
    "check if the solution is correct.\n\n"
    "Reply in this exact format:\n"
    "VERDICT: <your decision>\n"
    "FEEDBACK: <your explanation>\n\n"
    "If the solution is correct, write VERDICT: ACCEPT and briefly confirm.\n"
    "If incorrect, write VERDICT: REJECT and explain the error concisely."
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


def _parse_verdict(text: str) -> tuple[bool, str, bool]:
    """Parse VERDICT: ACCEPT/REJECT and FEEDBACK from verifier output.

    Returns (accepted, feedback, parsed_ok).  ``parsed_ok`` is False when
    neither a ``VERDICT:`` tag nor clear accept/reject keywords are found.
    """
    parsed_ok = True

    # Look for VERDICT line — exclude "ACCEPT or REJECT" (model parroting instructions)
    verdict_match = re.search(r"VERDICT:\s*(ACCEPT|REJECT)(?!\s+or\s)", text, re.IGNORECASE)
    if verdict_match is None:
        # Fallback: look for keywords
        text_lower = text.lower()
        has_accept = "accept" in text_lower
        has_reject = "reject" in text_lower
        if has_accept and not has_reject:
            accepted = True
        elif has_reject and not has_accept:
            accepted = False
        else:
            # Cannot determine — treat as parse failure
            accepted = True  # default accept to not waste solver rounds
            parsed_ok = False
    else:
        accepted = verdict_match.group(1).upper() == "ACCEPT"

    # Extract feedback
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if feedback_match is not None:
        feedback = feedback_match.group(1).strip()
    else:
        # Use full text as feedback if no FEEDBACK tag
        feedback = text.strip()

    return accepted, feedback, parsed_ok


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


@register_workflow("solve_and_verify")
class SolveAndVerifyWorkflow(RolloutWorkflow):
    """Two-model solve-and-verify workflow with iterative feedback.

    Parameters
    ----------
    reward_fn : callable or str
        Reward function for checking correctness of solver's answer.
    gconfig : GenerationHyperparameters
        Generation config (n_samples generates n independent sample chains).
    tokenizer : str or PreTrainedTokenizerFast
        Tokenizer (shared by both models).
    max_rounds : int
        Maximum solver attempts (1 = no retry, 5 = up to 4 rejections).
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
        max_rounds: int = 5,
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
        self.max_rounds = max_rounds
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

    async def _solve_and_verify_one(
        self,
        engine0: InferenceEngine,
        engine1: InferenceEngine,
        messages: list[dict],
        task_data: dict[str, Any],
    ) -> tuple[list[dict], float, dict] | None:
        """Run one iterative solve-and-verify chain.

        Returns a list of sequence dicts (variable length), the final
        trajectory reward, and debug info.
        """
        clean_msgs = _strip_dataset_suffix(messages)
        problem_text = ""
        for msg in clean_msgs:
            if msg["role"] == "user":
                problem_text = msg["content"]
                break

        sequences = []
        round_infos = []
        last_solve_text = ""
        feedback = None

        for round_idx in range(self.max_rounds):
            is_first_seq = len(sequences) == 0

            # --- Solver ---
            if feedback is None:
                # First attempt: just the problem
                solve_messages = [
                    {"role": "system", "content": SOLVE_SYSTEM},
                ] + clean_msgs
            else:
                # Retry: problem + feedback
                solve_messages = [
                    {"role": "system", "content": SOLVE_RETRY_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Problem:\n{problem_text}\n\n"
                            f"Feedback on previous attempt:\n{feedback}"
                        ),
                    },
                ]

            solve_input_ids = self._apply_chat_template(
                solve_messages, tokenize=True, add_generation_prompt=True,
            )
            solve_resp = await engine0.agenerate(
                ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=solve_input_ids,
                    gconfig=self.solver_gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
            )
            solve_text = self.tokenizer.decode(solve_resp.output_tokens)
            solve_output_ids = list(solve_resp.output_tokens)
            solve_logprobs = list(solve_resp.output_logprobs)
            solve_versions = (
                solve_resp.output_versions
                if hasattr(solve_resp, "output_versions") and solve_resp.output_versions
                else [engine0.get_version()] * len(solve_output_ids)
            )
            last_solve_text = solve_text

            # Compute solver reward (correctness)
            prompt_str = self.tokenizer.decode(solve_input_ids)
            solver_reward = await self.async_reward_fn(
                prompt_str,
                solve_text,
                solve_input_ids,
                solve_output_ids,
                **task_data,
            )
            if not isinstance(solver_reward, (int, float)):
                solver_reward = float(solver_reward)

            solver_seq = _build_seq_dict(
                input_ids=solve_input_ids,
                output_ids=solve_output_ids,
                output_logprobs=solve_logprobs,
                output_versions=list(solve_versions),
                model_id=MODEL_ID_SOLVER,
                reward=solver_reward,
                is_first=is_first_seq,
            )
            sequences.append(solver_seq)

            solve_prompt_text = self.tokenizer.decode(solve_input_ids)
            round_info = {
                "round": round_idx + 1,
                "solve_prompt": solve_prompt_text,
                "solve": solve_text,
                "solver_reward": solver_reward,
                "solve_prompt_len": len(solve_input_ids),
                "solve_output_len": len(solve_output_ids),
            }

            # Last round: no verification, use the solution as-is
            if round_idx == self.max_rounds - 1:
                round_info["verify"] = "(skipped — final round)"
                round_info["verdict"] = "N/A"
                round_info["verifier_reward"] = None
                round_infos.append(round_info)
                break

            # --- Verifier ---
            verify_messages = [
                {"role": "system", "content": VERIFY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Problem:\n{problem_text}\n\n"
                        f"Proposed solution:\n{solve_text}"
                    ),
                },
            ]
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

            accepted, feedback, parsed_ok = _parse_verdict(verify_text)
            solution_is_correct = solver_reward > 0.5

            if not parsed_ok:
                # Parse failure — penalize verifier, treat as accept to
                # not waste solver rounds
                verifier_reward = 0.0
            else:
                # Verifier reward: 1.0 if decision matches actual correctness
                verifier_reward = 1.0 if (accepted == solution_is_correct) else 0.0

            verifier_seq = _build_seq_dict(
                input_ids=verify_input_ids,
                output_ids=verify_output_ids,
                output_logprobs=verify_logprobs,
                output_versions=list(verify_versions),
                model_id=MODEL_ID_VERIFIER,
                reward=verifier_reward,
                is_first=False,
            )
            sequences.append(verifier_seq)

            verify_prompt_text = self.tokenizer.decode(verify_input_ids)
            round_info["verify_prompt"] = verify_prompt_text
            round_info["verify"] = verify_text
            round_info["verdict"] = "ACCEPT" if accepted else "REJECT"
            round_info["verifier_reward"] = verifier_reward
            round_info["verify_prompt_len"] = len(verify_input_ids)
            round_info["verify_output_len"] = len(verify_output_ids)
            round_infos.append(round_info)

            stats_tracker.get(self.rollout_stat_scope).scalar(
                solver_reward=solver_reward,
                verifier_reward=verifier_reward,
            )

            if accepted:
                break
            # Rejected — loop continues with feedback

        # Final trajectory reward = last solver's reward
        final_reward = solver_reward

        trajectory_info = {
            "prompt": problem_text,
            "n_rounds": len(round_infos),
            "final_solve": last_solve_text,
            "final_reward": final_reward,
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

        # Generate n_samples solve-verify chains in parallel
        sample_coros = [
            self._solve_and_verify_one(engine0, engine1, messages, data)
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
                        f"n_rounds={traj_info['n_rounds']}) ===\n\n"
                    )
                    for ri in traj_info["rounds"]:
                        await f.write(
                            f"--- Round {ri['round']} "
                            f"(solver_reward={ri['solver_reward']}) ---\n"
                            f"--- Solve Prompt (full) ---\n"
                            f"{ri['solve_prompt']}\n\n"
                            f"--- Solve Output (model0) ---\n"
                            f"{ri['solve']}\n\n"
                        )
                        if "verify_prompt" in ri:
                            await f.write(
                                f"--- Verify Prompt (full) ---\n"
                                f"{ri['verify_prompt']}\n\n"
                                f"--- Verify Output (model1) "
                                f"[{ri['verdict']}, "
                                f"verifier_reward={ri['verifier_reward']}] "
                                f"---\n"
                                f"{ri['verify']}\n\n"
                            )
                        else:
                            await f.write(
                                f"--- Verify: {ri['verify']} ---\n\n"
                            )
                    await f.write("\n")

        # Compute agent-scoped metrics for wandb logging.
        # verifier_accept_rate: fraction of verifier decisions that were ACCEPT.
        # avg_turns: average number of solver rounds used per sample.
        agent_metrics: dict[str, float] = {}
        total_accepts = 0
        total_verdicts = 0
        total_rounds = 0
        for traj_info in trajectory_infos:
            n_rounds = traj_info["n_rounds"]
            total_rounds += n_rounds
            for ri in traj_info["rounds"]:
                verdict = ri.get("verdict")
                if verdict in ("ACCEPT", "REJECT"):
                    total_verdicts += 1
                    if verdict == "ACCEPT":
                        total_accepts += 1
        if total_verdicts > 0:
            agent_metrics["verifier_accept_rate"] = total_accepts / total_verdicts
        if len(trajectory_infos) > 0:
            agent_metrics["avg_turns"] = total_rounds / len(trajectory_infos)

        return {
            "prompt_id": qid,
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "trajectories": trajectories,
            "agent_metrics": agent_metrics,
        }
