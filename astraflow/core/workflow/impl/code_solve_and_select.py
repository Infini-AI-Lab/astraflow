"""Two-model code workflow: solver (two parallel attempts) + selector.

High-level pipeline per sample:

  1. The solver (``model0``) produces two independent code attempts in parallel
     from the same prompt.
  2. Each attempt is executed against the ground-truth I/O spec to yield
     ``code_eval_A`` / ``code_eval_B`` and per-code binary rewards
     ``r_A`` = ``float(all_passed_A)``, ``r_B`` = ``float(all_passed_B)``.
  3. The selector (``model1``) sees the problem plus both extracted code
     blocks and emits ``<final>A</final>`` or ``<final>B</final>``.
  4. Selector reward is shaped so it only gets gradient from samples where
     the pick actually matters:

       * parse failure (no valid tag)    →  0.0
       * both codes pass / both fail     →  0.5  (neutral — no signal)
       * exactly one passes, picked it   →  1.0
       * exactly one passes, picked other →  0.0

     Same philosophy as ``code_actor_and_verify_v3``'s ``R_diag``: decouple
     the second actor's reward from the first actor's skill so it isn't
     rewarded or punished for outcomes it did not control.

Episode emits three sequences per sample: [solver_A, solver_B, selector].
``final_code_reward`` tracks the picked code's ground-truth reward; a parse
failure defaults the pick to A for bookkeeping but the selector's own reward
is 0.0. ``eval_correct`` is 1.0 iff the picked code passes all GT tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import aiofiles
import aiofiles.os
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import EngineGroup, InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id
from astraflow.core.workflow.utils.code_execution_mraas import (
    SINGLE_CASE_EXEC_TIMEOUT,
    call_verify_collect_all,
    extract_python_code,
)

logger = logging.getLogger("CodeSolveAndSelect workflow")

_CONFIGURED_LOOPS: set[int] = set()

MODEL_ID_PROMPT = -1
MODEL_ID_CODE_GENERATOR = 0
MODEL_ID_SELECTOR = 1


SELECTOR_SYSTEM = (
    "You are a code selector. You will be shown a Python coding problem and "
    "two candidate solutions, Candidate A and Candidate B. Analyze both "
    "solutions for correctness: check the algorithm, edge cases, off-by-one "
    "errors, and whether each matches the problem specification. First think "
    "step by step inside an <analysis>...</analysis> block. Then output "
    "exactly one of <final>A</final> or <final>B</final> on its own line and "
    "nothing after it."
)


# Matches the header produced by SINGLE_TURN_LCB_PROMPT_TEMPLATE at the start
# of the dataset-built user prompt. Same regex as v3; repeated here so the
# workflow stays self-contained.
_LCB_PROMPT_WRAPPER_RE = re.compile(
    r"\ASolve the following coding problem in Python 3\.\s*\n+"
    r"Return only one final ```python``` code block containing the complete solution\.\s*\n+"
    r"Question:\s*\n+",
)


def _strip_lcb_prompt_wrapper(text: str) -> str:
    m = _LCB_PROMPT_WRAPPER_RE.match(text)
    if not m:
        return text
    return text[m.end():].rstrip()


def _build_seq_dict(
    input_ids: list[int],
    output_ids: list[int],
    output_logprobs: list[float],
    output_versions: list[int],
    model_id: int,
    reward: float,
    is_first: bool,
) -> dict[str, Any]:
    full_ids = input_ids + output_ids
    total_len = len(full_ids)
    prompt_len = len(input_ids)
    output_len = len(output_ids)

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * prompt_len + list(output_logprobs), dtype=torch.float32
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [0] * prompt_len + [1] * output_len, dtype=torch.int32
        ).unsqueeze(0),
        "model_ids": torch.cat([
            torch.full((prompt_len,), MODEL_ID_PROMPT, dtype=torch.long),
            torch.full((output_len,), model_id, dtype=torch.long),
        ]).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * prompt_len + list(output_versions), dtype=torch.int32
        ).unsqueeze(0),
        "attention_mask": torch.ones(total_len, dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "begin_of_trajectory": torch.tensor([int(is_first)]),
    }


def _extract_problem_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _load_input_output(input_output: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(input_output, str):
        return json.loads(input_output)
    return input_output


def _case_count(io_spec: dict[str, Any]) -> int:
    return len(io_spec.get("inputs", []))


def _synthesize_failure_detail(
    io_spec: dict[str, Any],
    case_index: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    message = info.get(
        "error_message",
        "Verifier wrapper failed before producing per-case details.",
    )
    return {
        "passed": False,
        "inputs": io_spec["inputs"][case_index],
        "expected": io_spec["outputs"][case_index],
        "output": None,
        "error": info.get("error"),
        "error_code": info.get("error_code", -4),
        "error_message": message,
    }


def _run_code_and_collect(
    code: str,
    io_spec: dict[str, Any],
    task_data: dict[str, Any],
    verify_timeout: int,
    *,
    suffix: str,
) -> dict[str, Any]:
    query_id = task_data.get("query_id", task_data.get("idx", task_data.get("id", "unknown")))
    problem = {
        "input_output": json.dumps(io_spec, ensure_ascii=False),
        "query_id": f"{query_id}-{suffix}",
    }
    results, info = call_verify_collect_all(
        problem=problem,
        generation=code,
        timeout=verify_timeout,
    )
    details = list(info.get("details", []))
    total_cases = _case_count(io_spec)
    results = list(results[:total_cases]) + [False] * max(0, total_cases - len(results))
    if len(details) < total_cases:
        for case_index in range(len(details), total_cases):
            details.append(_synthesize_failure_detail(io_spec, case_index, info))
    passed_count = sum(1 for detail in details if detail.get("passed") is True)
    pass_rate = passed_count / total_cases if total_cases else 0.0
    return {
        "results": results,
        "details": details,
        "passed_count": passed_count,
        "total_cases": total_cases,
        "pass_rate": pass_rate,
        "all_passed": total_cases > 0 and passed_count == total_cases,
    }


# Anchored on <final>...</final>; case-insensitive; last match wins so
# reasoning that mentions the tag form earlier (e.g. "I will output <final>A</final>")
# doesn't beat the concluding verdict.
_FINAL_TAG_RE = re.compile(r"<\s*final\s*>\s*([AB])\s*<\s*/\s*final\s*>", re.IGNORECASE)


def _parse_selector_choice(text: str) -> str | None:
    matches = _FINAL_TAG_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].upper()


def _format_code_block(code_text: str) -> str:
    """Extract the python code from a solver output; fall back to a sentinel
    if extraction fails so the selector sees something stable to reason about.
    """
    code = extract_python_code(code_text)
    if code is None or not code.strip():
        return "```\n<no valid python code block was produced>\n```"
    return f"```python\n{code.rstrip()}\n```"


def _build_selector_messages(
    problem_text: str,
    code_text_A: str,
    code_text_B: str,
) -> list[dict[str, str]]:
    inner_question = _strip_lcb_prompt_wrapper(problem_text)
    user_content = "\n\n".join([
        "# Problem",
        inner_question,
        "# Candidate A",
        _format_code_block(code_text_A),
        "# Candidate B",
        _format_code_block(code_text_B),
        "Decide which candidate is more likely to be correct (it is possible "
        "that neither is fully correct, or that both are; in those cases pick "
        "the one you judge more trustworthy). Reason inside an "
        "<analysis>...</analysis> block, then emit exactly one of "
        "<final>A</final> or <final>B</final> on its own line and nothing "
        "after it.",
    ])
    return [
        {"role": "system", "content": SELECTOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]


@register_workflow("code_solve_and_select")
class CodeSolveAndSelectWorkflow(RolloutWorkflow):
    """Solver generates two codes in parallel; selector picks one.

    Solver reward:
      r_A = float(code_A.all_passed), r_B = float(code_B.all_passed)
      (or pass_rate if ``discrete_code_reward=False``).

    Selector reward:
      * parse failure                →  0.0
      * c_A == c_B (both pass/both fail) →  0.5
      * mixed, picked correct        →  1.0
      * mixed, picked wrong          →  0.0

    Emits three sequences per sample: [solver_A, solver_B, selector].
    ``final_code_reward`` is the picked code's reward (A on parse failure).
    ``eval_correct`` is 1.0 iff the picked code passes all GT tests.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        enable_selector_thinking: bool = True,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        gconfigs: dict[str, GenerationHyperparameters] | None = None,
        verify_timeout: int = SINGLE_CASE_EXEC_TIMEOUT,
        discrete_code_reward: bool = True,
    ):
        del reward_fn
        if isinstance(tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        if gconfigs is not None:
            self.codegen_gconfig = gconfigs.get(
                "model0", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
            self.selector_gconfig = gconfigs.get(
                "model1", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
        else:
            self.codegen_gconfig = self.gconfig
            self.selector_gconfig = self.gconfig
        self.enable_thinking = enable_thinking
        self.enable_selector_thinking = enable_selector_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.verify_timeout = verify_timeout
        self.discrete_code_reward = discrete_code_reward
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    def _apply_chat_template(self, messages, *, enable_thinking=None, **kwargs):
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        from astraflow.core.workflow.utils.hf_utils import apply_chat_template_to_ids
        return apply_chat_template_to_ids(
            self.tokenizer, messages, enable_thinking=enable_thinking, **kwargs
        )

    async def _agenerate(
        self,
        engine: InferenceEngine,
        messages: list[dict[str, str]],
        gconfig: GenerationHyperparameters,
        *,
        enable_thinking: bool | None = None,
    ) -> tuple[list[int], list[int], list[float], list[int], str]:
        input_ids = self._apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        response = await engine.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        output_ids = list(response.output_tokens)
        output_logprobs = list(response.output_logprobs)
        output_versions = (
            response.output_versions
            if hasattr(response, "output_versions") and response.output_versions
            else [engine.get_version()] * len(output_ids)
        )
        output_text = self.tokenizer.decode(output_ids)
        return input_ids, output_ids, output_logprobs, list(output_versions), output_text

    def _code_reward(self, code_eval: dict[str, Any]) -> float:
        return (
            float(code_eval["all_passed"])
            if self.discrete_code_reward
            else float(code_eval["pass_rate"])
        )

    async def _run_one(
        self,
        code_engine: InferenceEngine,
        selector_engine: InferenceEngine,
        messages: list[dict[str, Any]],
        task_data: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
        problem_text = _extract_problem_text(messages)
        gt_io_spec = _load_input_output(task_data["input_output"])

        # Two independent solver rollouts in parallel from the same prompt.
        solver_A_task = self._agenerate(code_engine, messages, self.codegen_gconfig)
        solver_B_task = self._agenerate(code_engine, messages, self.codegen_gconfig)
        (
            (
                code_input_ids_A,
                code_output_ids_A,
                code_logprobs_A,
                code_versions_A,
                code_text_A,
            ),
            (
                code_input_ids_B,
                code_output_ids_B,
                code_logprobs_B,
                code_versions_B,
                code_text_B,
            ),
        ) = await asyncio.gather(solver_A_task, solver_B_task)

        code_A = extract_python_code(code_text_A)
        code_B = extract_python_code(code_text_B)

        empty_eval = {"pass_rate": 0.0, "all_passed": False, "details": []}

        async def _eval(code: str | None, suffix: str) -> dict[str, Any]:
            if code is None:
                return empty_eval
            return await asyncio.to_thread(
                _run_code_and_collect,
                code,
                gt_io_spec,
                task_data,
                self.verify_timeout,
                suffix=suffix,
            )

        code_eval_A, code_eval_B = await asyncio.gather(
            _eval(code_A, "attemptA-gt"),
            _eval(code_B, "attemptB-gt"),
        )

        r_A = self._code_reward(code_eval_A)
        r_B = self._code_reward(code_eval_B)
        c_A = bool(code_eval_A["all_passed"])
        c_B = bool(code_eval_B["all_passed"])

        # Build & run the selector after both solver outputs are known.
        selector_messages = _build_selector_messages(
            problem_text=problem_text,
            code_text_A=code_text_A,
            code_text_B=code_text_B,
        )
        (
            selector_input_ids,
            selector_output_ids,
            selector_logprobs,
            selector_versions,
            selector_text,
        ) = await self._agenerate(
            selector_engine,
            selector_messages,
            self.selector_gconfig,
            enable_thinking=self.enable_selector_thinking,
        )

        pick = _parse_selector_choice(selector_text)
        parse_success = pick is not None
        is_wrong_correct_pair = c_A != c_B
        picked_correct = parse_success and (
            (pick == "A" and c_A) or (pick == "B" and c_B)
        )

        if not parse_success:
            r_sel = 0.0
            effective_pick = "A"
        elif c_A == c_B:
            r_sel = 0.5
            effective_pick = pick
        else:
            r_sel = 1.0 if picked_correct else 0.0
            effective_pick = pick

        final_code_text = code_text_A if effective_pick == "A" else code_text_B
        final_code_eval = code_eval_A if effective_pick == "A" else code_eval_B
        final_code_reward = r_A if effective_pick == "A" else r_B

        solver_seq_A = _build_seq_dict(
            input_ids=code_input_ids_A,
            output_ids=code_output_ids_A,
            output_logprobs=code_logprobs_A,
            output_versions=code_versions_A,
            model_id=MODEL_ID_CODE_GENERATOR,
            reward=r_A,
            is_first=True,
        )
        solver_seq_B = _build_seq_dict(
            input_ids=code_input_ids_B,
            output_ids=code_output_ids_B,
            output_logprobs=code_logprobs_B,
            output_versions=code_versions_B,
            model_id=MODEL_ID_CODE_GENERATOR,
            reward=r_B,
            is_first=False,
        )
        selector_seq = _build_seq_dict(
            input_ids=selector_input_ids,
            output_ids=selector_output_ids,
            output_logprobs=selector_logprobs,
            output_versions=selector_versions,
            model_id=MODEL_ID_SELECTOR,
            reward=r_sel,
            is_first=False,
        )
        sequences = [solver_seq_A, solver_seq_B, selector_seq]

        stats_tracker.get(self.rollout_stat_scope).scalar(
            code_a_reward=r_A,
            code_b_reward=r_B,
            selector_reward=r_sel,
            final_code_reward=final_code_reward,
            final_gt_pass_rate=final_code_eval["pass_rate"],
            both_correct=float(c_A and c_B),
            both_wrong=float(not c_A and not c_B),
            mixed=float(is_wrong_correct_pair),
            # Same quantity as ``mixed`` under a name that maps directly onto
            # the user's wandb dashboard vocabulary ("wrong-correct pair").
            wrong_correct_pair_rate=float(is_wrong_correct_pair),
            selector_parse_success=float(parse_success),
            picked_A=float(effective_pick == "A"),
        )
        # Conditional metric: accuracy of the selector specifically on
        # wrong-correct pairs. Only emitted on those samples, so the tracker's
        # mean over the scalar list IS the conditional accuracy directly —
        # no division needed in the dashboard. Parse failures on mixed
        # samples count as incorrect (selector produced no valid pick).
        if is_wrong_correct_pair:
            stats_tracker.get(self.rollout_stat_scope).scalar(
                selector_acc_on_wrong_correct_pair=float(picked_correct),
            )

        trajectory_info = {
            "prompt": problem_text,
            "code_text_A": code_text_A,
            "code_text_B": code_text_B,
            "code_a_pass_rate": code_eval_A["pass_rate"],
            "code_b_pass_rate": code_eval_B["pass_rate"],
            "code_a_reward": r_A,
            "code_b_reward": r_B,
            "selector_prompt": self.tokenizer.decode(selector_input_ids),
            "selector_text": selector_text,
            "selector_pick": pick,
            "selector_effective_pick": effective_pick,
            "selector_reward": r_sel,
            "selector_parse_success": parse_success,
            "final_pick": effective_pick,
            "final_code": final_code_text,
            "final_reward": final_code_reward,
            "final_gt_pass_rate": final_code_eval["pass_rate"],
            # Needed by ``arun_episode`` to compute the agent_metrics payload
            # that actually reaches wandb (under the ``agent/`` namespace).
            "is_wrong_correct_pair": is_wrong_correct_pair,
            "picked_correct": picked_correct,
        }
        return sequences, final_code_reward, trajectory_info

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        if id(loop) not in _CONFIGURED_LOOPS:
            loop.set_default_executor(ThreadPoolExecutor(max_workers=64))
            _CONFIGURED_LOOPS.add(id(loop))

        if isinstance(engine, EngineGroup):
            code_engine = engine["model0"]
            selector_engine = engine["model1"]
        else:
            code_engine = engine
            selector_engine = engine

        messages = data["messages"]
        n_samples = self.gconfig.n_samples
        version = code_engine.get_version()

        raw_results = await asyncio.gather(*[
            self._run_one(code_engine, selector_engine, messages, data)
            for _ in range(n_samples)
        ])

        trajectories = []
        trajectory_infos = []
        rewards = []
        for result in raw_results:
            if result is None:
                continue
            sequences, reward, trajectory_info = result
            trajectories.append({"sequences": sequences})
            trajectory_infos.append(trajectory_info)
            rewards.append(reward)

        # Canonical helper — same id the curator gate saw on this prompt.
        qid = resolve_prompt_id(data)

        if self.dump_dir is not None and random.random() < 1 / 32:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            dump_qid = qid or uuid.uuid4().hex
            file_path = os.path.join(dump_path, f"{dump_qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for index, (info, reward) in enumerate(zip(trajectory_infos, rewards)):
                    await f.write(
                        f"idx: {index + 1} / {n_samples}, "
                        f"final_reward={reward:.4f}, "
                        f"gt_pass_rate={info['final_gt_pass_rate']:.4f}, "
                        f"pick={info['final_pick']}, "
                        f"parse_ok={info['selector_parse_success']}\n"
                    )
                    await f.write(
                        f"--- Solver A (pass_rate={info['code_a_pass_rate']:.4f}, "
                        f"reward={info['code_a_reward']:.4f}) ---\n"
                    )
                    await f.write(f"{info['code_text_A']}\n\n")
                    await f.write(
                        f"--- Solver B (pass_rate={info['code_b_pass_rate']:.4f}, "
                        f"reward={info['code_b_reward']:.4f}) ---\n"
                    )
                    await f.write(f"{info['code_text_B']}\n\n")
                    await f.write(
                        f"--- Selector (pick={info['selector_pick']}, "
                        f"reward={info['selector_reward']:.4f}) ---\n"
                    )
                    await f.write(f"prompt is\n{info['selector_prompt']}\n")
                    await f.write(f"selector_text is\n{info['selector_text']}\n\n")
                    await f.write("\n")

        # eval_correct: binary signal for eval aggregation.
        # 1.0 iff the picked code passes all GT tests.
        eval_correct = [
            1.0 if info["final_gt_pass_rate"] == 1.0 else 0.0
            for info in trajectory_infos
        ]

        # agent_metrics: the ONE path that actually reaches wandb. Values land
        # under the ``agent/`` namespace via ``data_acquisition._ingest_structured_result``
        # → ``data_serving.accumulate_agent_metrics`` → ``service.py`` buffer_stats.
        # ``selector_acc_on_wrong_correct_pair`` is only emitted on episodes that
        # actually contain mixed samples, so the accumulated mean across episodes
        # is the conditional accuracy (episodes with no mixed samples don't
        # contribute, preserving the "acc on mixed" semantics).
        agent_metrics: dict[str, float] = {}
        if trajectory_infos:
            n = len(trajectory_infos)
            mixed_infos = [i for i in trajectory_infos if i["is_wrong_correct_pair"]]
            agent_metrics["wrong_correct_pair_rate"] = len(mixed_infos) / n
            if mixed_infos:
                agent_metrics["selector_acc_on_wrong_correct_pair"] = (
                    sum(1 for i in mixed_infos if i["picked_correct"])
                    / len(mixed_infos)
                )

        return {
            "prompt_id": qid,
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "eval_correct": torch.tensor(eval_correct, dtype=torch.float32),
            "trajectories": trajectories,
            "agent_metrics": agent_metrics,
        }
