"""Two-model code workflow v3: v2 behavior with plumbing bugs fixed.

Differences vs ``code_actor_and_verify_v2``:

1. ``_build_retry_messages`` no longer stacks a second ``SINGLE_TURN_LCB_PROMPT_TEMPLATE``
   wrapper on top of ``problem_text``. The dataset already applies that template when it
   builds the user message, so the original v2 prompt ended up with two stacked
   "Solve the following coding problem in Python 3... Question:" headers on every
   retry. v3 strips the wrapper off ``problem_text`` before re-wrapping so the retry
   prompt matches the original single-wrapped shape.

2. ``_normalize_generated_cases`` no longer calls ``_validate_stdio_cases``. The
   v2 heuristic rejected any generated output blob with fewer lines than the
   reference output's max line count, on the assumption that a shorter-than-reference
   output must be a "fragmented" multi-line case split into separate entries. In
   practice it fired on ~23% of testgen outputs, most of which were semantically
   valid cases that just chose fewer queries per case than the reference. Every
   other verifier call in the workflow (attempt1-vs-gt, attempt2-vs-gt,
   reference-vs-gt, eval) decides pass/fail by executing code and comparing stdout;
   there is no reason to gate the generated-case path differently. Genuinely
   malformed generated cases are already caught by ``reference_case_eval``
   (the reference solution won't match them), without the false positives.

3. Retry is gated on real, useful failure. The v2 condition ``if not approve:``
   fired attempt2 whenever anything wasn't a clean approval — including cases
   where the testgen produced no usable JSON (no actionable feedback) and
   cases where attempt1 had already passed all GT tests (no room to improve,
   only room to regress). v3 requires all of:
     * ``generated_cases is not None`` — real per-case PASS/FAIL feedback
       exists (otherwise feedback_text is a content-free "invalid cases"
       string that trains the codegen on noise).
     * ``not approve`` — attempt1 failed at least one generated case.
     * ``not code_eval_1["all_passed"]`` — attempt1 did not already pass all
       GT tests (otherwise attempt2 would be asked to rewrite a correct
       answer, and any regression creates a negative gradient against a
       known-good solution).
   The testgen sequence is still emitted with its natural reward in every
   branch so the model keeps learning to produce valid, discriminating cases.

4. Testgen reward is decoupled from the codegen's outcome. v2 rewarded the
   testgen with ``int(final_code.all_passed) + int(reference_case.all_passed)``,
   which tied the testgen's gradient to the code generator's skill — outside
   its control. v3 replaces this with:
     R_valid = int(reference_case.all_passed)
     R_diag  = int(R_valid AND
                   generated_case_eval.all_passed == code_eval_1.all_passed)
     testcase_reward = 0.5 * (R_valid + R_diag)   ∈ {0, 0.5, 1}
   R_valid asks "did you write valid cases?"; R_diag asks "did your
   accept/reject verdict on attempt1 agree with ground truth?", gated on
   R_valid so right-for-wrong-reasons agreement earns 0. Reward is normalized
   to [0, 1] matching the codegen reward range. The
   ``discrete_testgen_reward=False`` branch uses the continuous
   generalization (pass_rate for R_valid, gated agreement for R_diag).
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

logger = logging.getLogger("CodeActorAndVerifyV3 workflow")

_CONFIGURED_LOOPS: set[int] = set()

MODEL_ID_PROMPT = -1
MODEL_ID_CODE_GENERATOR = 0
MODEL_ID_TESTCASE_GENERATOR = 1


TESTCASE_GENERATOR_SYSTEM = (
    "You are a testcase generator for a Python coding problem. Given the problem "
    "description (with built-in examples removed) and a candidate solution, produce "
    "exactly {generated_case_count} valid test cases that match the original "
    "interface. The cases must follow the problem statement, not the candidate "
    "solution. First think step by step about edge cases and potential bugs in the "
    "candidate inside an <analysis>...</analysis> block. Then output exactly one "
    "```json fenced code block containing the final test cases, and nothing after it."
)


_EXAMPLE_MARKER_RE = re.compile(
    r"(?im)^\s*(?:"
    r"examples?\s*\d*\s*[:.]?"
    r"|sample\s+(?:input|output|tests?)\s*\d*\s*[:.]?"
    r"|sample\s*\d+\s*[:.]?"
    r")\s*$"
)
_SECTION_HEADER_RE = re.compile(
    r"(?im)^\s*(?:constraints?|notes?|note\s*\d*|follow[- ]?up|hints?"
    r"|explanations?)\s*[:.]?\s*$"
)


def _strip_examples(text: str, min_keep_ratio: float = 0.25) -> str:
    """Heuristically strip Example / Sample I/O blocks from a problem statement.

    Cuts from each example marker line to the next recognized section header
    (Constraints / Notes / Follow-up / Hints / Explanation) or to EOF. Falls
    back to the original text if the result drops below ``min_keep_ratio`` of
    the original length — a guard against the heuristic nuking the whole spec.
    """
    if not text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _EXAMPLE_MARKER_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and not _SECTION_HEADER_RE.match(lines[j]):
                j += 1
            i = j
            continue
        out.append(lines[i])
        i += 1
    stripped = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    if not stripped or len(stripped) < min_keep_ratio * len(text):
        return text
    return stripped


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


def _is_function_call(io_spec: dict[str, Any]) -> bool:
    fn_name = io_spec.get("fn_name")
    return isinstance(fn_name, str) and bool(fn_name)


def _case_count(io_spec: dict[str, Any]) -> int:
    return len(io_spec.get("inputs", []))


def _normalize_stdio_blob(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, (dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _truncate_prompt_blob(value: str, limit: int = 160) -> str:
    value = value.replace("\r\n", "\n").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _count_nonempty_lines(value: str) -> int:
    lines = [line for line in value.splitlines() if line.strip()]
    return len(lines) if lines else int(bool(value.strip()))


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


def _extract_json_candidate(text: str) -> dict[str, Any] | list[Any] | None:
    candidates: list[str] = []

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    start_positions = [idx for idx, char in enumerate(text) if char in "[{"]
    end_positions = [idx for idx, char in enumerate(text) if char in "]}"]
    for start in start_positions:
        for end in reversed(end_positions):
            if end <= start:
                continue
            candidates.append(text[start : end + 1].strip())
            break

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _validate_stdio_cases(
    inputs: list[str],
    outputs: list[str],
    reference_io_spec: dict[str, Any],
) -> None:
    reference_inputs = [
        _normalize_stdio_blob(value)
        for value in reference_io_spec.get("inputs", [])
    ]
    reference_outputs = [
        _normalize_stdio_blob(value)
        for value in reference_io_spec.get("outputs", [])
    ]
    ref_input_line_hint = max(
        (_count_nonempty_lines(value) for value in reference_inputs),
        default=1,
    )
    ref_output_line_hint = max(
        (_count_nonempty_lines(value) for value in reference_outputs),
        default=1,
    )

    if ref_input_line_hint > 1:
        for index, blob in enumerate(inputs):
            if _count_nonempty_lines(blob) < 2:
                raise ValueError(
                    "Generated stdin case "
                    f"{index + 1} looks fragmented. Each inputs[i] must be one complete "
                    "stdin blob for a whole case, not a single line from a larger sample."
                )

    if ref_output_line_hint > 1:
        for index, blob in enumerate(outputs):
            if _count_nonempty_lines(blob) < 2:
                raise ValueError(
                    "Generated stdout case "
                    f"{index + 1} looks fragmented. Each outputs[i] must be one complete "
                    "stdout blob for the corresponding stdin case."
                )


def _normalize_generated_cases(
    generated_payload: Any,
    reference_io_spec: dict[str, Any],
    generated_case_count: int,
) -> dict[str, Any]:
    is_function_call = _is_function_call(reference_io_spec)
    reference_fn_name = reference_io_spec.get("fn_name") if is_function_call else None

    if isinstance(generated_payload, list):
        generated_payload = {"cases": generated_payload}

    if isinstance(generated_payload, dict) and ("cases" in generated_payload or "tests" in generated_payload):
        raw_cases = generated_payload.get("cases") or generated_payload.get("tests")
        if not isinstance(raw_cases, list):
            raise ValueError("Generated cases must be a list")
        inputs = []
        outputs = []
        for case in raw_cases:
            if not isinstance(case, dict):
                raise ValueError("Each generated case must be an object")
            if "input" not in case or "output" not in case:
                raise ValueError("Each generated case must contain input and output")
            inputs.append(case["input"])
            outputs.append(case["output"])
        generated_payload = {
            "fn_name": generated_payload.get("fn_name", reference_fn_name),
            "inputs": inputs,
            "outputs": outputs,
        }

    if not isinstance(generated_payload, dict):
        raise ValueError("Generated cases must be a JSON object")

    inputs = generated_payload.get("inputs")
    outputs = generated_payload.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("Generated cases must contain list fields `inputs` and `outputs`")
    if len(inputs) < generated_case_count or len(outputs) < generated_case_count:
        raise ValueError(
            f"Generated cases must contain at least {generated_case_count} inputs and outputs"
        )

    inputs = inputs[:generated_case_count]
    outputs = outputs[:generated_case_count]
    if len(inputs) != len(outputs):
        raise ValueError("Generated cases must have matching input/output counts")

    if is_function_call:
        return {
            "fn_name": reference_fn_name,
            "inputs": inputs,
            "outputs": outputs,
        }

    normalized = {
        "inputs": [_normalize_stdio_blob(value) for value in inputs],
        "outputs": [_normalize_stdio_blob(value) for value in outputs],
    }
    # v3: skip _validate_stdio_cases. The "looks fragmented" line-count
    # heuristic rejected ~23% of testgen outputs in v2 runs (most were
    # semantically valid cases that simply used fewer queries per case than
    # the reference). The reference-solution execution pass (reference_case_eval)
    # catches genuinely malformed cases via pass_rate=0 without the false
    # positives.
    return normalized


def _serialize_io_spec(io_spec: dict[str, Any]) -> str:
    return json.dumps(io_spec, ensure_ascii=False)


def _verifier_problem(io_spec: dict[str, Any], task_data: dict[str, Any], *, suffix: str) -> dict[str, Any]:
    query_id = task_data.get("query_id", task_data.get("idx", task_data.get("id", "unknown")))
    return {
        "input_output": _serialize_io_spec(io_spec),
        "query_id": f"{query_id}-{suffix}",
    }


def _case_schema_text(
    io_spec: dict[str, Any],
    generated_case_count: int,
    include_example: bool = False,
) -> str:
    if _is_function_call(io_spec):
        fn_name = io_spec["fn_name"]
        example_input = None
        example_output = None
        if include_example and io_spec.get("inputs"):
            example_input = _truncate_prompt_blob(
                json.dumps(io_spec["inputs"][0], ensure_ascii=False)
            )
        if include_example and io_spec.get("outputs"):
            example_output = _truncate_prompt_blob(
                json.dumps(io_spec["outputs"][0], ensure_ascii=False)
            )
        return (
            f"Return JSON with exactly this shape for {generated_case_count} cases:\n"
            "{\n"
            f'  "fn_name": "{fn_name}",\n'
            '  "inputs": [<case1_args>, <case2_args>],\n'
            '  "outputs": [<case1_expected>, <case2_expected>]\n'
            "}\n"
            "Each entry in `inputs` must match the function-call format used by the original problem."
            + (
                f"\nExample reference case:\ninputs[0]={example_input}\noutputs[0]={example_output}"
                if example_input is not None and example_output is not None
                else ""
            )
        )
    example_input = None
    example_output = None
    if include_example and io_spec.get("inputs"):
        example_input = _truncate_prompt_blob(_normalize_stdio_blob(io_spec["inputs"][0]))
    if include_example and io_spec.get("outputs"):
        example_output = _truncate_prompt_blob(_normalize_stdio_blob(io_spec["outputs"][0]))
    return (
        f"Return JSON with exactly this shape for {generated_case_count} cases:\n"
        "{\n"
        '  "inputs": ["<stdin case 1>", "<stdin case 2>"],\n'
        '  "outputs": ["<expected stdout case 1>", "<expected stdout case 2>"]\n'
        "}\n"
        "Each `inputs[i]` must be one complete stdin blob for a whole case, including embedded newlines.\n"
        "Never split one stdin example across multiple JSON list entries.\n"
        "Each `outputs[i]` must be the complete expected stdout blob for that input.\n"
        + (
            "Example reference case:\n"
            f"inputs[0]={json.dumps(example_input, ensure_ascii=False)}\n"
            f"outputs[0]={json.dumps(example_output, ensure_ascii=False)}"
            if example_input is not None and example_output is not None
            else ""
        )
    )


def _run_code_and_collect(
    code: str,
    io_spec: dict[str, Any],
    task_data: dict[str, Any],
    verify_timeout: int,
    *,
    suffix: str,
) -> dict[str, Any]:
    problem = _verifier_problem(io_spec, task_data, suffix=suffix)
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


def _format_case_feedback(
    generated_cases: dict[str, Any] | None,
    generated_case_eval: dict[str, Any] | None,
    reference_case_eval: dict[str, Any] | None,
) -> str:
    if generated_cases is None or generated_case_eval is None:
        return "Generated test cases could not be parsed, so no structured execution summary is available."

    lines = ["Generated cases and execution summary:"]
    total_cases = _case_count(generated_cases)
    for index in range(total_cases):
        input_blob = generated_cases["inputs"][index]
        expected_blob = generated_cases["outputs"][index]
        model_details = generated_case_eval.get("details", [])
        model_detail = model_details[index] if index < len(model_details) else {
            "passed": False,
            "output": None,
            "error_message": "No executable candidate code was available.",
        }

        lines.append(f"Case {index + 1}:")
        lines.append(f"input: {json.dumps(input_blob, ensure_ascii=False)}")
        lines.append(f"expected_output: {json.dumps(expected_blob, ensure_ascii=False)}")
        if model_detail.get("passed") is True:
            lines.append(
                "current_code_result: PASS, "
                f"output={json.dumps(model_detail.get('output'), ensure_ascii=False)}"
            )
        else:
            lines.append(
                "current_code_result: FAIL, "
                f"output={json.dumps(model_detail.get('output'), ensure_ascii=False)}, "
                f"error={model_detail.get('error_message', 'Wrong Answer')}"
            )
    return "\n".join(lines)


# Matches the header produced by SINGLE_TURN_LCB_PROMPT_TEMPLATE at the start
# of the dataset-built user prompt. Anchored so it only strips a leading wrapper
# and never touches a mention of the same phrase mid-problem. Falls back to the
# raw text if the wrapper isn't present (other datasets / changed templates).
_LCB_PROMPT_WRAPPER_RE = re.compile(
    r"\ASolve the following coding problem in Python 3\.\s*\n+"
    r"Return only one final ```python``` code block containing the complete solution\.\s*\n+"
    r"Question:\s*\n+",
)


def _strip_lcb_prompt_wrapper(text: str) -> str:
    """Strip the SINGLE_TURN_LCB_PROMPT_TEMPLATE wrapper from a dataset-built prompt.

    The dataset loader wraps each ``{question}`` in the template shown above. When
    the workflow later calls ``_extract_problem_text`` it gets back the entire
    wrapped user message, not the inner question. Re-wrapping that in
    ``_build_retry_messages`` produced a double-nested retry prompt; v3 strips
    the outer wrapper before re-wrapping.
    """
    m = _LCB_PROMPT_WRAPPER_RE.match(text)
    if not m:
        return text
    return text[m.end():].rstrip()


def _build_retry_messages(problem_text: str, feedback: str) -> list[dict[str, str]]:
    """Build retry prompt: same format as the dataset prompt, with feedback appended.

    v3: the incoming ``problem_text`` is the full dataset-wrapped user message
    (output of ``_extract_problem_text``), so we strip the LCB prompt wrapper
    off before inserting it again. This avoids the double-nested "Solve the
    following coding problem in Python 3… Question:" header that v2 produced.
    """
    inner_question = _strip_lcb_prompt_wrapper(problem_text)
    user_content = (
        "Solve the following coding problem in Python 3.\n\n"
        "Return only one final ```python``` code block containing the complete solution.\n\n"
        "Question:\n"
        f"{inner_question}\n\n"
        "Your previous solution failed on some test cases.\n\n"
        f"{feedback}\n\n"
        "Now solve the problem and return the code."
    )
    return [{"role": "user", "content": user_content}]


def _build_testcase_generation_messages(
    problem_text: str,
    code_text: str,
    io_spec: dict[str, Any],
    generated_case_count: int,
    *,
    strip_examples: bool = True,
    include_schema_example: bool = False,
) -> list[dict[str, str]]:
    problem_for_testgen = (
        _strip_examples(problem_text) if strip_examples else problem_text
    )
    return [
        {
            "role": "system",
            "content": TESTCASE_GENERATOR_SYSTEM.format(
                generated_case_count=generated_case_count
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join([
                "# Problem",
                problem_for_testgen,
                "# Candidate Solution",
                code_text,
                "# Required Output Format",
                _case_schema_text(
                    io_spec,
                    generated_case_count,
                    include_example=include_schema_example,
                ),
            ]),
        },
    ]


@register_workflow("code_actor_and_verify_v3")
class CodeActorAndVerifyV3Workflow(RolloutWorkflow):
    """v3 variant: v2 behavior with plumbing bugs fixed.

    Three differences vs ``code_actor_and_verify_v2``:

    * ``_build_retry_messages`` strips the ``SINGLE_TURN_LCB_PROMPT_TEMPLATE``
      wrapper from ``problem_text`` before re-wrapping, so attempt2 prompts no
      longer stack two copies of the "Solve the following coding problem in
      Python 3... Question:" header.
    * ``_normalize_generated_cases`` no longer runs the ``_validate_stdio_cases``
      line-count preflight. Malformed testgen cases are still caught via
      ``reference_case_eval`` (reference solution execution), without the ~23%
      false-rejection tax on valid smaller cases.
    * Retry fires only when all three conditions hold: testgen produced
      usable cases, attempt1 failed at least one of them, AND attempt1 had
      not already passed all GT tests. This blocks two categories of noisy
      attempt2 training: on content-free "invalid cases" feedback, and on
      samples where attempt1 was already fully correct (where any
      regression would train the codegen away from a correct solution).
      The testgen sequence is still emitted with its normal reward in
      every branch.
    * Testgen reward is ``0.5 * (R_valid + R_diag) ∈ {0, 0.5, 1}``, where
      R_valid checks that the reference solution passes the testgen's
      cases, and R_diag (gated on R_valid) checks that attempt1's
      pass/fail on generated cases matches its pass/fail on GT. This
      decouples the testgen's gradient from the code generator's skill.

    All other workflow mechanics — example-stripped testgen prompt,
    reason-then-emit, discrete rewards — match v2.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        enable_testgen_thinking: bool = True,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        gconfigs: dict[str, GenerationHyperparameters] | None = None,
        generated_case_count: int = 2,
        verify_timeout: int = SINGLE_CASE_EXEC_TIMEOUT,
        strip_examples_for_testgen: bool = True,
        discrete_code_reward: bool = True,
        discrete_testgen_reward: bool = True,
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
            self.testgen_gconfig = gconfigs.get(
                "model1", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
        else:
            self.codegen_gconfig = self.gconfig
            self.testgen_gconfig = self.gconfig
        self.enable_thinking = enable_thinking
        self.enable_testgen_thinking = enable_testgen_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.generated_case_count = generated_case_count
        self.verify_timeout = verify_timeout
        self.strip_examples_for_testgen = strip_examples_for_testgen
        self.discrete_code_reward = discrete_code_reward
        self.discrete_testgen_reward = discrete_testgen_reward
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    def _apply_chat_template(self, messages, *, enable_thinking=None, **kwargs):
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        try:
            return list(self.tokenizer.apply_chat_template(
                messages, **kwargs, enable_thinking=enable_thinking,
            ))
        except TypeError:
            return list(self.tokenizer.apply_chat_template(messages, **kwargs))

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

    async def _select_reference_solution(
        self,
        task_data: dict[str, Any],
        gt_io_spec: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any] | None]:
        best_code = None
        best_summary = None
        best_rate = -1.0
        for index, raw_solution in enumerate(task_data.get("solutions", [])):
            code = extract_python_code(str(raw_solution))
            if code is None:
                continue
            summary = await asyncio.to_thread(
                _run_code_and_collect,
                code,
                gt_io_spec,
                task_data,
                self.verify_timeout,
                suffix=f"reference-{index}",
            )
            if summary["pass_rate"] > best_rate:
                best_code = code
                best_summary = summary
                best_rate = summary["pass_rate"]
            if summary["all_passed"]:
                break
        return best_code, best_summary

    async def _run_one(
        self,
        code_engine: InferenceEngine,
        testcase_engine: InferenceEngine,
        messages: list[dict[str, Any]],
        task_data: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], float, dict[str, Any]] | None:
        problem_text = _extract_problem_text(messages)
        gt_io_spec = _load_input_output(task_data["input_output"])
        reference_code, reference_summary = await self._select_reference_solution(
            task_data,
            gt_io_spec,
        )

        round_infos = []

        code_messages_1 = messages
        (
            code_input_ids_1,
            code_output_ids_1,
            code_logprobs_1,
            code_versions_1,
            code_text_1,
        ) = await self._agenerate(code_engine, code_messages_1, self.codegen_gconfig)
        code_prompt_1 = self.tokenizer.decode(code_input_ids_1)
        code_1 = extract_python_code(code_text_1)

        # Build the testgen prompt now so we can launch its LLM call in
        # parallel with the attempt1-gt verifier (Phase 1). Both depend
        # only on attempt1's code, not on each other's results.
        testcase_messages = _build_testcase_generation_messages(
            problem_text=problem_text,
            code_text=code_text_1,
            io_spec=gt_io_spec,
            generated_case_count=self.generated_case_count,
            strip_examples=self.strip_examples_for_testgen,
            include_schema_example=False,
        )

        async def _verify_attempt1_gt():
            if code_1 is None:
                return {"pass_rate": 0.0, "all_passed": False, "details": []}
            return await asyncio.to_thread(
                _run_code_and_collect,
                code_1,
                gt_io_spec,
                task_data,
                self.verify_timeout,
                suffix="attempt1-gt",
            )

        # Phase 1: attempt1-gt verifier (subprocess on a worker thread)
        # and testgen LLM call (HTTP) in parallel. Different resources,
        # both release the asyncio loop, so wall-clock = max of the two.
        # Halves wall-clock on samples where attempt1-gt hits the
        # PROGRAM_HARD_DEADLINE (300s).
        code_eval_1, (
            testcase_input_ids,
            testcase_output_ids,
            testcase_logprobs,
            testcase_versions,
            testcase_text,
        ) = await asyncio.gather(
            _verify_attempt1_gt(),
            self._agenerate(
                testcase_engine,
                testcase_messages,
                self.testgen_gconfig,
                enable_thinking=self.enable_testgen_thinking,
            ),
        )
        code_reward_1 = (
            float(code_eval_1["all_passed"])
            if self.discrete_code_reward
            else code_eval_1["pass_rate"]
        )

        generated_cases = None
        generated_cases_error = None
        testcase_payload = _extract_json_candidate(testcase_text)
        if testcase_payload is None:
            generated_cases_error = "Failed to parse testcase generator output as JSON."
        else:
            try:
                generated_cases = _normalize_generated_cases(
                    testcase_payload,
                    gt_io_spec,
                    self.generated_case_count,
                )
            except Exception as exc:
                generated_cases_error = str(exc)

        # Phase 2: attempt1-generated and reference-generated verifiers
        # in parallel. Both need generated_cases, neither depends on the
        # other's result. Both run as subprocesses on worker threads;
        # benefit caps at the asyncio default-executor pool size, but at
        # minimum we save wall-clock on samples where one verify is
        # slower than the other.
        async def _verify_attempt1_generated():
            if generated_cases is None:
                return None
            if code_1 is None:
                return {
                    "pass_rate": 0.0,
                    "all_passed": False,
                    "details": [
                        {
                            "passed": False,
                            "output": None,
                            "error_message": "No executable candidate code was available.",
                        }
                        for _ in range(_case_count(generated_cases))
                    ],
                }
            return await asyncio.to_thread(
                _run_code_and_collect,
                code_1,
                generated_cases,
                task_data,
                self.verify_timeout,
                suffix="attempt1-generated",
            )

        async def _verify_reference_generated():
            if generated_cases is None or reference_code is None:
                return None
            return await asyncio.to_thread(
                _run_code_and_collect,
                reference_code,
                generated_cases,
                task_data,
                self.verify_timeout,
                suffix="reference-generated",
            )

        generated_case_eval, reference_case_eval = await asyncio.gather(
            _verify_attempt1_generated(),
            _verify_reference_generated(),
        )
        approve = bool(generated_case_eval and generated_case_eval.get("all_passed"))

        feedback_sections = []
        if generated_cases_error is not None:
            feedback_sections.append(
                "Generated test cases were invalid and could not be used as-is: "
                f"{generated_cases_error}"
            )
        else:
            feedback_sections.append(
                _format_case_feedback(
                    generated_cases,
                    generated_case_eval,
                    None,
                )
            )

        feedback_text = "\n\n".join(feedback_sections)

        final_code_text = code_text_1
        final_code_eval = code_eval_1
        final_code_reward = code_reward_1
        code_seq_2 = None
        # v3 retry gating. Attempt2 fires only when ALL of the following hold:
        #   (1) testgen produced usable cases (not a parse/schema failure) —
        #       otherwise feedback_text is a content-free "invalid cases"
        #       string and attempt2 trains the codegen on noise.
        #   (2) attempt1 failed at least one generated case (not approved).
        #   (3) attempt1 did NOT already pass all GT tests — otherwise we'd
        #       be training the codegen to rewrite a correct answer, which
        #       can regress it. Skipping here also keeps final_code_eval
        #       tied to attempt1's correct outcome, so the testgen reward's
        #       "final.all_passed" component stays 1 and doesn't punish the
        #       testgen for doing its job on an edge case attempt1 caught.
        if (
            generated_cases is not None
            and not approve
            and not code_eval_1["all_passed"]
        ):
            code_messages_2 = _build_retry_messages(problem_text, feedback=feedback_text)
            (
                code_input_ids_2,
                code_output_ids_2,
                code_logprobs_2,
                code_versions_2,
                code_text_2,
            ) = await self._agenerate(code_engine, code_messages_2, self.codegen_gconfig)
            code_prompt_2 = self.tokenizer.decode(code_input_ids_2)
            code_2 = extract_python_code(code_text_2)
            if code_2 is None:
                code_eval_2 = {
                    "pass_rate": 0.0,
                    "all_passed": False,
                    "details": [],
                }
                code_reward_2 = 0.0
            else:
                code_eval_2 = await asyncio.to_thread(
                    _run_code_and_collect,
                    code_2,
                    gt_io_spec,
                    task_data,
                    self.verify_timeout,
                    suffix="attempt2-gt",
                )
                code_reward_2 = (
                    float(code_eval_2["all_passed"])
                    if self.discrete_code_reward
                    else code_eval_2["pass_rate"]
                )

            code_seq_2 = _build_seq_dict(
                input_ids=code_input_ids_2,
                output_ids=code_output_ids_2,
                output_logprobs=code_logprobs_2,
                output_versions=code_versions_2,
                model_id=MODEL_ID_CODE_GENERATOR,
                reward=code_reward_2,
                is_first=False,
            )
            round_infos.append({
                "round": 2,
                "role": "code_generator",
                "prompt": code_prompt_2,
                "code": code_text_2,
                "gt_pass_rate": code_eval_2["pass_rate"],
                "reward": code_reward_2,
            })
            final_code_text = code_text_2
            final_code_eval = code_eval_2
            final_code_reward = code_reward_2

        generated_case_good_rate = (
            reference_case_eval["pass_rate"] if reference_case_eval is not None else 0.0
        )
        reference_case_all_passed = bool(
            reference_case_eval is not None and reference_case_eval.get("all_passed")
        )
        # v3 testgen reward: decoupled from the codegen's outcome, normalized to [0, 1].
        #   R_valid = testgen's cases are valid (reference solution passes all of them)
        #   R_diag  = testgen's verdict on attempt1 agrees with GT's verdict
        #             (both passed OR both failed), gated on R_valid so
        #             right-for-wrong-reasons agreement with bogus cases earns 0.
        # testcase_reward = 0.5 * (R_valid + R_diag) ∈ {0, 0.5, 1}.
        r_valid = 1 if reference_case_all_passed else 0
        if (
            r_valid
            and generated_case_eval is not None
            and (generated_case_eval["all_passed"] == code_eval_1["all_passed"])
        ):
            r_diag = 1
        else:
            r_diag = 0
        if self.discrete_testgen_reward:
            testcase_reward = 0.5 * (r_valid + r_diag)
        else:
            # Continuous variant: use pass_rate for R_valid and fraction-agreement
            # for R_diag, still gated on R_valid being ≈ 1.0 (0.99 threshold to
            # absorb float-accumulation noise in pass_rate).
            continuous_valid = generated_case_good_rate
            continuous_diag = 0.0
            if continuous_valid >= 0.99 and generated_case_eval is not None:
                continuous_diag = float(
                    generated_case_eval["all_passed"] == code_eval_1["all_passed"]
                )
            testcase_reward = 0.5 * (continuous_valid + continuous_diag)

        code_seq_1 = _build_seq_dict(
            input_ids=code_input_ids_1,
            output_ids=code_output_ids_1,
            output_logprobs=code_logprobs_1,
            output_versions=code_versions_1,
            model_id=MODEL_ID_CODE_GENERATOR,
            reward=code_reward_1,
            is_first=True,
        )
        testcase_seq = _build_seq_dict(
            input_ids=testcase_input_ids,
            output_ids=testcase_output_ids,
            output_logprobs=testcase_logprobs,
            output_versions=testcase_versions,
            model_id=MODEL_ID_TESTCASE_GENERATOR,
            reward=testcase_reward,
            is_first=False,
        )

        sequences = [code_seq_1, testcase_seq]
        if code_seq_2 is not None:
            sequences.append(code_seq_2)

        round_infos.insert(0, {
            "round": 1,
            "role": "code_generator",
            "prompt": code_prompt_1,
            "code": code_text_1,
            "gt_pass_rate": code_eval_1["pass_rate"],
            "reward": code_reward_1,
        })
        round_infos.insert(1, {
            "round": 1,
            "role": "testcase_generator",
            "prompt": self.tokenizer.decode(testcase_input_ids),
            "generated_cases_text": testcase_text,
            "generated_cases": generated_cases,
            "generated_cases_error": generated_cases_error,
            "generated_case_pass_rate": (
                generated_case_eval["pass_rate"] if generated_case_eval is not None else 0.0
            ),
            "generated_case_good_rate": generated_case_good_rate,
            "reference_case_all_passed": reference_case_all_passed,
            "r_valid": r_valid,
            "r_diag": r_diag,
            "reward": testcase_reward,
            "approve": approve,
            "feedback": feedback_text,
        })

        stats_tracker.get(self.rollout_stat_scope).scalar(
            code_attempt1_reward=code_reward_1,
            final_code_reward=final_code_reward,
            final_gt_pass_rate=final_code_eval["pass_rate"],
            generated_case_good_rate=generated_case_good_rate,
            generated_case_pass_rate=(
                generated_case_eval["pass_rate"] if generated_case_eval is not None else 0.0
            ),
            reference_case_all_passed=float(reference_case_all_passed),
            testgen_r_valid=float(r_valid),
            testgen_r_diag=float(r_diag),
            testcase_reward=testcase_reward,
        )

        trajectory_info = {
            "prompt": problem_text,
            "final_code": final_code_text,
            "final_reward": final_code_reward,
            "final_gt_pass_rate": final_code_eval["pass_rate"],
            "approved": approve,
            "reference_solution_pass_rate": (
                reference_summary["pass_rate"] if reference_summary is not None else None
            ),
            "rounds": round_infos,
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
            testcase_engine = engine["model1"]
        else:
            code_engine = engine
            testcase_engine = engine

        messages = data["messages"]
        n_samples = self.gconfig.n_samples
        version = code_engine.get_version()

        raw_results = await asyncio.gather(*[
            self._run_one(code_engine, testcase_engine, messages, data)
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
                for index, (trajectory_info, reward) in enumerate(zip(trajectory_infos, rewards)):
                    await f.write(
                        f"idx: {index + 1} / {n_samples}, "
                        f"final_reward={reward:.4f}, "
                        f"gt_pass_rate={trajectory_info['final_gt_pass_rate']:.4f}, "
                        f"approved={trajectory_info['approved']}\n"
                    )
                    for round_info in trajectory_info["rounds"]:
                        role = round_info["role"]
                        rnd = round_info["round"]
                        if role == "code_generator":
                            await f.write(
                                f"--- Round {rnd} {role} "
                                f"(gt_pass_rate={round_info['gt_pass_rate']:.4f}, "
                                f"reward={round_info['reward']:.4f}) ---\n"
                            )
                            await f.write(f"prompt is\n{round_info['prompt']}\n")
                            await f.write(f"code is\n{round_info['code']}\n\n")
                        elif role == "testcase_generator":
                            await f.write(
                                f"--- Round {rnd} {role} "
                                f"(reward={round_info['reward']:.4f}, "
                                f"approve={round_info.get('approve', False)}, "
                                f"gen_case_pass_rate={round_info.get('generated_case_pass_rate', 0):.4f}, "
                                f"gen_case_good_rate={round_info.get('generated_case_good_rate', 0):.4f}) ---\n"
                            )
                            await f.write(f"prompt is\n{round_info['prompt']}\n")
                            await f.write(f"generated_cases is\n{round_info.get('generated_cases_text', '')}\n")
                            if round_info.get("generated_cases_error"):
                                await f.write(f"error: {round_info['generated_cases_error']}\n")
                            await f.write(f"feedback is\n{round_info.get('feedback', '')}\n\n")
                    await f.write("\n")

        # eval_correct: binary signal for eval aggregation.
        # 1.0 if ALL ground-truth tests pass (same standard as
        # livecodebench_single_turn), 0.0 otherwise.  The continuous
        # reward (gt_pass_rate × 2.0) is kept for training gradients.
        eval_correct = [
            1.0 if info["final_gt_pass_rate"] == 1.0 else 0.0
            for info in trajectory_infos
        ]

        return {
            "prompt_id": qid,
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "eval_correct": torch.tensor(eval_correct, dtype=torch.float32),
            "trajectories": trajectories,
        }
