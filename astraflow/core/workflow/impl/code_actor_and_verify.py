"""Two-model code workflow with testcase generation and one solver retry."""

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

logger = logging.getLogger("CodeActorAndVerify workflow")

MODEL_ID_PROMPT = -1
MODEL_ID_CODE_GENERATOR = 0
MODEL_ID_TESTCASE_GENERATOR = 1


TESTCASE_GENERATOR_SYSTEM = (
    "You are a testcase generator for a Python coding problem. Given the original "
    "problem and a candidate solution, generate exactly {generated_case_count} valid "
    "test cases that match the original interface. The cases must follow the problem "
    "statement, not the candidate solution. Output only one JSON object and no extra text."
)


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
    _validate_stdio_cases(
        normalized["inputs"],
        normalized["outputs"],
        reference_io_spec,
    )
    return normalized


def _serialize_io_spec(io_spec: dict[str, Any]) -> str:
    return json.dumps(io_spec, ensure_ascii=False)


def _verifier_problem(io_spec: dict[str, Any], task_data: dict[str, Any], *, suffix: str) -> dict[str, Any]:
    query_id = task_data.get("query_id", task_data.get("idx", task_data.get("id", "unknown")))
    return {
        "input_output": _serialize_io_spec(io_spec),
        "query_id": f"{query_id}-{suffix}",
    }


def _case_schema_text(io_spec: dict[str, Any], generated_case_count: int) -> str:
    if _is_function_call(io_spec):
        fn_name = io_spec["fn_name"]
        example_input = None
        example_output = None
        if io_spec.get("inputs"):
            example_input = _truncate_prompt_blob(
                json.dumps(io_spec["inputs"][0], ensure_ascii=False)
            )
        if io_spec.get("outputs"):
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
    if io_spec.get("inputs"):
        example_input = _truncate_prompt_blob(_normalize_stdio_blob(io_spec["inputs"][0]))
    if io_spec.get("outputs"):
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


def _build_retry_messages(problem_text: str, feedback: str) -> list[dict[str, str]]:
    """Build retry prompt: same format as the dataset prompt, with feedback appended."""
    user_content = (
        "Solve the following coding problem in Python 3.\n\n"
        "Return only one final ```python``` code block containing the complete solution.\n\n"
        "Question:\n"
        f"{problem_text}\n\n"
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
) -> list[dict[str, str]]:
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
                problem_text,
                "# Candidate Solution",
                code_text,
                "# Required Output Format",
                _case_schema_text(io_spec, generated_case_count),
            ]),
        },
    ]


@register_workflow("code_actor_and_verify")
class CodeActorAndVerifyWorkflow(RolloutWorkflow):
    """Code generation workflow with testcase generation and one retry."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        gconfigs: dict[str, GenerationHyperparameters] | None = None,
        generated_case_count: int = 2,
        verify_timeout: int = SINGLE_CASE_EXEC_TIMEOUT,
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
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.generated_case_count = generated_case_count
        self.verify_timeout = verify_timeout
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    def _apply_chat_template(self, messages, **kwargs):
        from astraflow.core.workflow.utils.hf_utils import apply_chat_template_to_ids
        return apply_chat_template_to_ids(
            self.tokenizer, messages, enable_thinking=self.enable_thinking, **kwargs
        )

    async def _agenerate(
        self,
        engine: InferenceEngine,
        messages: list[dict[str, str]],
        gconfig: GenerationHyperparameters,
    ) -> tuple[list[int], list[int], list[float], list[int], str]:
        input_ids = self._apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
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

        if code_1 is None:
            code_eval_1 = {
                "pass_rate": 0.0,
                "all_passed": False,
                "details": [],
            }
            code_reward_1 = 0.0
        else:
            code_eval_1 = await asyncio.to_thread(
                _run_code_and_collect,
                code_1,
                gt_io_spec,
                task_data,
                self.verify_timeout,
                suffix="attempt1-gt",
            )
            code_reward_1 = code_eval_1["pass_rate"]

        testcase_messages = _build_testcase_generation_messages(
            problem_text=problem_text,
            code_text=code_text_1,
            io_spec=gt_io_spec,
            generated_case_count=self.generated_case_count,
        )
        (
            testcase_input_ids,
            testcase_output_ids,
            testcase_logprobs,
            testcase_versions,
            testcase_text,
        ) = await self._agenerate(
            testcase_engine,
            testcase_messages,
            self.testgen_gconfig,
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

        generated_case_eval = None
        reference_case_eval = None
        approve = False
        if generated_cases is not None and code_1 is not None:
            generated_case_eval = await asyncio.to_thread(
                _run_code_and_collect,
                code_1,
                generated_cases,
                task_data,
                self.verify_timeout,
                suffix="attempt1-generated",
            )
            approve = generated_case_eval["all_passed"]
        elif generated_cases is not None:
            generated_case_eval = {
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

        if generated_cases is not None and reference_code is not None:
            reference_case_eval = await asyncio.to_thread(
                _run_code_and_collect,
                reference_code,
                generated_cases,
                task_data,
                self.verify_timeout,
                suffix="reference-generated",
            )

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
        if not approve:
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
                code_reward_2 = code_eval_2["pass_rate"]

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
        testcase_reward = final_code_eval["pass_rate"] + generated_case_good_rate

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
