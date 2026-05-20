"""Dataset loaders for DeepCoder Preview single-turn code tasks."""

from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

from astraflow.dataflow.dataset.livecodebench import (
    SINGLE_TURN_LCB_PROMPT_TEMPLATE,
)
from astraflow.dataflow.dataset.utils import attach_query_ids
from astraflow.core.workflow.utils import logging

logger = logging.getLogger(__name__)

DEFAULT_HF_PATH = "agentica-org/DeepCoder-Preview-Dataset"


def _build_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": SINGLE_TURN_LCB_PROMPT_TEMPLATE.format(question=question),
        }
    ]


def _infer_test_type(test: dict[str, Any]) -> str | None:
    explicit_type = test.get("type")
    if explicit_type in {"stdin_stdout", "function_call"}:
        return explicit_type
    if "input" not in test or "output" not in test:
        return None
    if isinstance(test.get("fn_name"), str) and test["fn_name"]:
        return "function_call"
    return "stdin_stdout"


def _normalize_tests(tests: Any) -> dict[str, Any]:
    if isinstance(tests, str):
        tests = json.loads(tests)
    if not isinstance(tests, list) or len(tests) == 0:
        raise ValueError("DeepCoder Preview sample must contain a non-empty `tests` list")

    normalized_tests: list[dict[str, Any]] = []
    test_types: set[str] = set()
    for test in tests:
        if not isinstance(test, dict):
            raise ValueError("Each DeepCoder Preview test must be a dict")
        test_type = _infer_test_type(test)
        if test_type not in {"stdin_stdout", "function_call"}:
            raise ValueError(
                f"Unsupported test format {test!r}; expected either explicit "
                "`type` or at least `input`/`output` fields"
            )
        normalized_tests.append(test)
        test_types.add(test_type)

    if len(test_types) != 1:
        raise ValueError(
            f"Mixed test types within one sample are unsupported: {sorted(test_types)}"
        )

    test_type = next(iter(test_types))
    if test_type == "stdin_stdout":
        inputs: list[str] = []
        outputs: list[str] = []
        for test in normalized_tests:
            if "input" not in test or "output" not in test:
                raise ValueError(
                    "Each stdin_stdout test must contain `input` and `output`"
                )
            inputs.append(str(test["input"]))
            outputs.append(str(test["output"]))
        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    fn_names = {test.get("fn_name") for test in normalized_tests}
    if len(fn_names) != 1:
        raise ValueError(
            f"Function-call tests must agree on one fn_name, got {sorted(fn_names)}"
        )
    fn_name = next(iter(fn_names))
    if not isinstance(fn_name, str) or not fn_name:
        raise ValueError("Function-call tests must contain a non-empty `fn_name`")

    inputs: list[Any] = []
    outputs: list[Any] = []
    for test in normalized_tests:
        if "input" not in test or "output" not in test:
            raise ValueError(
                "Each function_call test must contain `input` and `output`"
            )
        inputs.append(test["input"])
        outputs.append(test["output"])

    return {
        "fn_name": fn_name,
        "inputs": inputs,
        "outputs": outputs,
    }


def _is_supported_stdin_stdout_row(sample: dict[str, Any]) -> bool:
    tests = sample.get("tests")
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            return False
    if not isinstance(tests, list) or len(tests) == 0:
        return False
    if not all(isinstance(test, dict) for test in tests):
        return False
    types = {_infer_test_type(test) for test in tests}
    if not types:
        return False
    return types in ({"stdin_stdout"}, {"function_call"})


def _load_deepcoder_preview_dataset(
    *,
    subset: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    hf_path: str = DEFAULT_HF_PATH,
    dataset_name: str = "deepcoder",
):
    logger.info(
        "Loading DeepCoder Preview dataset %s subset=%s split=%s",
        hf_path,
        subset,
        split,
    )
    dataset = load_dataset(hf_path, subset, split=split)
    # Stamp query_id on the freshly-loaded dataset, BEFORE the select()
    # below removes unsupported test rows. select() preserves per-row
    # values, so kept rows keep their original-position ids.
    dataset = attach_query_ids(dataset, dataset_name)
    original_size = len(dataset)
    kept_indices = [
        idx for idx, sample in enumerate(dataset) if _is_supported_stdin_stdout_row(sample)
    ]
    dataset = dataset.select(kept_indices)
    filtered_size = len(dataset)
    logger.info(
        "DeepCoder Preview kept %d/%d rows after filtering to supported test types",
        filtered_size,
        original_size,
    )

    def process(sample, idx: int):
        question = sample.get("problem")
        if not isinstance(question, str) or not question.strip():
            raise KeyError("DeepCoder Preview sample is missing a non-empty `problem` field")

        input_output = _normalize_tests(sample.get("tests"))
        return {
            **sample,
            "idx": sample.get("idx", idx),
            "source": "deepcoder_preview",
            "question": question.strip(),
            "input_output": json.dumps(input_output, ensure_ascii=False),
            "messages": _build_messages(question.strip()),
        }

    disable_progress_bar()
    try:
        dataset = dataset.map(
            process,
            with_indices=True,
            load_from_cache_file=False,
        )
    finally:
        enable_progress_bar()
    mapped_size = len(dataset)
    logger.info(
        "DeepCoder Preview produced %d mapped rows before length filtering",
        mapped_size,
    )
    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when max_length is set")
        kept_indices = []
        for idx, sample in enumerate(dataset):
            prompt = sample["messages"][0]["content"]
            if len(tokenizer.encode(prompt)) <= max_length:
                kept_indices.append(idx)
        dataset = dataset.select(kept_indices)
    final_size = len(dataset)
    logger.info(
        "DeepCoder Preview kept %d/%d rows after prompt length filtering",
        final_size,
        mapped_size,
    )
    return dataset


def get_deepcoder_preview_primeintellect_rl_dataset(
    split: str = "train",
    tokenizer=None,
    max_length: int | None = None,
    subset: str = "primeintellect",
    hf_path: str = DEFAULT_HF_PATH,
    dataset_name: str = "deepcoder_primeintellect",
):
    return _load_deepcoder_preview_dataset(
        subset=subset,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        hf_path=hf_path,
        dataset_name=dataset_name,
    )


def get_deepcoder_preview_codeforces_test_dataset(
    split: str = "test",
    tokenizer=None,
    max_length: int | None = None,
    subset: str = "codeforces",
    hf_path: str = DEFAULT_HF_PATH,
    dataset_name: str = "deepcoder_codeforces",
):
    return _load_deepcoder_preview_dataset(
        subset=subset,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        hf_path=hf_path,
        dataset_name=dataset_name,
    )
