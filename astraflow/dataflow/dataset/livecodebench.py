"""Dataset loaders for single-turn LiveCodeBench-style code tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from astraflow.dataflow.dataset.utils import attach_query_ids
from astraflow.workflow.utils import logging

logger = logging.getLogger(__name__)

SINGLE_TURN_LCB_PROMPT_TEMPLATE = """Solve the following coding problem in Python 3.

Return only one final ```python``` code block containing the complete solution.

Question:
{question}
"""


def _normalize_input_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _extract_question(sample: dict[str, Any]) -> str:
    for key in ("question", "prompt", "problem", "content", "instruction", "input"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(
        "LiveCodeBench sample is missing a question field. "
        "Expected one of: question, prompt, problem, content, instruction, input."
    )


def _build_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": SINGLE_TURN_LCB_PROMPT_TEMPLATE.format(question=question),
        }
    ]


def _filter_by_length(dataset, tokenizer, max_length: int | None):
    if max_length is None:
        return dataset
    if tokenizer is None:
        raise ValueError("tokenizer must be provided when max_length is set")

    def filter_length(sample):
        prompt = sample["messages"][0]["content"]
        return len(tokenizer.encode(prompt)) <= max_length

    return dataset.filter(filter_length)


def _load_single_turn_lcb_dataset(
    *,
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    dataset_name: str = "livecodebench",
):
    if not Path(path).exists():
        raise FileNotFoundError(f"LiveCodeBench dataset path does not exist: {path}")

    logger.info("Loading LiveCodeBench single-turn dataset from %s", path)
    dataset = load_dataset("json", data_files=path, split="train")
    # Stamp query_id BEFORE any map/filter so ids reflect source-row position.
    dataset = attach_query_ids(dataset, dataset_name)

    def process(sample, idx: int):
        if "input_output" not in sample:
            raise KeyError(
                "LiveCodeBench sample is missing `input_output`, which is required "
                "for execution-based verification."
            )

        question = _extract_question(sample)
        return {
            **sample,
            "idx": sample.get("idx", idx),
            "source": "livecodebench",
            "question": question,
            "input_output": _normalize_input_output(sample["input_output"]),
            "messages": _build_messages(question),
        }

    dataset = dataset.map(process, with_indices=True)
    dataset = _filter_by_length(dataset, tokenizer, max_length)
    return dataset


def get_livecodebench_single_turn_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    dataset_name: str = "livecodebench",
):
    return _load_single_turn_lcb_dataset(
        path=path,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_name=dataset_name,
    )


def get_livecodebench_single_turn_test_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    dataset_name: str = "livecodebench",
):
    return _load_single_turn_lcb_dataset(
        path=path,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_name=dataset_name,
    )
