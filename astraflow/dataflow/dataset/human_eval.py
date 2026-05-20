"""Dataset loaders for HumanEval code-generation tasks."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from astraflow.dataflow.dataset.utils import attach_query_ids
from astraflow.core.workflow.utils import logging

logger = logging.getLogger(__name__)
HF_DATASETS_CACHE_DIR = "/tmp/hf-datasets"

HUMAN_EVAL_PROMPT_TEMPLATE = """Solve the following HumanEval programming problem in Python 3.

Return only one final ```python``` code block containing the complete solution.

Problem:
{prompt}
"""


def _build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": HUMAN_EVAL_PROMPT_TEMPLATE.format(prompt=prompt.strip()),
        }
    ]


def _extract_reference_solution(sample: dict[str, object]) -> list[str]:
    prompt = sample.get("prompt")
    canonical_solution = sample.get("canonical_solution")
    if isinstance(prompt, str) and isinstance(canonical_solution, str):
        return [prompt + canonical_solution]
    return []


def _filter_by_length(dataset, tokenizer, max_length: int | None):
    if max_length is None:
        return dataset
    if tokenizer is None:
        raise ValueError("tokenizer must be provided when max_length is set")

    def filter_length(sample):
        prompt = sample["messages"][0]["content"]
        return len(tokenizer.encode(prompt)) <= max_length

    return dataset.filter(filter_length)


def _load_human_eval_dataset(
    *,
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    dataset_name: str = "human_eval",
):
    del split
    if not Path(path).exists():
        raise FileNotFoundError(f"HumanEval dataset path does not exist: {path}")

    logger.info("Loading HumanEval dataset from %s", path)
    dataset = load_dataset(
        "json",
        data_files=path,
        split="train",
        cache_dir=HF_DATASETS_CACHE_DIR,
    )
    # Stamp query_id BEFORE any map/filter so ids reflect source-row position.
    dataset = attach_query_ids(dataset, dataset_name)

    def process(sample, idx: int):
        prompt = sample["prompt"]
        return {
            **sample,
            "idx": sample.get("idx", idx),
            "source": "human_eval",
            "messages": _build_messages(prompt),
            "solutions": _extract_reference_solution(sample),
        }

    dataset = dataset.map(process, with_indices=True)
    dataset = _filter_by_length(dataset, tokenizer, max_length)
    return dataset


def get_human_eval_test_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
    dataset_name: str = "human_eval",
):
    return _load_human_eval_dataset(
        path=path,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_name=dataset_name,
    )
