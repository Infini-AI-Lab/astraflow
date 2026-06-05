"""DeepDive dataset loaders for the deepdive_recursive workflow.

Each row references one DeepDive task by `task_id`; the workflow re-loads
the full Task (question + ground_truth) via `get_task` on demand.
"""

from __future__ import annotations

from logging import getLogger

from datasets import Dataset

from astraflow.core.workflow.impl.deepdive.tasks import (
    Task,
    get_task,
    get_task_ids,
)
from astraflow.dataflow.dataset.utils import attach_query_ids

logger = getLogger(__name__)


def _row_for_task_id(task_id: str) -> dict:
    task = get_task(task_id)
    user = f"Question: {task.goal}"
    return {
        "task_id": task.id,
        "messages": [{"role": "user", "content": user}],
        "answer": task.id,
    }


def get_deepdive_rl_dataset(
    tokenizer=None,
    max_length: int | None = None,
    split: str = "qa_rl",
    num_tasks: int | None = None,
    seed: int = 42,
    dataset_name: str = "deepdive",
) -> Dataset:
    """HF Dataset of DeepDive tasks for RL training (default split: qa_rl)."""
    task_ids = get_task_ids(split=split)
    if num_tasks is not None:
        task_ids = task_ids[:num_tasks]
    rows = [_row_for_task_id(tid) for tid in task_ids]
    ds = Dataset.from_list(rows)
    ds = attach_query_ids(ds, dataset_name)
    if max_length is not None and tokenizer is not None:
        def short_enough(sample):
            content = sample["messages"][0]["content"]
            return len(tokenizer.encode(content)) <= max_length
        ds = ds.filter(short_enough)
    return ds


def get_deepdive_eval_dataset(
    tokenizer=None,
    max_length: int | None = None,
    split: str = "qa_sft",
    num_val: int | None = 100,
    seed: int = 42,
    dataset_name: str = "deepdive_val",
) -> Dataset:
    """HF Dataset of DeepDive tasks for eval (default split: qa_sft)."""
    task_ids = get_task_ids(split=split)
    if num_val is not None:
        task_ids = task_ids[:num_val]
    rows = [_row_for_task_id(tid) for tid in task_ids]
    ds = Dataset.from_list(rows)
    ds = attach_query_ids(ds, dataset_name)
    if max_length is not None and tokenizer is not None:
        def short_enough(sample):
            content = sample["messages"][0]["content"]
            return len(tokenizer.encode(content)) <= max_length
        ds = ds.filter(short_enough)
    return ds
