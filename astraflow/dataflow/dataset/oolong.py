"""Oolong dataset loaders for the oolong_recursive workflow.

Each row references one Oolong task by `task_id`; the workflow re-loads
the full Task (including the large `context` string) via `get_task` —
keeping the per-row payload tiny and avoiding cache bloat.
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path

from datasets import Dataset

from astraflow.core.workflow.impl.oolong.tasks import (
    Task,
    get_task,
    get_task_ids,
)
from astraflow.dataflow.dataset.utils import attach_query_ids

logger = getLogger(__name__)


def _row_for_task_id(task_id: str) -> dict:
    """Materialize one HF-Dataset row. Schema-uniform across rows."""
    task = get_task(task_id)
    # The agent sees only the goal in the chat; `context` is pre-loaded
    # into the Python sandbox by OolongEnv (not put in the prompt).
    user = f"Goal: {task.goal}"
    return {
        "task_id": task.id,
        "messages": [{"role": "user", "content": user}],
        "answer": task.id,
    }


def get_oolong_rl_dataset(
    tokenizer=None,
    max_length: int | None = None,
    dataset: str = "synth",
    split: str = "validation",
    num_tasks: int | None = None,
    max_context_len: int | None = None,
    seed: int = 42,  # unused — Oolong is deterministic
    dataset_name: str = "oolong",
) -> Dataset:
    """Return a HF Dataset of Oolong tasks for RL training.

    Args:
        dataset: "synth" or "real"
        split:   "validation" or "test" (Oolong has no separate "train" split;
                  we use "validation" as training data, "test" as held-out)
        num_tasks: cap the dataset size (None = all)
        max_context_len: filter out tasks with context_window_text > this length
    """
    task_ids = get_task_ids(
        dataset=dataset, split=split, max_context_len=max_context_len,
    )
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


def get_oolong_eval_dataset(
    tokenizer=None,
    max_length: int | None = None,
    dataset: str = "synth",
    split: str = "test",
    num_val: int | None = 100,
    max_context_len: int | None = None,
    seed: int = 42,
    dataset_name: str = "oolong_val",
) -> Dataset:
    """Return a HF Dataset of Oolong tasks for eval."""
    task_ids = get_task_ids(
        dataset=dataset, split=split, max_context_len=max_context_len,
    )
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
