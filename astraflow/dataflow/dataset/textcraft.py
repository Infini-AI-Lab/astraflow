"""TextCraft dataset for the recursive_agent workflow.

Each row is a task spec:
  {
    "messages": [{"role": "user", "content": <goal text>}],
    "task_id": "textcraft.train.42",
    "target_items": {...},
    "initial_inventory": {...},
    "max_steps": 50,
    "answer": "<task_id>",   # for AstraFlow's prompt_id machinery
  }

The workflow reads ``task_id`` (or falls back to the inline fields) and
materializes a Task via ``textcraft.tasks.get_task``.

Tasks are SYNTHESIZED locally from the bundled recipe database — no
network. ``download_dataset(offline_dir, ...)`` writes the materialized
jsonl files (textcraft_train.jsonl / textcraft_val.jsonl) into the
textcraft package directory so subsequent loads are deterministic.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from logging import getLogger
from pathlib import Path

from datasets import Dataset

from astraflow.core.workflow.impl.textcraft.tasks import (
    create_textcraft_datasets,
    get_task,
    get_task_ids,
)
from astraflow.dataflow.dataset.utils import attach_query_ids

logger = getLogger(__name__)


_TEXTCRAFT_PKG_DIR = Path(__file__).resolve().parents[2] / "core" / "workflow" / "impl" / "textcraft"


def _ensure_tasks_generated(
    num_train: int = 1000,
    num_val: int = 100,
    seed: int = 42,
    force: bool = False,
) -> tuple[Path, Path]:
    """Materialize textcraft_train.jsonl + textcraft_val.jsonl beside the
    workflow code, regenerating if missing OR if the existing files are
    smaller than the requested counts.

    Returns (train_path, val_path).
    """
    train_path = _TEXTCRAFT_PKG_DIR / "textcraft_train.jsonl"
    val_path = _TEXTCRAFT_PKG_DIR / "textcraft_val.jsonl"
    if not force and train_path.exists() and val_path.exists():
        train_lines = sum(1 for _ in open(train_path))
        val_lines = sum(1 for _ in open(val_path))
        if train_lines >= num_train and val_lines >= num_val:
            return train_path, val_path
        logger.info(
            "existing textcraft files too small (train %d<%d or val %d<%d); regenerating",
            train_lines, num_train, val_lines, num_val,
        )
    logger.info(
        "generating textcraft tasks: %d train / %d val (seed=%d)",
        num_train, num_val, seed,
    )
    train_tasks, val_tasks = create_textcraft_datasets(
        seed=seed, num_samples_train=num_train, num_samples_val=num_val,
    )
    with open(train_path, "w") as f:
        for t in train_tasks:
            f.write(json.dumps(asdict(t)) + "\n")
    with open(val_path, "w") as f:
        for t in val_tasks:
            f.write(json.dumps(asdict(t)) + "\n")
    logger.info("wrote %d train tasks → %s", len(train_tasks), train_path)
    logger.info("wrote %d val tasks → %s",   len(val_tasks),   val_path)
    return train_path, val_path


def download_dataset(
    offline_dir: str | None = None,
    num_train: int = 1000,
    num_val: int = 100,
    seed: int = 42,
):
    """Generate the train/val jsonl files (no network)."""
    _ensure_tasks_generated(num_train=num_train, num_val=num_val, seed=seed, force=True)


def _row_for_task_id(task_id: str) -> dict:
    """Materialize a HuggingFace-Dataset row for a single task.

    Keep the schema uniform across rows so HF Dataset doesn't merge dict
    keys: only ``task_id`` (workflow re-loads the full Task via
    ``get_task``) and a single ``messages`` field.
    """
    task = get_task(task_id)
    target_str = ", ".join(
        f"{c}x {it}" for it, c in (task.misc.get("target_items") or {}).items()
    )
    user = (
        f"Task: {task.goal or 'Craft target items'}\n"
        f"Targets: {target_str}\n"
        f"Initial inventory: {json.dumps(task.misc.get('initial_inventory') or {})}\n"
        f"Step budget: {task.max_steps}"
    )
    return {
        "task_id": task.id,
        "messages": [{"role": "user", "content": user}],
        "answer": task.id,  # placeholder for AstraFlow's prompt_id resolution
    }


def get_textcraft_rl_dataset(
    tokenizer=None,
    max_length: int | None = None,
    num_tasks: int = 1000,
    num_val: int = 100,
    seed: int = 42,
    offline_dir: str | None = None,
    dataset_name: str = "textcraft",
) -> Dataset:
    """Return a HF Dataset of TextCraft train tasks."""
    _ensure_tasks_generated(num_train=num_tasks, num_val=num_val, seed=seed)
    task_ids = get_task_ids("train", num_samples_train=num_tasks)
    rows = [_row_for_task_id(tid) for tid in task_ids]
    ds = Dataset.from_list(rows)
    ds = attach_query_ids(ds, dataset_name)
    if max_length is not None and tokenizer is not None:
        def short_enough(sample):
            content = sample["messages"][0]["content"]
            return len(tokenizer.encode(content)) <= max_length
        ds = ds.filter(short_enough)
    return ds


def get_textcraft_eval_dataset(
    tokenizer=None,
    max_length: int | None = None,
    num_val: int = 100,
    num_train: int = 1000,
    seed: int = 42,
    offline_dir: str | None = None,
    dataset_name: str = "textcraft_val",
) -> Dataset:
    """Return a HF Dataset of TextCraft val tasks."""
    _ensure_tasks_generated(num_train=num_train, num_val=num_val, seed=seed)
    task_ids = get_task_ids("val", num_samples_val=num_val)
    rows = [_row_for_task_id(tid) for tid in task_ids]
    ds = Dataset.from_list(rows)
    ds = attach_query_ids(ds, dataset_name)
    if max_length is not None and tokenizer is not None:
        def short_enough(sample):
            content = sample["messages"][0]["content"]
            return len(tokenizer.encode(content)) <= max_length
        ds = ds.filter(short_enough)
    return ds
