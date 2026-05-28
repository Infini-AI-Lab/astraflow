"""Task loader for DeepDive (`zai-org/DeepDive`) — Q&A web-research benchmark.

Mirrors the platoon loader. Splits:
  - qa_rl  : intended for RL training (used as our train split)
  - qa_sft : intended for SFT / eval  (used as our held-out eval split)

A `Task` here is a lightweight dataclass:
    Task(goal=question, id=task_id, max_steps=..., misc={"ground_truth": answer, ...})

On first call we hit HuggingFace, cache the rows to a local JSONL, and
from then on load from disk. The cache file is gitignored.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_DATA_DIR = Path(__file__).parent
_VALID_SPLITS = ("qa_rl", "qa_sft")
_DEFAULT_MAX_STEPS = 50
_HF_DATASET = "zai-org/DeepDive"

_TASKS: dict[str, "Task"] = {}
_RAW_BY_FILE: dict[Path, list[dict[str, Any]]] = {}


@dataclass
class Task:
    """Lightweight task carrier matching platoon's shape."""
    goal: str
    id: str
    max_steps: int = _DEFAULT_MAX_STEPS
    misc: dict[str, Any] = field(default_factory=dict)


def _load_from_hf(split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        ) from exc
    ds = load_dataset(_HF_DATASET, split=split)
    return [dict(ex) for ex in ds]


def _cache_path(split: str) -> Path:
    return _DATA_DIR / f"deepdive_{split}.jsonl"


def _ensure_cached(split: str) -> list[dict[str, Any]]:
    path = _cache_path(split)
    if path in _RAW_BY_FILE:
        return _RAW_BY_FILE[path]

    if not path.exists():
        rows = _load_from_hf(split)
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)
    else:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    _RAW_BY_FILE[path] = rows
    return rows


def _example_to_task(example: dict[str, Any], split: str, idx: int) -> Task:
    task_id = f"deepdive.{split}.{idx}"
    misc = dict(example)
    # Normalize: ground truth goes under a predictable key for the grader.
    misc["ground_truth"] = str(example.get("answer", ""))
    misc["dataset_split"] = split
    misc["dataset_index"] = idx
    return Task(
        goal=str(example.get("question", "")),
        id=task_id,
        max_steps=_DEFAULT_MAX_STEPS,
        misc=misc,
    )


def get_task_ids(split: Literal["qa_rl", "qa_sft"] = "qa_rl") -> list[str]:
    """Return the ordered list of task IDs for a split."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be in {_VALID_SPLITS}, got {split!r}")
    rows = _ensure_cached(split)
    return [f"deepdive.{split}.{idx}" for idx in range(len(rows))]


def get_task(task_id: str) -> Task:
    """Return a deep-copy of the Task for a given ID."""
    if task_id in _TASKS:
        return deepcopy(_TASKS[task_id])

    parts = task_id.split(".")
    if len(parts) != 3 or parts[0] != "deepdive":
        raise ValueError(f"Invalid task ID: {task_id!r}")
    _, split, idx_s = parts
    if split not in _VALID_SPLITS:
        raise ValueError(f"Invalid DeepDive split: {split!r}")
    idx = int(idx_s)
    rows = _ensure_cached(split)
    if idx >= len(rows):
        raise IndexError(f"Task index {idx} out of range for split {split}")
    task = _example_to_task(rows[idx], split, idx)
    _TASKS[task_id] = task
    return deepcopy(task)
