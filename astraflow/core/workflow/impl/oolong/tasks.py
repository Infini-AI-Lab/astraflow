"""Task loading for Oolong from HuggingFace, with on-disk JSONL cache.

Mirrors the platoon loader (oolong-synth / oolong-real). On first call
we hit HuggingFace (`oolongbench/oolong-synth` / `oolong-real`), cache the
examples to `<this dir>/oolong_{dataset}_{split}.jsonl`, and from then on
load from disk.

A `Task` here is a lightweight dataclass mirroring platoon's:
  Task(goal=question, id=task_id, max_steps=..., misc={...example...})
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_DATA_DIR = Path(__file__).parent
_VALID_DATASETS = ("synth", "real")
_DEFAULT_MAX_STEPS = 50

_TASKS: dict[str, "Task"] = {}
_RAW_BY_FILE: dict[Path, list[dict[str, Any]]] = {}


@dataclass
class Task:
    """Lightweight task carrier matching platoon's shape."""
    goal: str
    id: str
    max_steps: int = _DEFAULT_MAX_STEPS
    misc: dict[str, Any] = field(default_factory=dict)


def _hf_name(dataset: str) -> str:
    return f"oolongbench/oolong-{dataset}"


def _load_from_hf(dataset: str, split: str) -> list[dict[str, Any]]:
    """Hit HF; return list of dicts."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        ) from exc

    name = _hf_name(dataset)
    if dataset == "real":
        ds = load_dataset(name, "dnd", split=split)
    else:
        ds = load_dataset(name, split=split)
    return [dict(ex) for ex in ds]


def _cache_path(dataset: str, split: str) -> Path:
    return _DATA_DIR / f"oolong_{dataset}_{split}.jsonl"


def _ensure_cached(dataset: str, split: str) -> list[dict[str, Any]]:
    """Return parsed raw rows, downloading/caching from HF if needed."""
    path = _cache_path(dataset, split)
    if path in _RAW_BY_FILE:
        return _RAW_BY_FILE[path]

    if not path.exists():
        rows = _load_from_hf(dataset, split)
        # Atomic write
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


def _example_to_task(example: dict[str, Any], dataset: str, split: str, idx: int) -> Task:
    """Convert one HF row into a Task. Pre-extracts the context for env use."""
    task_id = f"oolong.{dataset}.{split}.{idx}"
    # Normalize: HF field is `context_window_text`; we expose it as `context`.
    misc = dict(example)
    if "context_window_text" in misc and "context" not in misc:
        misc["context"] = misc.pop("context_window_text")
    misc.pop("context_window_text_with_labels", None)
    return Task(
        goal=str(example.get("question", "")),
        id=task_id,
        max_steps=_DEFAULT_MAX_STEPS,
        misc=misc,
    )


def get_task_ids(
    dataset: Literal["synth", "real"] = "synth",
    split: Literal["validation", "test"] = "validation",
    max_context_len: int | None = None,
) -> list[str]:
    """Return the ordered list of task IDs for a (dataset, split)."""
    if dataset not in _VALID_DATASETS:
        raise ValueError(f"dataset must be in {_VALID_DATASETS}, got {dataset!r}")
    rows = _ensure_cached(dataset, split)
    ids: list[str] = []
    for idx, ex in enumerate(rows):
        if max_context_len is not None:
            ctx = ex.get("context_window_text") or ex.get("context") or ""
            if len(ctx) > max_context_len:
                continue
        ids.append(f"oolong.{dataset}.{split}.{idx}")
    return ids


def get_task(task_id: str) -> Task:
    """Return a deep-copy of the Task for a given ID."""
    if task_id in _TASKS:
        return deepcopy(_TASKS[task_id])

    parts = task_id.split(".")
    if len(parts) != 4 or parts[0] != "oolong":
        raise ValueError(f"Invalid task ID: {task_id!r}")
    _, dataset, split, idx_s = parts
    idx = int(idx_s)
    rows = _ensure_cached(dataset, split)
    if idx >= len(rows):
        raise IndexError(f"Task index {idx} out of range for {dataset}/{split}")
    task = _example_to_task(rows[idx], dataset, split, idx)
    _TASKS[task_id] = task
    return deepcopy(task)
