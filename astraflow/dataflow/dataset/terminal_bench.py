"""Dataset loaders for Terminal-Bench tasks run through Harbor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from astraflow.dataflow.dataset.utils import attach_query_ids


TERMINAL_BENCH_2_TASKS: tuple[str, ...] = (
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "break-filter-js-from-html",
    "build-cython-ext",
    "build-pmars",
    "build-pov-ray",
    "caffe-cifar-10",
    "cancel-async-tasks",
    "chess-best-move",
    "circuit-fibsqrt",
    "cobol-modernization",
    "code-from-image",
    "compile-compcert",
    "configure-git-webserver",
    "constraints-scheduling",
    "count-dataset-tokens",
    "crack-7z-hash",
    "custom-memory-heap-crash",
    "db-wal-recovery",
    "distribution-search",
    "dna-assembly",
    "dna-insert",
    "extract-elf",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "filter-js-from-html",
    "financial-document-processor",
    "fix-code-vulnerability",
    "fix-git",
    "fix-ocaml-gc",
    "gcode-to-text",
    "git-leak-recovery",
    "git-multibranch",
    "gpt2-codegolf",
    "headless-terminal",
    "hf-model-inference",
    "install-windows-3.11",
    "kv-store-grpc",
    "large-scale-text-editing",
    "largest-eigenval",
    "llm-inference-batching-scheduler",
    "log-summary-date-ranges",
    "mailman",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "merge-diff-arc-agi-task",
    "model-extraction-relu-logits",
    "modernize-scientific-stack",
    "mteb-leaderboard",
    "mteb-retrieve",
    "multi-source-data-merger",
    "nginx-request-logging",
    "openssl-selfsigned-cert",
    "overfull-hbox",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-c-py",
    "polyglot-rust-c",
    "portfolio-optimization",
    "protein-assembly",
    "prove-plus-comm",
    "pypi-server",
    "pytorch-model-cli",
    "pytorch-model-recovery",
    "qemu-alpine-ssh",
    "qemu-startup",
    "query-optimize",
    "raman-fitting",
    "regex-chess",
    "regex-log",
    "reshard-c4-data",
    "rstan-to-pystan",
    "sam-cell-seg",
    "sanitize-git-repo",
    "schemelike-metacircular-eval",
    "sparql-university",
    "sqlite-db-truncate",
    "sqlite-with-gcov",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "tune-mjcf",
    "video-processing",
    "vulnerable-secret",
    "winning-avg-corewars",
    "write-compressor",
)


def _normalise_task_row(sample: dict[str, Any], idx: int) -> dict[str, Any]:
    task_name = (
        sample.get("task_name")
        or sample.get("name")
        or sample.get("task_id")
        or sample.get("index")
    )
    row = {
        **sample,
        "index": sample.get("index", task_name if task_name is not None else idx),
        "task_name": task_name,
        "messages": sample.get("messages", []),
        "source": sample.get("source", "terminal_bench"),
    }
    if row["task_name"] is not None:
        row["task_name"] = str(row["task_name"])
    return row


def _is_harbor_task_dir(path: Path) -> bool:
    return path.is_dir() and (path / "instruction.md").is_file()


def _find_harbor_task_dirs(path: str) -> list[Path]:
    root = Path(os.path.expandvars(path)).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Harbor task dataset path does not exist: {path}")

    if _is_harbor_task_dir(root):
        return [root]

    direct_children = [
        child for child in sorted(root.iterdir()) if _is_harbor_task_dir(child)
    ]
    if direct_children:
        return direct_children

    return sorted(
        child
        for child in root.rglob("*")
        if _is_harbor_task_dir(child)
    )


def get_harbor_task_path_dataset(
    path: str,
    split: str = "train",
    tokenizer=None,
    max_length: int | None = None,
    max_samples: int | None = None,
    dataset_name: str = "harbor_tasks",
):
    """Create a dataset of local Harbor task directories.

    This matches SkyRL's Harbor dataset shape: a prepared directory such as
    ``~/data/harbor/CodeContests`` contains task subdirectories, each with an
    ``instruction.md`` file.  The Harbor workflow runs each sample via
    ``harbor run --path <task_dir>``.
    """
    del split, tokenizer, max_length

    task_dirs = _find_harbor_task_dirs(path)
    if max_samples is not None:
        task_dirs = task_dirs[: int(max_samples)]

    dataset = Dataset.from_list(
        [
            {
                "task_path": str(task_dir),
                "prompt": str(task_dir),
                "task_name": task_dir.name,
                "index": task_dir.name,
                "messages": [],
                "source": "harbor_task_path",
            }
            for task_dir in task_dirs
        ]
    )
    return attach_query_ids(dataset, dataset_name)


def get_terminal_bench_2_test_dataset(
    path: str | None = None,
    split: str = "test",
    tokenizer=None,
    max_length: int | None = None,
    max_samples: int | None = None,
    dataset_name: str = "terminal_bench_2",
    tasks: list[str] | None = None,
):
    """Create an eval dataset for Terminal-Bench 2.0 via Harbor.

    By default this returns the 89 Terminal-Bench 2.0 task names currently
    published in Harbor's registry, so AstraFlow submits one Harbor trial per
    benchmark task.  ``tasks`` or ``path`` can be supplied to evaluate a subset
    or a locally maintained task-name manifest.
    """
    del tokenizer, max_length

    if tasks is not None:
        dataset = Dataset.from_list(
            [
                {"task_name": str(task), "index": str(task), "messages": []}
                for task in tasks
            ]
        )
    elif path is not None:
        dataset = load_dataset("json", data_files=path, split="train")
    else:
        dataset = Dataset.from_list(
            [
                {"task_name": task, "index": task, "messages": []}
                for task in TERMINAL_BENCH_2_TASKS
            ]
        )

    dataset = attach_query_ids(dataset, dataset_name)
    dataset = dataset.map(_normalise_task_row, with_indices=True)

    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))

    return dataset
