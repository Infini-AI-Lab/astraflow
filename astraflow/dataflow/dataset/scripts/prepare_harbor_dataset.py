#!/usr/bin/env python3
"""Prepare local Harbor task datasets from Hugging Face Hub.

Some Harbor training datasets, such as ``open-thoughts/CodeContests``, are
published as parquet rows with:

    path: relative task directory
    task_binary: tar archive bytes for that task directory

This script downloads the dataset snapshot and extracts those task archives into
a local directory that AstraFlow can load with
``get_harbor_task_path_dataset``. By default the output is repo-local:

    ./data-data/harbor/<dataset-repo-name>

Example:

    python astraflow/dataflow/dataset/scripts/prepare_harbor_dataset.py \
      --dataset open-thoughts/CodeContests
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def _is_within(base: Path, target: Path) -> bool:
    try:
        return (
            os.path.commonpath([str(base.resolve()), str(target.resolve())])
            == str(base.resolve())
        )
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    path = PurePosixPath(name)
    parts = [part for part in path.parts if part not in ("..", ".", "", "/")]
    return str(PurePosixPath(*parts)) if parts else ""


def _safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tar:
        for member in tar.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name:
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                continue

            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _safe_relative_path(path: str) -> Path:
    posix_path = PurePosixPath(path)
    parts = [part for part in posix_path.parts if part not in ("..", ".", "", "/")]
    return Path(*parts) if parts else Path("task_unknown")


def _extract_one(args: tuple[str, bytes | bytearray | memoryview, str]) -> bool:
    rel_path, archive_data, output_dir_str = args
    if not isinstance(rel_path, str):
        return False
    if not isinstance(archive_data, bytes | bytearray | memoryview):
        return False

    output_dir = Path(output_dir_str).resolve()
    target_dir = (output_dir / _safe_relative_path(rel_path)).resolve()
    if not _is_within(output_dir, target_dir):
        return False

    if target_dir.exists() and (target_dir / "instruction.md").is_file():
        return True

    try:
        _safe_extract_tar(bytes(archive_data), target_dir)
    except Exception as exc:
        print(f"Warning: failed to extract {rel_path}: {exc}", flush=True)
        return False

    return (target_dir / "instruction.md").is_file()


def _find_task_parquet_files(snapshot_dir: Path) -> list[Path]:
    parquet_files: list[Path] = []
    for parquet_path in snapshot_dir.glob("**/*.parquet"):
        try:
            schema = pq.read_schema(parquet_path)
        except Exception as exc:
            print(f"Warning: could not read schema from {parquet_path}: {exc}")
            continue
        if "path" in schema.names and "task_binary" in schema.names:
            parquet_files.append(parquet_path)
    return parquet_files


def _extract_parquet(
    parquet_path: Path,
    output_dir: Path,
    workers: int,
) -> int:
    table = pq.read_table(parquet_path, columns=["path", "task_binary"])
    paths = table.column("path").to_pylist()
    archives = table.column("task_binary").to_pylist()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(path, archive, str(output_dir)) for path, archive in zip(paths, archives)]
    if workers <= 1:
        return sum(_extract_one(task) for task in tasks)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(_extract_one, tasks, chunksize=64))


def _repo_name(dataset: str) -> str:
    return dataset.rstrip("/").split("/")[-1]


def _default_output_dir(dataset: str) -> Path:
    return Path("./data-data/harbor") / _repo_name(dataset)


def _replace_path_with_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    target.symlink_to(source, target_is_directory=True)


def prepare(
    dataset: str,
    output_dir: str | None = None,
    workers: int = 8,
    direct_mode: str = "symlink",
) -> str:
    output_path = Path(
        os.path.expandvars(output_dir) if output_dir else _default_output_dir(dataset)
    ).expanduser().resolve()

    print(f"Downloading {dataset}...")
    snapshot_dir = Path(snapshot_download(repo_id=dataset, repo_type="dataset"))
    print(f"Downloaded snapshot to {snapshot_dir}")

    parquet_files = _find_task_parquet_files(snapshot_dir)
    if not parquet_files:
        print("No Harbor task parquet files found.")
        if direct_mode == "copy":
            print(f"Copying snapshot to {output_path}...")
            if output_path.exists() or output_path.is_symlink():
                if output_path.is_symlink() or output_path.is_file():
                    output_path.unlink()
                else:
                    shutil.rmtree(output_path)
            shutil.copytree(snapshot_dir, output_path)
        else:
            print(f"Symlinking {output_path} -> {snapshot_dir}")
            _replace_path_with_symlink(output_path, snapshot_dir)
        return str(output_path)

    total = 0
    for parquet_path in parquet_files:
        print(f"Extracting {parquet_path.name}...")
        total += _extract_parquet(parquet_path, output_path, workers=workers)

    print(f"Done. Extracted {total} Harbor task(s) to {output_path}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a Harbor task dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset repo, e.g. open-thoughts/CodeContests.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=None,
        help="Output directory. Defaults to ./data-data/harbor/<dataset-name>.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel extraction workers for parquet task archives.",
    )
    parser.add_argument(
        "--direct-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help=(
            "When no path/task_binary parquet files are found, either symlink "
            "or copy the downloaded snapshot."
        ),
    )
    args = parser.parse_args()
    prepare(
        dataset=args.dataset,
        output_dir=args.output_dir,
        workers=max(1, args.workers),
        direct_mode=args.direct_mode,
    )


if __name__ == "__main__":
    main()
