"""Download all math training + eval datasets for offline AstraFlow runs.

Layout written to ``--root`` (default: ``./data-data/math``)::

    <root>/
      deepscaler/         # RL train (agentica-org/DeepScaleR-Preview-Dataset)
      dapo_filter/        # RL train (aaabiao/dapo_filter)
      aime24/             # eval (HuggingFaceH4/aime_2024)
      aime25/             # eval (math-ai/aime25)
      amc/                # eval (rawsh/2024_AMC12)
      math500/            # eval (HuggingFaceH4/MATH-500)
      minerva/            # eval (math-ai/minervamath)
      olympiadbench/      # eval (math-ai/olympiadbench)
      MANIFEST.json       # source repo + split + row count for each

The directory names align with the auto-derived ``offline_dir`` convention
used by ``astraflow.dataflow.service`` when ``dataflow.data_root`` is set
in the experiment YAML — so the matching recipe just needs::

    dataflow:
      data_root: ./data-data/math

Example::

    python examples/math/offline/download_math_datasets.py
    python examples/math/offline/download_math_datasets.py --root /scratch/math
    python examples/math/offline/download_math_datasets.py --only deepscaler,aime24
    python examples/math/offline/download_math_datasets.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Each entry: (logical_name, module_path, hf_repo, split)
# logical_name MUST match the offline_dir convention used by
# astraflow.dataflow.service._create_dataset_from_config — i.e. for evals
# the YAML dict key, and for the rollout the dataset_fn module basename.
MATH_DATASETS: list[tuple[str, str, str, str]] = [
    # Training
    ("deepscaler",     "astraflow.dataflow.dataset.deepscaler",     "agentica-org/DeepScaleR-Preview-Dataset", "train"),
    ("dapo_filter",    "astraflow.dataflow.dataset.dapo_filter",    "aaabiao/dapo_filter",                     "train"),
    # Eval
    ("aime24",         "astraflow.dataflow.dataset.aime24x4",       "HuggingFaceH4/aime_2024",                 "train"),
    ("aime25",         "astraflow.dataflow.dataset.aime25x4",       "math-ai/aime25",                          "test"),
    ("amc",            "astraflow.dataflow.dataset.amc24",          "rawsh/2024_AMC12",                        "train"),
    ("math500",        "astraflow.dataflow.dataset.math500",        "HuggingFaceH4/MATH-500",                  "test"),
    ("minerva",        "astraflow.dataflow.dataset.minervamath",    "math-ai/minervamath",                     "test"),
    ("olympiadbench",  "astraflow.dataflow.dataset.olympiadbench",  "math-ai/olympiadbench",                   "test"),
]

logger = logging.getLogger("download_math_datasets")


def _is_populated(p: Path) -> bool:
    """A HF save_to_disk directory contains a dataset_info.json."""
    return p.exists() and (p / "dataset_info.json").exists()


def _download_one(
    name: str,
    module_path: str,
    hf_repo: str,
    split: str,
    out_dir: Path,
    force: bool,
) -> dict:
    import importlib
    from datasets import load_from_disk

    if _is_populated(out_dir) and not force:
        ds = load_from_disk(str(out_dir))
        logger.info("[skip] %-14s already populated (%d rows) at %s", name, len(ds), out_dir)
        return {"name": name, "repo": hf_repo, "split": split, "path": str(out_dir),
                "rows": len(ds), "status": "skipped"}

    mod = importlib.import_module(module_path)
    download_fn = getattr(mod, "download_dataset", None)
    if download_fn is None:
        raise RuntimeError(
            f"{module_path} has no download_dataset() helper — offline mode "
            f"is not supported for this dataset."
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info("[start] %-14s  %s [%s] -> %s", name, hf_repo, split, out_dir)
    download_fn(offline_dir=str(out_dir), dataset_path=hf_repo, split=split)
    ds = load_from_disk(str(out_dir))
    dt = time.time() - t0
    logger.info("[ok]    %-14s  %d rows in %.1fs", name, len(ds), dt)
    return {"name": name, "repo": hf_repo, "split": split, "path": str(out_dir),
            "rows": len(ds), "status": "downloaded", "elapsed_sec": round(dt, 2)}


def _verify_one(name: str, out_dir: Path) -> dict:
    from datasets import load_from_disk
    if not _is_populated(out_dir):
        logger.error("[fail]  %-14s missing or incomplete: %s", name, out_dir)
        return {"name": name, "path": str(out_dir), "status": "missing"}
    ds = load_from_disk(str(out_dir))
    n = len(ds)
    if n == 0:
        logger.error("[fail]  %-14s loaded but empty: %s", name, out_dir)
        return {"name": name, "path": str(out_dir), "status": "empty"}
    logger.info("[ok]    %-14s  %d rows", name, n)
    return {"name": name, "path": str(out_dir), "rows": n, "status": "ok"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default="data-data/math",
                   help="Root directory for offline datasets (default: %(default)s)")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated subset of dataset names (e.g. deepscaler,aime24)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the directory already exists")
    p.add_argument("--verify", action="store_true",
                   help="Skip download; just verify each dataset loads from disk")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    logger.info("offline root: %s", root)

    selected = MATH_DATASETS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [d for d in MATH_DATASETS if d[0] in wanted]
        missing = wanted - {d[0] for d in selected}
        if missing:
            logger.error("Unknown dataset names in --only: %s", sorted(missing))
            return 2
        if not selected:
            logger.error("No datasets matched --only %s", args.only)
            return 2

    manifest: list[dict] = []
    failed: list[str] = []

    for name, module_path, hf_repo, split in selected:
        out_dir = root / name
        try:
            if args.verify:
                entry = _verify_one(name, out_dir)
                if entry["status"] != "ok":
                    failed.append(name)
            else:
                entry = _download_one(name, module_path, hf_repo, split, out_dir, args.force)
        except Exception as e:
            logger.exception("[fail]  %-14s %s", name, e)
            entry = {"name": name, "repo": hf_repo, "split": split,
                     "path": str(out_dir), "status": "failed", "error": str(e)}
            failed.append(name)
        manifest.append(entry)

    if not args.verify:
        manifest_path = root / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("wrote manifest: %s", manifest_path)

    if failed:
        logger.error("FAILED: %s", failed)
        return 1
    logger.info("All %d dataset(s) %s.", len(selected), "verified" if args.verify else "ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
