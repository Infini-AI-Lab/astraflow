# Offline math datasets

Pre-download every dataset used by the math recipes so training can run on
a node with no internet access.

## 1. Download (one-time)

From the repo root:

```bash
python examples/math/offline/download_math_datasets.py --root data-data/math
```

This writes 8 dataset directories under `data-data/math/` and a
`MANIFEST.json` summary.  Re-running is a no-op (skips populated dirs);
pass `--force` to re-download, or `--only deepscaler,aime24` for a subset.

| dir              | HF source                                   | split | use     |
|------------------|---------------------------------------------|-------|---------|
| `deepscaler`     | `agentica-org/DeepScaleR-Preview-Dataset`   | train | rollout |
| `dapo_filter`    | `aaabiao/dapo_filter`                       | train | rollout |
| `aime24`         | `HuggingFaceH4/aime_2024`                   | train | eval    |
| `aime25`         | `math-ai/aime25`                            | test  | eval    |
| `amc`            | `rawsh/2024_AMC12`                          | train | eval    |
| `math500`        | `HuggingFaceH4/MATH-500`                    | test  | eval    |
| `minerva`        | `math-ai/minervamath`                       | test  | eval    |
| `olympiadbench`  | `math-ai/olympiadbench`                     | test  | eval    |

## 2. Verify

```bash
python examples/math/offline/download_math_datasets.py --verify
```

Loads every directory with `load_from_disk` and prints row counts; exits
non-zero if any dataset is missing or empty.

## 3. Run training with offline data

The matching recipe is `examples/math/offline/qwen3-8b-m2po-full-offline/`.  Its
`experiment.yaml` sets `dataflow.data_root: data-data/math`, which causes
`astraflow.dataflow.service` to auto-derive each loader's `offline_dir`
as `data-data/math/<name>` (the dict key for evals, or the `dataset_fn`
module name for the rollout).  No per-entry edits required.

```bash
bash examples/math/offline/qwen3-8b-m2po-full-offline/scripts/run_qwen3-8b-m2po-full-offline.sh
```

## Notes

- **Model weights are *not* covered.**  `model_path` / `tokenizer_path`
  still point at `Qwen/Qwen3-8B` and will be pulled from HF Hub on first
  use.  Either let HF cache them once, or pre-fetch with
  `huggingface-cli download Qwen/Qwen3-8B` and point the YAML at the
  local snapshot for a fully air-gapped run.
- Convention: a dataset directory name in `--root` must match the
  `name` used by `_create_dataset_from_config` (eval dict key, or
  rollout `dataset_fn` module basename).  The download script and the
  service use the same `MATH_DATASETS` table / derivation, so they stay
  in sync.
