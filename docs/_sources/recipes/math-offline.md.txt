# Math (Offline)

Run the math RL recipe on a node with **no internet access** by pre-downloading every training and evaluation dataset to a local directory.

**Recipe**: [`examples/math/offline/qwen3-8b-m2po-full-offline/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/math/offline/qwen3-8b-m2po-full-offline)

**Downloader**: [`examples/math/offline/download_math_datasets.py`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/math/offline/download_math_datasets.py)

This is the same Qwen3-8B / M2PO / TCP recipe as [Math](math.md), with one difference: at startup the AstraFlow service loads every dataset from disk instead of fetching from the HuggingFace Hub.

## 1. One-time prep — download datasets

From the repo root:

```bash
python examples/math/offline/download_math_datasets.py --root data-data/math
```

This writes 8 dataset directories under `data-data/math/` (~400 MB total) plus a `MANIFEST.json`:

| Directory       | HF source                                   | Split | Use     |
|-----------------|---------------------------------------------|-------|---------|
| `deepscaler`    | `agentica-org/DeepScaleR-Preview-Dataset`   | train | rollout |
| `dapo_filter`   | `aaabiao/dapo_filter`                       | train | rollout |
| `aime24`        | `HuggingFaceH4/aime_2024`                   | train | eval    |
| `aime25`        | `math-ai/aime25`                            | test  | eval    |
| `amc`           | `rawsh/2024_AMC12`                          | train | eval    |
| `math500`       | `HuggingFaceH4/MATH-500`                    | test  | eval    |
| `minerva`       | `math-ai/minervamath`                       | test  | eval    |
| `olympiadbench` | `math-ai/olympiadbench`                     | test  | eval    |

Re-running is idempotent (skips populated dirs). Useful flags:

- `--force` — re-download even if a directory exists
- `--only deepscaler,aime24` — partial subset
- `--verify` — skip download; just load each from disk and assert non-empty

## 2. Run training

```bash
bash examples/math/offline/qwen3-8b-m2po-full-offline/scripts/run_qwen3-8b-m2po-full-offline.sh
```

You can confirm the offline path is active by looking for these lines in the AstraFlow service log:

```text
Auto-derived offline_dir for dataset 'deepscaler': data-data/math/deepscaler
Loading DeepScaleR dataset from offline path: data-data/math/deepscaler
Auto-derived offline_dir for dataset 'aime24': data-data/math/aime24
... (same for aime25, amc, minerva, math500)
```

## How it works

The recipe's `experiment.yaml` sets a single field under `dataflow`:

```yaml
dataflow:
  data_root: data-data/math
```

At startup `astraflow.dataflow.service` walks every entry in `rollout_dataset` and `eval_datasets`; for each one that does not already specify `offline_dir`, it auto-derives `offline_dir = f"{data_root}/{name}"`. The `name` is:

- the **dict key** for eval datasets (`aime24`, `aime25`, `amc`, `minerva`, `math500`)
- the **`dataset_fn` module basename** for the rollout dataset (`deepscaler` from `astraflow.dataflow.dataset.deepscaler:get_deepscaler_rl_dataset`)

The downloader uses the same naming convention, so the two sides stay in sync. To opt a single dataset out — e.g. point one eval at a different snapshot — just set `offline_dir:` explicitly on that entry; explicit values always win.

To convert any other recipe to offline mode, add the same `dataflow.data_root` field; no other changes are required.

## Caveats

- **Model and tokenizer weights are *not* covered** by the dataset downloader. `model_path` / `tokenizer_path` still point at `Qwen/Qwen3-8B` and resolve via the HuggingFace cache. For a fully air-gapped run, pre-fetch them with `huggingface-cli download Qwen/Qwen3-8B --local-dir /local/models/Qwen3-8B` and edit the two paths in `experiment.yaml`.
- The downloader needs internet at prep time. Once `data-data/math/` is populated, training itself works with `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1`.
