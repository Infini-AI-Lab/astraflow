# Math

Reinforcement learning for single-agent math reasoning with RLVR.

**Math recipes**: [`examples/math/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math)

Each recipe ships an all-in-one launch script under `scripts/` and its config under `yaml/`.

## Qwen3-1.7B — 2 GPUs (minimal reproduction)

The smallest recipe. It runs on a single 2-GPU node — 1 GPU for inference, 1 for training — so it is the quickest way to verify an AstraFlow setup end to end. It comes in two variants that differ **only** in weight transfer mode:

- [`qwen3-1.7b-m2po-2gpus-full/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math/qwen3-1.7b-m2po-2gpus-full) — full weight transfer
- [`qwen3-1.7b-m2po-2gpus-delta/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math/qwen3-1.7b-m2po-2gpus-delta) — delta weight transfer (only changed weights are sent)

### Run

One script launches all three processes — the AstraFlow service, the RaaS inference server, and the trainer:

```bash
# delta weight transfer
bash examples/math/qwen3-1.7b-m2po-2gpus-delta/scripts/run_qwen3-1.7b-m2po-2gpus-delta.sh

# full weight transfer
bash examples/math/qwen3-1.7b-m2po-2gpus-full/scripts/run_qwen3-1.7b-m2po-2gpus-full.sh
```

### Settings

| Setting | Value |
|---|---|
| Model | Qwen3-1.7B |
| GPUs | 2 — RaaS ×1 (SGLang, DP=1), Trainer ×1 (FSDP, DP=1) |
| Algorithm | M2PO (`m2_threshold` 0.01) |
| Weight transfer | TCP — full, or delta (`delta_full_sync_interval` 10) |
| Context length | 7168 |
| Max new tokens | 4000 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 256 |
| Learning rate | 5e-6 (Adam, constant schedule) |
| Train steps | 800 |
| Workflow / reward | `rlvr` / `math_verify` |
| Train dataset | DeepScaleR |
| Eval datasets | AIME24, AIME25, AMC, Minerva Math, MATH500 |

## Qwen3-8B — 8 GPUs

The full-scale recipe. It needs an 8-GPU node — 4 GPUs for inference, 4 for training — and also comes in full and delta transfer variants:

- [`qwen3-8b-m2po-full/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math/qwen3-8b-m2po-full) — full weight transfer
- [`qwen3-8b-m2po-delta/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math/qwen3-8b-m2po-delta) — delta weight transfer

### Run

The same single-script pattern launches the whole job:

```bash
bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full.sh
```

### Settings

| Setting | Value |
|---|---|
| Model | Qwen3-8B |
| GPUs | 8 — RaaS ×4 (SGLang, DP=4), Trainer ×4 (FSDP, DP=4) |
| Algorithm | M2PO (`m2_threshold` 0.01) |
| Weight transfer | TCP — full or delta |
| Context length | 16384 |
| Max new tokens | 14000 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 256 |
| Learning rate | 5e-6 (Adam, constant schedule) |
| Train steps | 800 |
| Workflow / reward | `rlvr` / `math_verify` |
| Train dataset | DeepScaleR |
| Eval datasets | AIME24, AIME25, AMC, Minerva Math, MATH500 |
