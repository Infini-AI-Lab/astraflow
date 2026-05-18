# Math

Train reasoning models on math datasets using GRPO/M2PO with AstraFlow.

## Overview

- **Task**: Offline math reasoning (GSM8K-style problems)
- **Models**: Qwen3-1.7B, Qwen3-4B, Qwen3-8B
- **Algorithm**: GRPO / M2PO (m2_threshold=0.004–0.01)
- **Workflow**: `rlvr` with `math_verify` reward
- **Eval**: AIME24/25, AMC, Minerva Math, Math500

## Example: Qwen3-1.7B

**Config**: `examples/math/qwen3-1.7b-m2po-full/`

Key settings:
- M2PO, batch size and learning rate as set in the recipe yaml
- RaaS: SGLang on 4 GPUs, context_length: 16384
- Trainer: FSDP on 4 GPUs, TCP weight transfer (full)

**Launch** (3 processes in separate terminals):

```bash
# 1. RaaS (inference)
SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. AstraFlow (CPU-only, port 8000)

# 3. Trainer (training GPUs + sender agent)
ASTRAFLOW_CUDA_VISIBLE_DEVICES=4,5,6,7
```

Recipes: `examples/math/`

See `examples/math/qwen3-1.7b-m2po-full/scripts/` for the full launch scripts.

## Example: Qwen3-8B

**Config**: `examples/math/qwen3-8b-m2po-full/`

Key differences from the 1.7B recipe:
- M2PO with m2_threshold=0.01
- context_length: 16384, max_new_tokens: 4096

## Config Knobs

| Parameter | Description | Typical Values |
|---|---|---|
| `m2_threshold` | M2PO clipping threshold | 0.004–0.01 |
| `replay_ratio` | Fraction of replay data in each batch | 0–0.7 |
| `max_staleness` | Max weight version gap for accepted rollouts | 2–8 |
| `n_samples` | Rollouts per prompt | 8 |
| `eps_clip` | PPO clip range (set high to disable) | 100 |
