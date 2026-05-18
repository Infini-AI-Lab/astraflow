# Multi-Agent

Train multiple cooperating models, each with a distinct role, within a single AstraFlow loop.

## Actor-Verifier (8B 2-agent math)

Two models collaborate on math problems: an **actor** generates solutions and a **verifier** provides accept/reject feedback.

- **Models**: 2x Qwen3-8B (separate instances)
  - **model0 (actor)**: generates solution
  - **model1 (verifier)**: accepts/rejects with feedback
- **Algorithm**: M2PO for both
- **Workflow**: `actor_and_verify`

**Config**: `examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta/`

Key settings:
- RaaS: 2 SGLang servers (allocation_mode: `sglang[model0]:d2+sglang[model1]:d2`)
- Trainers: 2 separate FSDP trainers, 2 GPUs each

## Launch

4 processes, each on dedicated GPUs:

```bash
# 1. RaaS (2 SGLang servers)
SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. AstraFlow (CPU-only)

# 3. Trainer model0 (actor)
TRAINER_MODEL0_GPUS=4,5
WEIGHT_TRANSFER_HTTP_PORT_MODEL0=19861

# 4. Trainer model1 (verifier)
TRAINER_MODEL1_GPUS=6,7
WEIGHT_TRANSFER_HTTP_PORT_MODEL1=19862
```

See `examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta/scripts/` for the full launch scripts.

## Key Differences from Single-Agent

| | Single-Agent | Multi-Agent |
|---|---|---|
| Models | 1 | 2+ (each with own role) |
| RaaS allocation | `sglang:d4` | `sglang[model0]:d2+sglang[model1]:d2` |
| Trainers | 1 | 1 per model (separate weight transfer ports) |
| Workflow | `rlvr` | `solve_and_verify` |
| GPU requirement | 8 (4 inference + 4 training) | 8 (4 inference + 2+2 training) |
