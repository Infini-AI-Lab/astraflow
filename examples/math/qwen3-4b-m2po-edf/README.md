# Qwen3-4B Math RL — staleness A/B (queue_order=edf)

One arm of a controlled A/B on the rollout buffer's **consumption order**. The
sibling recipe [`qwen3-4b-m2po-fifo`](../qwen3-4b-m2po-fifo/) is identical
except `dataflow.buffer.queue_order` (and `trial_name`).

## Why this experiment exists

Long generations span several weight versions, so their `min_version` is old
the moment they finish. Consuming in completion order (FIFO — the historical
behavior) parks them behind fresher short samples until they exceed
`max_staleness` and are dropped — a systematic **difficulty bias**: the
hard/long prompts a model most needs to learn are the ones discarded.

`queue_order: edf` (earliest deadline first, now the default) consumes by
ascending `min_version` instead, training on the most staleness-critical
samples before they expire. The per-token staleness bound is unchanged.

## Setup

The GPU split deliberately over-provisions rollout so the buffer builds real
staleness pressure:

```text
SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 -> RaaS (SGLang, DP=6)
TRAINER_MODEL0_GPUS=6,7                  -> Trainer (FSDP, DP=2)
```

Model Qwen/Qwen3-4B, M2PO, ctx 16k / max_new_tokens 14000, batch 256,
`max_staleness: 8` — everything except `queue_order` matches the sibling.

## Run

```bash
bash examples/math/qwen3-4b-m2po-edf/scripts/run_qwen3-4b-m2po-edf.sh
# sibling arm:
bash examples/math/qwen3-4b-m2po-fifo/scripts/run_qwen3-4b-m2po-fifo.sh
```

## Results (8×H100, overall avg@k across AIME24/25, AMC, Minerva, MATH500)

| step | edf | fifo | Δ |
|---|---|---|---|
| 0 | 42.2 | 43.1 | — |
| 100 | **57.6** | 52.5 | **+5.1** |
| 200 | **59.2** | 56.9 | +2.3 |
| 250 | **59.5** | 58.1 | +1.4 |
| 350 | 58.6 | 59.6 | −1.0 (converged) |

- The @100 gap reproduced across two independent edf runs (57.8 and 57.6).
- Gains are largest on the hardest sets (@100: AIME25 +9.4, AIME24 +5.4).
- Mechanism metrics: fifo's staleness drops are length-biased (dropped
  samples 1.3–1.7× longer than consumed); edf's are fair (~1.1×) with a
  lower overall drop rate.
- edf is a **sample-efficiency** win (reaches fifo's plateau ~150 steps
  earlier); with enough steps both arms converge.
