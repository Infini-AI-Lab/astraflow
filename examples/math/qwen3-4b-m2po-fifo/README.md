# Qwen3-4B Math RL — staleness A/B (queue_order=fifo)

The FIFO arm of the rollout-buffer consumption-order A/B. Identical to
[`qwen3-4b-m2po-edf`](../qwen3-4b-m2po-edf/) except
`dataflow.buffer.queue_order: fifo` (the historical completion-order
behavior) and `trial_name`.

See the sibling recipe's README for the motivation, setup (6 RaaS GPUs vs
2 FSDP trainer GPUs to build staleness pressure), run commands, and the
A/B results. Summary: FIFO's staleness drops are length-biased (it
preferentially expires long/hard generations — dropped samples ~1.3–1.7×
longer than consumed ones), which costs ~5 overall avg@k points at step 100
versus edf and delays reaching the eval plateau by ~150 steps.

`queue_order: edf` is the default; this recipe exists to reproduce the
comparison.

## Run

```bash
bash examples/math/qwen3-4b-m2po-fifo/scripts/run_qwen3-4b-m2po-fifo.sh
```
