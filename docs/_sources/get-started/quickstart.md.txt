# Quick Start

Run your first AstraFlow training job. This guide uses the smallest recipe — it
needs just **2 GPUs**, the quickest way to see the whole system working end to end.

## Prerequisites

- AstraFlow installed — see [Installation](installation.md).
- A machine with at least **2 NVIDIA GPUs**.

## Launch a training run

AstraFlow runs as three coordinated processes:

- **AstraFlow** — the data orchestrator (CPU-only HTTP service)
- **RaaS** — the inference server that generates rollouts (1 GPU)
- **Trainer** — the training worker that updates weights (1 GPU)

Every recipe ships an all-in-one script that starts all three for you. The smallest
recipe is Qwen3-1.7B math RL on 2 GPUs. From the repo root:

```bash
bash examples/math/qwen3-1.7b-m2po-2gpus-full/scripts/run_qwen3-1.7b-m2po-2gpus-full.sh
```

The script launches the three processes in order — AstraFlow service, RaaS server,
then the trainer — and training starts once the trainer connects. Per-process logs
are written under `data-log/`.

:::{note}
This recipe logs to **Weights & Biases** (`stats_logger.wandb.mode: online` in its
`experiment.yaml`). Run `wandb login` before launching, or set that field to
`disabled` to skip W&B.
:::

## Next steps

Explore the other recipes for larger models and other tasks:

- [Math](../recipes/math.md) — including the 8-GPU Qwen3-8B recipe
- [Code](../recipes/code.md)
- [Multi-Agent (Math)](../recipes/multi-agent.md)
- [AgentBench](../recipes/agentbench.md)
- [Search](../recipes/search.md)
