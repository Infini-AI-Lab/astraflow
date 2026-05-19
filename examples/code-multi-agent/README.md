# Code Multi-Agent Recipes

Reinforcement-learning recipes for code generation on AstraFlow with M2PO. Two
recipes ship here:

| Recipe | Agents | Hardware | What it trains |
|---|---|---|---|
| [`qwen3-8b-single-agent-m2po-full`](qwen3-8b-single-agent-m2po-full) | 1 | 1 node × 8 GPUs | Single-model code generator (baseline) |
| [`qwen3-8b-codegen-verifier-m2po-full-2node`](qwen3-8b-codegen-verifier-m2po-full-2node) | 2 | 2 nodes × 8 GPUs | Code generator + test-case verifier |

Both train Qwen3-8B. Background: [`docs/en/recipes/code.md`](../../docs/en/recipes/code.md)
covers the task, datasets, and eval; [`docs/en/recipes/multi-agent.md`](../../docs/en/recipes/multi-agent.md)
covers the multi-agent design.

## Prerequisites

- One or two 8-GPU nodes (8×H100 or similar), depending on the recipe.
- Training data (`agentica-org/DeepCoder-Preview-Dataset`) is fetched from
  Hugging Face automatically on first run.
- **LiveCodeBench v5 eval data needs a one-time manual download.** Follow the
  "Eval Dataset Setup" section of
  [`docs/en/recipes/code.md`](../../docs/en/recipes/code.md) before launching —
  otherwise the periodic eval steps will fail.

## Single-agent recipe — 1 node

RaaS, the AstraFlow service, and the trainer all run on one node over
`localhost`:

```bash
bash examples/code-multi-agent/qwen3-8b-single-agent-m2po-full/scripts/run_qwen3-8b-single-agent-m2po-full.sh
```

Default GPU split: `0–3` inference, `4–7` training. Override with
`SERVICE_CUDA_VISIBLE_DEVICES` / `TRAINER_GPUS`. Nothing is node-specific — this
recipe runs as-is on any single 8-GPU node.

## Codegen/verifier recipe — 2 nodes

This recipe spans **two nodes**. Throughout its scripts and configs the nodes
are named with **placeholders** — substitute your own hostnames:

| Placeholder | Role | GPU layout |
|---|---|---|
| `compute-node-0` | AstraFlow service + RaaS-B + both trainers — *the node you start on* | RaaS-B `0–3`, trainer-model0 `4,5`, trainer-model1 `6,7` |
| `compute-node-1` | RaaS-A — an extra inference replica | RaaS-A `0–7` |

### How to launch

1. **On `compute-node-0`**, run the all-in-one launcher. It starts the
   AstraFlow service, RaaS-B, and both trainers, then prints the exact command
   to run on the other node:

   ```bash
   bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/run_qwen3-8b-codegen-verifier-m2po-full-2node.sh
   ```

2. **On `compute-node-1`**, paste the command it printed. It looks like:

   ```bash
   ASTRAFLOW_HOST_EXTERNAL=<compute-node-0 host> \
     bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/1_raas_a.sh
   ```

`ASTRAFLOW_HOST_EXTERNAL` is **required**: it is the hostname/IP of
`compute-node-0` and must be reachable from `compute-node-1`. The launcher on
`compute-node-0` fills in that node's hostname automatically — if the printed
hostname is not routable from `compute-node-1`, use `compute-node-0`'s IP
instead. Run `1_raas_a.sh` without `ASTRAFLOW_HOST_EXTERNAL` and it exits
immediately with an explanatory error rather than failing later.

### Required open ports (`compute-node-1` → `compute-node-0`)

| Port | Purpose |
|---|---|
| `8000` | AstraFlow HTTP service |
| `19861` | weight sender — trainer model0 |
| `19862` | weight sender — trainer model1 |
| `21000` | weight-transfer handshake |

### Step-by-step alternative

Instead of the all-in-one `run_*.sh`, you can launch each component yourself
with the numbered scripts — `1_raas_a.sh`, `1_raas_b.sh`, `2_astraflow.sh`,
`3_trainer_model0.sh`, `4_trainer_model1.sh`. GPU and port assignments are
overridable via env vars (`RAAS_B_GPUS`, `TRAINER_MODEL0_GPUS`,
`ASTRAFLOW_PORT`, …); see the header comment of each script.
