# Multi-Agent (Math)

Multi-policy collaborative RL for math reasoning — an actor and a verifier are each trained as a separate policy within one AstraFlow loop.

**Multi-agent recipes**: [`examples/math-multi-agent/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent)

Each recipe ships an all-in-one launch script under `scripts/` and its config under `yaml/`.

## Qwen3-8B Actor + Verifier — 8 GPUs

The headline multi-agent recipe. Two Qwen3-8B policies cooperate on each math problem: **model0 (actor/solver)** generates a solution, **model1 (verifier)** approves or rejects it once, and if rejected the actor retries once with the verifier's full output as context. Each policy gets its own SGLang inference instance and its own FSDP trainer, so the job runs four processes — AstraFlow service, RaaS (2 SGLang servers), and two trainers — on an 8-GPU node.

It comes in two dataset variants, plus a `no-ds` (no dynamic sampling) variant of each:

- [`qwen3-8b-actor-verifier-m2po-delta-dapo-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-dapo-data) — trains on the DAPO-filtered dataset; buffer uses `filter_zero_adv` (dynamic sampling, drops zero-advantage prompts)
- [`qwen3-8b-actor-verifier-m2po-delta-deepscaler-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-deepscaler-data) — same recipe trained on DeepScaleR instead
- [`qwen3-8b-actor-verifier-m2po-delta-no-ds-dapo-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-no-ds-dapo-data) — DAPO data with dynamic sampling **off** (buffer keeps all prompts, no `filter_zero_adv`)
- [`qwen3-8b-actor-verifier-m2po-delta-no-ds-deepscaler-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-no-ds-deepscaler-data) — DeepScaleR data with dynamic sampling **off**

### Run

One script launches all four processes — the AstraFlow service, the RaaS inference server, and the two trainers:

```bash
# DAPO data (dynamic sampling on)
bash examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-dapo-data/scripts/run_qwen3-8b-actor-verifier-m2po-delta-dapo-data.sh

# DeepScaleR data (dynamic sampling on)
bash examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-deepscaler-data/scripts/run_qwen3-8b-actor-verifier-m2po-delta-deepscaler-data.sh
```

### Settings

| Setting | Value |
|---|---|
| Models | 2 × Qwen3-8B — model0 (actor/solver), model1 (verifier) |
| GPUs | 8 — RaaS ×4 (SGLang, model0 DP=2 + model1 DP=2), Trainer model0 ×2 (FSDP, DP=2), Trainer model1 ×2 (FSDP, DP=2) |
| Algorithm | M2PO for both policies (`m2_threshold` 0.01) |
| Weight transfer | TCP — delta (`delta_full_sync_interval` 10), one weight-transfer port per trainer |
| Context length | 16384 |
| Max new tokens | 4096 (per policy) |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 256 |
| Learning rate | 5e-6 (Adam, constant schedule) |
| Train steps | 1200 |
| Workflow / reward | `actor_and_verify` / `math_verify` |
| Train dataset | DAPO-filtered (`dapo-data`) or DeepScaleR (`deepscaler-data`) |
| Eval datasets | AIME24, AIME25, AMC, Minerva Math, MATH500 |

## Single-agent baselines — 8 GPUs

For A/B comparison, each actor+verifier recipe has a matching `single-agent` baseline that drops the verifier: a single Qwen3-8B policy trained with the plain `rlvr` workflow, but the same hyperparameters and inference budget. With only one policy the job runs three processes — AstraFlow service, one SGLang server, one FSDP trainer — using RaaS ×4 (SGLang, DP=4) and Trainer ×4 (FSDP, DP=4).

- [`qwen3-8b-single-agent-m2po-delta-dapo-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-single-agent-m2po-delta-dapo-data) — DAPO data, dynamic sampling on
- [`qwen3-8b-single-agent-m2po-delta-deepscaler-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-single-agent-m2po-delta-deepscaler-data) — DeepScaleR data, dynamic sampling on
- [`qwen3-8b-single-agent-m2po-delta-no-ds-dapo-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-single-agent-m2po-delta-no-ds-dapo-data) — DAPO data, dynamic sampling off
- [`qwen3-8b-single-agent-m2po-delta-no-ds-deepscaler-data/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/math-multi-agent/qwen3-8b-single-agent-m2po-delta-no-ds-deepscaler-data) — DeepScaleR data, dynamic sampling off

### Run

```bash
bash examples/math-multi-agent/qwen3-8b-single-agent-m2po-delta-dapo-data/scripts/run_qwen3-8b-single-agent-m2po-delta-dapo-data.sh
```

### Settings

| Setting | Value |
|---|---|
| Model | Qwen3-8B (single policy, no verifier) |
| GPUs | 8 — RaaS ×4 (SGLang, DP=4), Trainer ×4 (FSDP, DP=4) |
| Algorithm | M2PO (`m2_threshold` 0.01) |
| Weight transfer | TCP — delta (`delta_full_sync_interval` 10) |
| Context length | 16384 |
| Max new tokens | 4096 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 256 |
| Learning rate | 5e-6 (Adam, constant schedule) |
| Train steps | 1200 |
| Workflow / reward | `rlvr` / `math_verify` |
| Train dataset | DAPO-filtered (`dapo-data`) or DeepScaleR (`deepscaler-data`) |
| Eval datasets | AIME24, AIME25, AMC, Minerva Math, MATH500 |
