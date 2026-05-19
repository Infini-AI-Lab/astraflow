# Code

Reinforcement learning for code-generation reasoning with RLVR, rewarded by test-case execution.

**Code recipes**: [`examples/code/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code) and [`examples/code-multi-agent/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code-multi-agent)

Each recipe ships an all-in-one launch script under `scripts/` and its config under `yaml/`.

LiveCodeBench v5 eval data needs a one-time manual download — see [Eval Dataset Setup](#eval-dataset-setup) below.

## Eval Dataset Setup

The recipes evaluate on HumanEval, LiveCodeBench v5, and DeepCoder Codeforces. Only
LiveCodeBench v5 needs a one-time manual download before launching — otherwise the
periodic eval steps fail.

Download the AReaL-boba-2-RL-Code dataset from the repo root:

```bash
huggingface-cli download inclusionAI/AReaL-boba-2-RL-Code \
  --repo-type dataset \
  --local-dir ./data-data/AReaL-boba-2-RL-Code
```

This provides `./data-data/AReaL-boba-2-RL-Code/code_benchmark/lcb_v5/test.jsonl`.

HumanEval ships vendored in the repo (`astraEnv/human-eval/data/HumanEval.jsonl`), and
DeepCoder Codeforces is loaded directly from Hugging Face during eval — neither needs
a download.

The training dataset (DeepCoder-Preview, `primeintellect` subset) is fetched from
Hugging Face automatically on first run, so LiveCodeBench v5 above is the only manual
download needed to run a recipe end to end.

## Qwen3-8B — 8 GPUs (single-agent)

Single-agent code-generation RL on one 8-GPU node — 4 GPUs for inference, 4 for training. It comes in two variants that differ **only** in weight transfer mode:

- [`code/qwen3-8b-m2po-full/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code/qwen3-8b-m2po-full) — full weight transfer
- [`code/qwen3-8b-m2po-delta/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code/qwen3-8b-m2po-delta) — delta weight transfer (only changed weights are sent)

A near-identical single-agent baseline also lives at [`code-multi-agent/qwen3-8b-single-agent-m2po-full/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code-multi-agent/qwen3-8b-single-agent-m2po-full) as the fair-comparison reference for the codegen+verifier recipe below.

### Run

One script launches all three processes — the AstraFlow service, the RaaS inference server, and the trainer:

```bash
# delta weight transfer
bash examples/code/qwen3-8b-m2po-delta/scripts/run_qwen3-8b-m2po-delta.sh

# full weight transfer
bash examples/code/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full.sh
```

### Settings

| Setting | Value |
|---|---|
| Model | Qwen3-8B |
| GPUs | 8 — RaaS ×4 (SGLang, DP=4), Trainer ×4 (FSDP, DP=4) |
| Algorithm | M2PO (`m2_threshold` 0.01) |
| Weight transfer | TCP — full, or delta (`delta_full_sync_interval` 10) |
| Context length | 12288 |
| Max new tokens | 4096 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 128 |
| Learning rate | 3e-6 (Adam, constant schedule) |
| Train steps | 1200 |
| Workflow / reward | `livecodebench_single_turn` / `livecodebench_reward` |
| Train dataset | DeepCoder-Preview (`primeintellect` subset) |
| Eval datasets | HumanEval, LiveCodeBench v5, DeepCoder Codeforces |

## Qwen3-8B codegen + verifier — 2 nodes × 8 GPUs (multi-agent)

A two-agent recipe: model0 generates code, model1 generates verification test cases, and the two are co-trained against each other's outputs. It spans two 8-GPU nodes — both nodes host both models for inference, while the two trainers live on the starting node:

- [`code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node) — codegen + verifier, full weight transfer

### Run

Launch the all-in-one script on the first node; it starts the AstraFlow service, RaaS-B, and both trainers, then prints the exact command to run on the second node (RaaS-A):

```bash
bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/run_qwen3-8b-codegen-verifier-m2po-full-2node.sh
```

### Settings

| Setting | Value |
|---|---|
| Models | model0 codegen (Qwen3-8B), model1 testcase verifier (Qwen3-8B) |
| GPUs | 16 — RaaS-A ×8 (SGLang, model0 DP=6 / model1 DP=2), RaaS-B ×4 (SGLang, model0 DP=3 / model1 DP=1), Trainer model0 ×2 (FSDP, DP=2), Trainer model1 ×2 (FSDP, DP=2) |
| Algorithm | M2PO (`m2_threshold` 0.01) |
| Weight transfer | TCP — full |
| Context length | 12288 |
| Max new tokens | 4096 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 256 |
| Learning rate | 3e-6 (Adam, constant schedule) |
| Train steps | 800 |
| Workflow / reward | `code_actor_and_verify_v2` / `livecodebench_reward` (2 generated test cases, `verify_timeout` 6s) |
| Train dataset | DeepCoder-Preview (`primeintellect` subset) |
| Eval datasets | HumanEval, LiveCodeBench v5, DeepCoder Codeforces |
