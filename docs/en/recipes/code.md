# Code

Reinforcement learning for code-generation reasoning with RLVR, rewarded by test-case execution.

**Code recipes**: [`examples/code/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code), [`examples/code-multi-agent/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/code-multi-agent), and [`examples/terminal-bench/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/terminal-bench)

Each recipe ships an all-in-one launch script under `scripts/` and its config under `yaml/`.

LiveCodeBench v5 eval data needs a one-time manual download — see [Eval Dataset Setup](#eval-dataset-setup) below. Terminal-Bench 2 recipes also need Harbor setup and, for Harbor RL, LiveCodeBench v6 eval data.

## Eval Dataset Setup

The standard code recipes evaluate on HumanEval, LiveCodeBench v5, and DeepCoder Codeforces. Only
LiveCodeBench v5 needs a one-time manual download for those recipes — otherwise the
periodic eval steps fail. Terminal-Bench Harbor RL also needs LiveCodeBench v6, covered below.

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

## LiveCodeBench v6 Setup

The Terminal-Bench Harbor RL recipe evaluates on LiveCodeBench v6. Prepare it from the repo root:

```bash
mkdir -p ./data-data/lcb_v6_raw ./data-data/lcb_v6

huggingface-cli download livecodebench/code_generation_lite test6.jsonl \
  --repo-type dataset \
  --local-dir ./data-data/lcb_v6_raw

python astraflow/dataflow/dataset/scripts/convert_livecodebench_lite.py \
  ./data-data/lcb_v6_raw/test6.jsonl \
  ./data-data/lcb_v6/test.jsonl
```

This produces `./data-data/lcb_v6/test.jsonl`.

## Terminal-Bench 2 + Harbor Setup

AstraFlow has two Harbor-backed Terminal-Bench workflows:

- `terminal_bench_harbor` runs `terminal-bench@2.0` eval tasks through Harbor and reports eval rewards.
- `terminal_bench_harbor_rl` runs local Harbor task directories and converts Harbor rollout details into RL tensors.

The Terminal-Bench recipes live under `examples/terminal-bench/`. They invoke Harbor from a separate conda env so the main AstraFlow training env does not need to import Harbor directly.

### Create the Harbor Env

Harbor uses containerized task sandboxes. Make sure Docker or Podman is usable on the host first:

```bash
docker info
# or, for Podman-backed recipes
podman info
```

Create the conda env expected by the recipes:

```bash
conda create -n harbor-tb2 python=3.12 -y
conda activate harbor-tb2
pip install harbor
```

Sanity-check Harbor against Terminal-Bench 2:

```bash
harbor run -d terminal-bench@2.0 -a oracle --yes
```

The recipe configs call Harbor with:

```yaml
harbor_command:
  - conda
  - run
  - "--no-capture-output"
  - "-n"
  - harbor-tb2
  - harbor
```

### Terminal-Bench 2 Eval Recipe

The eval recipe trains on DeepCoder Preview and periodically evaluates with Harbor + Terminus-2 on Terminal-Bench 2:

```bash
bash examples/terminal-bench/terminal-bench-2-qwen3-8b/scripts/run_terminal-bench-2-qwen3-8b.sh
```

Important config pieces:

```yaml
dataflow:
  eval_workflows:
    terminal_bench_2:
      workflow_cls: "terminal_bench_harbor"
      dataset: "terminal-bench@2.0"
      agent_name: "terminus-2"
      model_name: "openai/Qwen/Qwen3-8B"
      max_parallel_jobs: 4
      agent_kwargs:
        temperature: 0.6
        max_turns: 10
        enable_summarize: true

  eval_datasets:
    terminal_bench_2:
      dataset_fn: "astraflow.dataflow.dataset.terminal_bench:get_terminal_bench_2_test_dataset"
      split: "test"
      repeat: 1
      eval_workflow: terminal_bench_2
```

`max_parallel_jobs` controls how many Harbor subprocesses AstraFlow launches at once. Lower it if Docker or Podman reports resource pressure or stale container conflicts.

The corresponding RaaS config must keep tokenizer initialization enabled:

```yaml
sglang:
  skip_tokenizer_init: false
```

Terminus-2 talks to SGLang through the OpenAI-compatible chat endpoint, and SGLang needs its tokenizer for chat-template application.

### Harbor RL Dataset Prep

The Harbor RL recipe expects local task directories, each containing an `instruction.md` file and the task assets/tests:

```text
./data-data/harbor/CodeContests/
  task-a/
    instruction.md
    environment/
    tests/
  task-b/
    instruction.md
    environment/
    tests/
```

Prepare the default dataset with the helper script:

```bash
python astraflow/dataflow/dataset/scripts/prepare_harbor_dataset.py \
  --dataset open-thoughts/CodeContests \
  --output-dir ./data-data/harbor/CodeContests
```

The helper downloads the Hugging Face dataset snapshot, finds parquet files with `path` and `task_binary` columns, and safely extracts each archived Harbor task into the output directory.

The launcher defaults to this path:

```bash
export HARBOR_TRAIN_DATA="${HARBOR_TRAIN_DATA:-./data-data/harbor/CodeContests}"
```

Override it when launching if your tasks live elsewhere:

```bash
HARBOR_TRAIN_DATA=/path/to/harbor/tasks \
  bash examples/terminal-bench/terminal-bench-rl-qwen3-14b-podman-test/scripts/run_terminal-bench-rl-qwen3-14b-podman-test.sh
```

### Harbor RL Recipe

The Harbor RL recipe trains Qwen3-14B on local Harbor tasks, uses the Podman custom environment, and evaluates on LiveCodeBench v6:

```bash
bash examples/terminal-bench/terminal-bench-rl-qwen3-14b-podman-test/scripts/run_terminal-bench-rl-qwen3-14b-podman-test.sh
```

Key workflow block:

```yaml
dataflow:
  rollout_dataset:
    dataset_fn: "astraflow.dataflow.dataset.terminal_bench:get_harbor_task_path_dataset"
    path: "$HARBOR_TRAIN_DATA"
    split: "train"
    dataset_name: "skyrl_codecontests"

  workflow_spec:
    workflow_cls: "terminal_bench_harbor_rl"
    extra_args:
      - "--environment-import-path"
      - "examples.terminal-bench.harbor_podman_env:PodmanEnvironment"
    agent_name: "terminus-2"
    model_name: "openai/Qwen/Qwen3-14B"
    max_parallel_jobs: 16
    agent_kwargs:
      temperature: 1.0
      max_turns: 1
      suppress_max_turns_warning: true
      enable_summarize: false
      collect_rollout_details: true
```

`collect_rollout_details: true` is required for RL. Harbor/Terminus-2 asks the OpenAI-compatible backend for token IDs and logprobs, and AstraFlow converts those rollout details into trainable trajectories. `enable_summarize: false` keeps conversation history linear for conservative token/logprob alignment.

The RL recipe evaluates LiveCodeBench v6 through the standard code workflow:

```yaml
dataflow:
  eval_datasets:
    lcb_v6:
      dataset_fn: "astraflow.dataflow.dataset.livecodebench:get_livecodebench_single_turn_test_dataset"
      path: "./data-data/lcb_v6/test.jsonl"
      split: "test"
      max_length: 6000
      repeat: 1
      eval_workflow: code_eval
```

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
