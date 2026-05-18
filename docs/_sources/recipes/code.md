# Code

Train code-generation models on DeepCoder Preview using M2PO with AstraFlow.

## Overview

- **Task**: Single-turn code generation
- **Models**: Qwen3-0.6B, Qwen3-4B, Qwen3-8B
- **Algorithm**: M2PO (m2_threshold=0.01)
- **Workflow**: `livecodebench_single_turn`
- **Training data**: `agentica-org/DeepCoder-Preview-Dataset` (`primeintellect` subset)
- **Eval**: LiveCodeBench v5/v6, HumanEval, DeepCoder Codeforces

## Eval Dataset Setup

### 1. LiveCodeBench v5 from AReaL-boba-2-RL-Code

Download the AReaL-boba-2-RL-Code dataset from the repo root:

```bash
huggingface-cli download inclusionAI/AReaL-boba-2-RL-Code \
  --repo-type dataset \
  --local-dir ./data-data/AReaL-boba-2-RL-Code
```

This provides:

```text
./data-data/AReaL-boba-2-RL-Code/code_benchmark/lcb_v5/test.jsonl
```

### 2. LiveCodeBench v6 from code_generation_lite

Download the official LiveCodeBench v6 slice from the repo root:

```bash
mkdir -p ./data-data/lcb_v6_raw ./data-data/lcb_v6

huggingface-cli download livecodebench/code_generation_lite test6.jsonl \
  --repo-type dataset \
  --local-dir ./data-data/lcb_v6_raw
```

The official LiveCodeBench file uses the upstream schema (`question_content`,
`public_test_cases`, encoded `private_test_cases`, and metadata). Convert it to
the v5-compatible `question`/`input_output` schema used by
`livecodebench_single_turn` and `livecodebench_reward`:

```bash
python astraflow/dataflow/dataset/scripts/convert_livecodebench_lite.py \
  ./data-data/lcb_v6_raw/test6.jsonl \
  ./data-data/lcb_v6/test.jsonl
```

This produces:

```text
./data-data/lcb_v6/test.jsonl
```

### 3. HumanEval from GitHub

Clone the HumanEval repo into `astraEnv/`:

```bash
git clone https://github.com/openai/human-eval.git astraEnv/human-eval
```

Unzip the benchmark file:

```bash
gzip -dk astraEnv/human-eval/data/HumanEval.jsonl.gz
```

This produces:

```text
./astraEnv/human-eval/data/HumanEval.jsonl
```

### 4. DeepCoder Codeforces from Hugging Face

No manual download is required for this eval dataset. It is loaded directly from Hugging Face during eval with:

```yaml
dataset_fn: "astraflow.dataflow.dataset.deepcoder_preview:get_deepcoder_preview_codeforces_test_dataset"
dataset_name: "agentica-org/DeepCoder-Preview-Dataset"
subset: "codeforces"
split: "test"
```

## Example: Qwen3-8B

**Config**: `examples/code/qwen3-8b-m2po-delta/`

Key settings:
- M2PO with delta weight transfer
- RaaS: SGLang
- Trainer: FSDP
- Training data: DeepCoder Preview `primeintellect`
- Eval data: HumanEval, LiveCodeBench v5/v6, DeepCoder Codeforces

Refer to `examples/code/qwen3-8b-m2po-delta/yaml/experiment.yaml` for the exact knobs (context length, batch size, learning rate, sync interval).

**Eval dataset block**:

```yaml
astraflow:
  eval_datasets:
    humaneval:
      dataset_fn: "astraflow.dataflow.dataset.human_eval:get_human_eval_test_dataset"
      path: "./astraEnv/human-eval/data/HumanEval.jsonl"
      split: "test"
      max_length: 6000
      repeat: 1

    lcb_v5:
      dataset_fn: "astraflow.dataflow.dataset.livecodebench:get_livecodebench_single_turn_test_dataset"
      path: "./data-data/AReaL-boba-2-RL-Code/code_benchmark/lcb_v5/test.jsonl"
      split: "test"
      max_length: 6000
      repeat: 1

    lcb_v6:
      dataset_fn: "astraflow.dataflow.dataset.livecodebench:get_livecodebench_single_turn_test_dataset"
      path: "./data-data/lcb_v6/test.jsonl"
      split: "test"
      max_length: 6000
      repeat: 1

    deepcoder_codeforces:
      dataset_fn: "astraflow.dataflow.dataset.deepcoder_preview:get_deepcoder_preview_codeforces_test_dataset"
      dataset_name: "agentica-org/DeepCoder-Preview-Dataset"
      subset: "codeforces"
      split: "test"
      max_length: 6000
      repeat: 1
```

**Eval workflow block**:

`lcb_v5`, `lcb_v6`, and `deepcoder_codeforces` use the standard execution reward, while `HumanEval` uses the vendored HumanEval harness:

```yaml
astraflow:
  eval_workflow_specs:
    default:
      workflow_cls: "livecodebench_single_turn"
      reward_fn: "livecodebench_reward"
      gconfig_overrides:
        temperature: 0.6
        n_samples: 1

    humaneval_0:
      workflow_cls: "livecodebench_single_turn"
      reward_fn: "human_eval_reward"
      gconfig_overrides:
        temperature: 0.6
        n_samples: 1
```

**Launch**: one-shot script that starts 3 processes internally:

```bash
# Run from the repo root.
bash examples/code/qwen3-8b-m2po-delta/scripts/run_qwen3-8b-m2po-delta.sh
```

Default GPU layout:
- `SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3` for RaaS
- `TRAINER_MODEL0_GPUS=4,5,6,7` for Trainer

The launcher starts:
- RaaS inference server
- AstraFlow HTTP service
- Trainer using `examples/launch_trainer.py`

## Config Knobs

| Parameter | Description | Typical Values |
|---|---|---|
| `m2_threshold` | M2PO clipping threshold | `0.01` |
| `replay_ratio` | Fraction of replay data in each batch | `0` |
| `max_staleness` | Max weight version gap for accepted rollouts | `8` |
| `n_samples` | Rollouts per prompt | `8` |
| `max_new_tokens` | Maximum generated code length | `4000`, `6000` |
| `weight_transfer_strategies` | Weight transfer mode | `delta` |
