# Running Local Models with FastChat in AgentBench

This guide explains how to run evaluations using locally-deployed models via FastChat.

## Overview

```
+-------------------+       +-------------------+       +-------------------+
|   AgentBench      | ----> |  FastChat         | ----> |  Local Model      |
|   (Assigner)      |       |  (Controller)     |       |  (Worker)         |
+-------------------+       +-------------------+       +-------------------+
        |                           |                           |
   Sends requests            Routes requests             Runs inference
   with model_name           to workers                  on GPU
```

## Prerequisites

```bash
# Install required packages
pip install --upgrade transformers fschat accelerate

# Activate environment
conda activate agent-bench
```

## Step-by-Step Guide

### Step 1: Configure the Agent

Edit `configs/agents/fs_agent.yaml` to add your model:

```yaml
YourModelName:
  parameters:
    model_name: "YourModelName"  # Must match FastChat's registered name
```

**Important:** The `model_name` must match exactly what FastChat registers (usually the last part of the model path without the org prefix).

Example for Qwen3:
```yaml
Qwen3-4B-Instruct-2507:
  parameters:
    model_name: "Qwen3-4B-Instruct-2507"  # Not "Qwen/Qwen3-4B-Instruct-2507"
```

### Step 2: Configure the Assignment

Edit or create an assignment file (e.g., `configs/assignments/your-task.yaml`):

```yaml
import: definition.yaml

concurrency:
  task:
    alfworld-std: 5
  agent:
    YourModelName: 5

assignments:
  - agent:
      - YourModelName
    task:
      - alfworld-std

output: "outputs/{TIMESTAMP}"
```

### Step 3: Start FastChat Controller (Terminal 1)

```bash
python -m fastchat.serve.controller --port 55555
```

Wait until you see:
```
INFO:     Uvicorn running on http://localhost:55555 (Press CTRL+C to quit)
```

**Keep this terminal running.**

### Step 4: Start Model Worker (Terminal 2)

```bash
python -m fastchat.serve.model_worker \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --controller http://localhost:55555 \
    --port 21002 \
    --worker-address http://localhost:21002
```

Optional flags:
- `--conv-template <template>`: Specify conversation template (e.g., `qwen2-7b-instruct`)
- `--num-gpus <n>`: Use multiple GPUs
- `--load-8bit`: Load model in 8-bit quantization

Wait until you see the model loaded and registered:
```
INFO | model_worker | Register to controller
```

**Keep this terminal running.**

### Step 5: Start Task Server (Terminal 3)

```bash
cd /path/to/AgentBench
python -m src.start_task -a
```

Wait about 1 minute until you see "200 OK" messages.

**Keep this terminal running.**

### Step 6: Run the Evaluation (Terminal 4)

```bash
python -m src.assigner --config configs/assignments/your-task.yaml
```

## Troubleshooting

### Error: "no worker: ModelName"

**Cause:** Model name mismatch between config and FastChat registration.

**Solution:** Check what name FastChat registered by looking at the worker startup log:
```
Loading the model ['ActualModelName'] on worker ...
```
Update your `fs_agent.yaml` to use `ActualModelName`.

### Error: "Connection refused" when starting worker

**Cause:** Controller is not running on the specified port.

**Solution:** Make sure the controller (Terminal 1) is running before starting the worker.

### Error: "KeyError: 'model_type'"

**Cause:** transformers version doesn't support the model.

**Solution:**
```bash
pip install --upgrade transformers>=4.43.0
```

### Error: "NETWORK ERROR DUE TO HIGH TRAFFIC"

**Cause:** Model name mismatch - AgentBench is requesting a model that doesn't exist.

**Solution:** Verify model names match (see "no worker" error above).

## Configuration Reference

### fs_agent.yaml Structure

```yaml
default:
  module: "src.client.agents.FastChatAgent"
  parameters:
    name: "FastChat"
    controller_address: "http://localhost:55555"
    max_new_tokens: 8192
    temperature: 0

YourModelName:
  parameters:
    model_name: "YourModelName"
    # Optional: custom prompter for chat format
    prompter:
      name: prompt_string
      args:
        prefix: ""
        user_format: "<|im_start|>user\n{content}<|im_end|>\n"
        agent_format: "<|im_start|>assistant\n{content}<|im_end|>\n"
        suffix: "<|im_start|>assistant\n"
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `controller_address` | FastChat controller URL |
| `model_name` | Name to request from FastChat |
| `max_new_tokens` | Maximum tokens to generate |
| `temperature` | Sampling temperature (0 = deterministic) |
| `prompter` | Custom prompt formatting |

## Terminal Summary

| Terminal | Command | Purpose |
|----------|---------|---------|
| 1 | `python -m fastchat.serve.controller --port 55555` | FastChat controller |
| 2 | `python -m fastchat.serve.model_worker --model-path ... --controller http://localhost:55555 ...` | Model inference |
| 3 | `python -m src.start_task -a` | AgentBench task server |
| 4 | `python -m src.assigner --config ...` | Run evaluation |

## Example: Complete Qwen3 Setup

```bash
# Terminal 1
python -m fastchat.serve.controller --port 55555

# Terminal 2
python -m fastchat.serve.model_worker \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --controller http://localhost:55555 \
    --port 21002 \
    --worker-address http://localhost:21002

# Terminal 3
python -m src.start_task -a

# Terminal 4 (after ~1 min)
python -m src.assigner --config configs/assignments/alfworld-std.yaml
```
