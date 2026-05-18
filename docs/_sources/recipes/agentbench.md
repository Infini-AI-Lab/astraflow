# AgentBench

This to set up env for Agentbench Task Server, and train interactive agents on AgentBench environments (WebShop, ALFWorld) using M2PO.

## Prerequisites

Both ALFWorld and WebShop require the AgentBench environment:

```bash
# 0. Prepare the AstraFlow env first.

# 1. Set up env for AgentBench
cd astraEnv/AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
conda install podman

# 2. Then pull docker images:
docker pull longinyu/agentbench-alfworld
docker pull longinyu/agentbench-webshop
```

## WebShop

Train a model to navigate and purchase products on a simulated e-commerce site.

- **Model**: Qwen2.5-7B-Instruct
- **Algorithm**: M2PO (m2_threshold=0.004), LR: 1e-6
- **Workflow**: `webshop_task_server`, max_turns: 10
- **Eval frequency**: every 10 steps

**Config**: `examples/webshop/qwen2.5-7b-instruct-m2po-full/`

Key settings:
- Buffer: 262144, replay_ratio: 0, max_staleness: 8
- Batch size: 256, max_new_tokens: 512, max_length: 4096
- RaaS: SGLang on 4 GPUs
- Trainer: FSDP on 4 GPUs, TCP weight transfer (full)

### Split launch (4 processes in separate terminals)

```bash
# 0. astraEnv/AgentBench WebShop server
bash examples/webshop/qwen2.5-7b-instruct-m2po-full/scripts/0_webshop_server.sh

# 1. RaaS
bash examples/webshop/qwen2.5-7b-instruct-m2po-full/scripts/2_raas.sh

# 2. AstraFlow
bash examples/webshop/qwen2.5-7b-instruct-m2po-full/scripts/1_astraflow.sh

# 3. Trainer
bash examples/webshop/qwen2.5-7b-instruct-m2po-full/scripts/3_trainer_model0.sh
```

### One-shot launch

```bash
bash examples/webshop/qwen2.5-7b-instruct-m2po-full/scripts/run_qwen2.5-7b-instruct-m2po-full.sh
```

### Stop WebShop Server
```bash
bash astraEnv/AgentBench/stop-webshop-server.sh
```

## ALFWorld

Train a model to complete embodied household tasks in a text-based environment.

- **Model**: Qwen2.5-7B-Instruct
- **Algorithm**: M2PO (m2_threshold=0.004), LR: 1e-6
- **Workflow**: `alfworld_task_server`, max_turns: 15
- **Eval frequency**: every 10 steps

**Config**: `examples/alfworld/qwen2.5-7b-instruct-m2po-full/`

Key settings:
- Nearly identical to WebShop, except:
  - max_turns: 15 (vs. 10 for WebShop)
  - mem_fraction_static: 0.7 (vs. 0.6 for WebShop)

### Split launch (4 processes in separate terminals)

```bash
# 0. astraEnv/AgentBench ALFWorld server
bash examples/alfworld/qwen2.5-7b-instruct-m2po-full/scripts/0_alfworld_server.sh

# 1. RaaS
bash examples/alfworld/qwen2.5-7b-instruct-m2po-full/scripts/2_raas.sh

# 2. AstraFlow
bash examples/alfworld/qwen2.5-7b-instruct-m2po-full/scripts/1_astraflow.sh

# 3. Trainer
bash examples/alfworld/qwen2.5-7b-instruct-m2po-full/scripts/3_trainer.sh
```

### One-shot launch

```bash
bash examples/alfworld/qwen2.5-7b-instruct-m2po-full/scripts/run_qwen2.5-7b-instruct-m2po-full.sh
```

### Stop ALFWorld Server
```bash
bash astraEnv/AgentBench/stop-alfworld-server.sh
```

## Key Differences from Math

| | Math | AgentBench |
|---|---|---|
| Interaction | Single-turn generation | Multi-turn with environment |
| Task server | None (offline dataset) | AgentBench server required |
| Processes | 3 (RaaS, AstraFlow, Trainer) | 4 (+ task server) |
| Buffer size | 10k | 262144 |
| Max new tokens | 4096–12000 | 512 |
