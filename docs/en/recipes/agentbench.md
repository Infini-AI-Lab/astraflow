# AgentBench

Reinforcement learning for multi-turn, interactive agents on the ALFWorld and WebShop environments from [AgentBench](https://github.com/THUDM/AgentBench), with M2PO.

**AgentBench recipes**:

- ALFWorld: [`examples/alfworld/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/alfworld)
- WebShop: [`examples/webshop/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/webshop)

Each recipe ships an all-in-one launch script under `scripts/` and its config under `yaml/`.

## Environment setup

The AgentBench recipes need the AgentBench environment installed in its own conda env, plus the task container image:

```bash
cd astraEnv/AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
conda install podman

# pull the image for the environment you want to train on
docker pull longinyu/agentbench-alfworld   # for ALFWorld
docker pull longinyu/agentbench-webshop    # for WebShop
```

The all-in-one `run_*.sh` starts the task server itself; the standalone `0_<env>_server.sh` is provided for split launches.

## ALFWorld — Qwen2.5-7B-Instruct — 8 GPUs

Trains an agent to complete embodied household tasks in a text-based environment. Each rollout is a multi-turn episode (up to 15 turns) against an AgentBench task server. The recipe comes in two variants that differ **only** in weight transfer mode:

- [`qwen2.5-7b-instruct-m2po-full/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/alfworld/qwen2.5-7b-instruct-m2po-full) — full weight transfer
- [`qwen2.5-7b-instruct-m2po-delta/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/alfworld/qwen2.5-7b-instruct-m2po-delta) — delta weight transfer (full sync every 10 steps)

### Run

One script launches four processes — the AgentBench ALFWorld environment server, the AstraFlow service, the RaaS inference server, and the trainer. The all-in-one script starts the local environment server itself; the standalone `0_alfworld_server.sh` is provided for split launches.

```bash
# full weight transfer
bash examples/alfworld/qwen2.5-7b-instruct-m2po-full/scripts/run_qwen2.5-7b-instruct-m2po-full.sh

# delta weight transfer
bash examples/alfworld/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh
```

### Settings

| Setting | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| GPUs | 8 — RaaS ×4 (SGLang, DP=4), Trainer ×4 (FSDP, DP=4) |
| Algorithm | M2PO (`m2_threshold` 0.004) |
| Weight transfer | TCP — full, or delta (`delta_full_sync_interval` 10) |
| Context length | 16384 |
| Max new tokens | 512 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 128 |
| Learning rate | 1e-6 (Adam, constant schedule) |
| Train steps | 1200 |
| Workflow / reward | `alfworld_task_server` (max 15 turns) |
| Train dataset | ALFWorld train indices |
| Eval datasets | ALFWorld valid indices (eval max 10 turns, every 10 steps) |
| Environment | ALFWorld AgentBench task server (`http://127.0.0.1:5000`) |

## WebShop — Qwen2.5-7B-Instruct — 8 GPUs

Trains an agent to navigate and purchase products on a simulated e-commerce site. Each rollout is a multi-turn episode (up to 10 turns) against an AgentBench task server. It also comes in full and delta transfer variants:

- [`qwen2.5-7b-instruct-m2po-full/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/webshop/qwen2.5-7b-instruct-m2po-full) — full weight transfer
- [`qwen2.5-7b-instruct-m2po-delta/`](https://github.com/haizhongzheng/astraflow/tree/main/examples/webshop/qwen2.5-7b-instruct-m2po-delta) — delta weight transfer (full sync every 10 steps)

### Run

The same single-script pattern launches four processes — the AgentBench WebShop environment server, the AstraFlow service, the RaaS inference server, and the trainer. The all-in-one script starts the local environment server itself; the standalone `0_webshop_server.sh` is provided for split launches.

```bash
# full weight transfer
bash examples/webshop/qwen2.5-7b-instruct-m2po-full/scripts/run_qwen2.5-7b-instruct-m2po-full.sh

# delta weight transfer
bash examples/webshop/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh
```

### Settings

| Setting | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| GPUs | 8 — RaaS ×4 (SGLang, DP=4), Trainer ×4 (FSDP, DP=4) |
| Algorithm | M2PO (`m2_threshold` 0.004) |
| Weight transfer | TCP — full, or delta (`delta_full_sync_interval` 10) |
| Context length | 16384 |
| Max new tokens | 512 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 256 |
| Learning rate | 1e-6 (Adam, constant schedule) |
| Train steps | 1200 |
| Workflow / reward | `webshop_task_server` (max 10 turns) |
| Train dataset | WebShop train indices |
| Eval datasets | WebShop valid indices (eval max 10 turns, every 10 steps) |
| Environment | WebShop AgentBench task server (`http://127.0.0.1:5000`) |
