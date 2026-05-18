# AgentBench

This directory contains the local AgentBench environment used by AstraFlow to
train interactive agents on the AgentBench tasks used in this repo:
WebShop and ALFWorld.

The environment code here is referred from the original AgentBench project and
adapted for AstraFlow task-server training.

Original project:
- https://github.com/THUDM/AgentBench

## Env Setup

Both ALFWorld and WebShop require the AgentBench environment and task server.

```bash
# 0. Prepare the AstraFlow env first.

# 1. Set up AgentBench
cd astraEnv/AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
conda install podman

# 2. Pull required images
docker pull longinyu/agentbench-alfworld
docker pull longinyu/agentbench-webshop
```

## Launch Env Server
```bash
# Launch ALFWorld server
bash astraEnv/AgentBench/start-alfworld-server.sh

# Launch WebShop server
bash astraEnv/AgentBench/start-webshop-server.sh
```

## Stop Env Server
```bash
# Stop ALFWorld server
bash astraEnv/AgentBench/stop-alfworld-server.sh

# Stop WebShop server
bash astraEnv/AgentBench/stop-webshop-server.sh
```

## Training

Please refer to `docs/en/recipes/agentbench.md` for more details.