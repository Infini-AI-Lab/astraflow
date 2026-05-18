# WebShop Recipes

WebShop recipes train agents on the AgentBench WebShop environment.

## Environment Server

Install AgentBench dependencies:

```bash
cd astraEnv/AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
conda install podman
docker pull longinyu/agentbench-webshop
```

Start the WebShop task server from the repo root before training:

```bash
bash examples/webshop/qwen2.5-7b-instruct-m2po-delta/scripts/0_webshop_server.sh
```

Run one example from the repo root:

```bash
bash examples/webshop/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh
```

Complete guidance: [`docs/en/recipes/agentbench.md`](../../docs/en/recipes/agentbench.md).

---
**GPU Resources**

These recipes default to one 8xH100 node.
