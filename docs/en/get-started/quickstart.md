# Quick Start

## Architecture

AstraFlow runs as **3 separate services** that you launch in different terminals:

1. **RaaS** — Inference server (SGLang or vLLM backend), handles rollout generation
2. **AstraFlow** — Async data orchestration HTTP service, manages buffering and data flow
3. **Trainer** — Training worker launched via `torchrun`, performs gradient updates

Some agentic recipes (ALFWorld, WebShop) require a **4th service**: the task environment server (step 0).

## Launch a Training Run

Runnable recipes live under `examples/`. Each recipe ships a `yaml/` directory of
configs and numbered launch scripts under `scripts/`. Task-specific walkthroughs:
[math](../recipes/math.md), [code](../recipes/code.md),
[multi-agent](../recipes/multi-agent.md),
[agentbench](../recipes/agentbench.md) (ALFWorld + WebShop), and
[search](../recipes/search.md).

## Validate Code Style

```bash
pre-commit run --all-files
```

## Build & Serve Docs Locally

```bash
bash docs/build.sh
bash docs/serve.sh 8000
```
