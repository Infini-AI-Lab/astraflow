# Overview

AstraFlow is an asynchronous RL training system for large reasoning and agentic models.
The project is designed for distributed GPU clusters and supports multiple algorithms,
including GRPO, GSPO, PPO, DAPO, LitePPO, Dr.GRPO, REINFORCE++, and RLOO.

## What This Docs Tree Covers

- Core system architecture and module boundaries.
- Quickstart-oriented setup and first build commands.
- Contributor workflow for docs and development.

## Project Layout Snapshot

- `astraflow/`: top-level package.
- `astraflow/train_worker/`: training engine, API contracts, datasets, workflows, tests.
- `astraflow/raas/`: inference serving stack.
- `astraflow/workflow/`: rollout and reward workflows.
- `examples/`: runnable recipes and YAML configs.

## Scope Note

This Sphinx site is currently a bootstrap documentation surface. Additional AstraFlow
guides can be added incrementally as the docs tree expands.
