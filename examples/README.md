# Example Training Recipes

This directory contains runnable training recipes grouped by task type.

For a concrete example, run the 2-GPU Qwen3 math M2PO delta recipe from the
repo root:

```bash
bash examples/math/qwen3-1.7b-m2po-2gpus-delta/scripts/run_qwen3-1.7b-m2po-2gpus-delta.sh
```

That launcher starts AstraFlow, the RaaS inference server, and the trainer with
configs from `examples/math/qwen3-1.7b-m2po-2gpus-delta/yaml/`.

Browse task-specific recipes in their own subfolders:

- `examples/math/`: RLVR-style math training recipes.
- `examples/math-multi-agent/`: actor/verifier and multi-model math workflows.
- `examples/math-efficient-data/`: composable data-algorithm math recipes (GRESO, dynamic sampling, buffer replay).
- `examples/code/`: code generation training recipes.
- `examples/code-multi-agent/`: codegen/verifier and multi-agent code workflows.
- `examples/search/`: search-augmented agent training with local retrieval.
- `examples/alfworld/`: AgentBench ALFWorld interactive task recipes.
- `examples/webshop/`: AgentBench WebShop interactive task recipes.
- `examples/textcraft-recursive-agent/`: recursive multi-agent crafting (TextCraft) with spawn + shared-inventory sub-agents.

---
**GPU Resources**

Most recipes default to one 8xH100 node. The math folder also includes 2xH100
recipes, such as `examples/math/qwen3-1.7b-m2po-2gpus-delta/`.
