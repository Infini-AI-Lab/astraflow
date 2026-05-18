# Math Recipes

Math recipes train models on RLVR-style math tasks with M2PO.

Run one example from the repo root:

```bash
bash examples/math/qwen3-1.7b-m2po-2gpus-delta/scripts/run_qwen3-1.7b-m2po-2gpus-delta.sh
```

Complete guidance: [`docs/en/recipes/math.md`](../../docs/en/recipes/math.md).

---
**GPU Resources**

Most math recipes default to one 8xH100 node. The `qwen3-1.7b-m2po-2gpus-*`
recipes are smaller 2xH100 variants.
