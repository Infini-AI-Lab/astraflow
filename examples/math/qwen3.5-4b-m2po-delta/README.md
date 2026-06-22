# Qwen3.5-4B — Math RL (M2PO), delta weight transfer

Same recipe as [`qwen3.5-4b-m2po-full`](../qwen3.5-4b-m2po-full/README.md), but
the trainer pushes **only changed weights** to the inference engine each sync
(`weight_transfer_strategies: delta`) instead of the full model.

See the [full recipe's README](../qwen3.5-4b-m2po-full/README.md) for the
validated environment (transformers 5.8.1 / kernels 0.14.1 / SGLang dev with
`qwen3_5`, `attention_backend: flashinfer` / `fla` 0.5.0 / torch 2.11.0+cu130),
GPU layout, install note, and validation results.

## Run

```bash
bash examples/math/qwen3.5-4b-m2po-delta/scripts/run_qwen3.5-4b-m2po-delta.sh
```
