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

---
**Attention kernel**

The dense recipes (`qwen3-1.7b-m2po-2gpus-*`, `qwen3-8b-m2po-*`,
`llama3-8b-instruct-m2po-*`) set
`attn_impl: kernels-community/flash-attn2` — a prebuilt, ABI-matched
FlashAttention-2 kernel pulled from the Hugging Face `kernels` hub (fetched and
cached on first use; no source build). This is the working FA2 on the validated
stack (`torch 2.11+cu130`): the literal `attn_impl: flash_attention_2` would
instead load the local `flash-attn` wheel and crash with an `undefined symbol`
ABI error (`is_flash_attn_2_available()` is metadata-only, so it never catches
the broken import). It is also the same kernel as `cli_args.py`'s default, so
recipes that omit `attn_impl` get it too.

`sdpa` and `eager` remain available; `sdpa` works but relies on per-sequence
`position_ids` resets for packed block-diagonal masking, whereas FA2 varlen
derives the block-diagonal mask from `cu_seqlens` directly. The Qwen3.5 recipes
use `sdpa` (hybrid Gated-DeltaNet + attention model).
