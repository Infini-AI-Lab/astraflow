# Qwen3.5-4B — Math RL (M2PO)

Text-only math RL on **Qwen/Qwen3.5-4B** with **M2PO**, context 8k, lr 5e-6,
DeepScaleR data, `math_verify` reward.

Qwen3.5-4B is a **hybrid Gated-DeltaNet + attention multimodal** checkpoint
(architecture `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`); these
recipes train it **text-only**. The checkpoint ships as an image-text-to-text
model, so AstraFlow loads it via the `AutoModelForImageTextToText` path (the
`model_type` is registered in `VALID_VISION_MODELS`), and the trainer uses
`attn_impl: sdpa` because a prebuilt flash-attn is not ABI-compatible with this
torch build.

Two variants:

| recipe | weight transfer |
|---|---|
| `qwen3.5-4b-m2po-full`  | full (push the whole model each sync) |
| `qwen3.5-4b-m2po-delta` | delta (push only changed weights) |

## Backend support

These recipes run on the **FSDP** trainer backend only. **The Megatron backend
does not support Qwen3.5 yet:** the HF→Megatron converter (`mbridge`) has no
`qwen3_5` bridge and Megatron-Core has no Gated-DeltaNet layer spec, so the
GDN-hybrid model cannot be built or weight-loaded under Megatron (TP/PP/EP).
Supporting it would require a new `mbridge` model plus a Megatron GDN layer spec
and weight mapping. Dense Qwen3 (standard attention) *does* run on Megatron — see
`examples/math/qwen3-8b-megatron-delta`.

## Validated environment

These recipes were validated end-to-end on the following stack (8× L40 /
Ada). The model and GDN kernels come from pip dependencies — there is no
hand-patched framework source:

| package | version |
|---|---|
| `torch` | `2.11.0+cu130` |
| `transformers` | `5.8.1` |
| `kernels` | `0.14.1` |
| `sglang` | `0.5.13.post1` (published release with `qwen3_5` support), served with `attention_backend: flashinfer` |
| `flash-linear-attention` (`fla`) | `0.5.0` |
| `flashinfer-python` | `0.6.12` (pulled by sglang) |
| attention impl | `sdpa` (set in `experiment.yaml`) |

> **Hopper (H100) note.** On sm_90 the GDN backward must use `fla`'s tilelang
> kernel (`FLA_TILELANG=1`) — `fla` blocks its triton path on Hopper as
> numerically wrong (fla#640) — and the tilelang JIT needs a full CUDA toolkit
> (`CUDA_HOME` with nvcc + CCCL headers; the pip-shipped nvcc has none). The
> trainer launch script (`scripts/3_trainer_model0.sh`) now detects Hopper and
> sets both automatically (respecting pre-set values). Re-validated end-to-end
> on 8×H100: baseline overall avg@k 47.9 (matches the L40 47.8), step 50 → 57.8.

> **Install note.** `pyproject.toml` pins the full validated stack:
> `transformers==5.8.1` (with `kernels>=0.14,<0.15`), `torch==2.11.0`, and
> `sglang==0.5.13.post1` — the published release that ships `qwen3_5` support (the
> older `0.5.12.post1` predated it). It pulls `flashinfer 0.6.12` in automatically,
> so `uv pip install -e ".[sglang]"` resolves the validated environment directly.

## GPU layout (default, 8 GPUs)

```text
SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3  ->  RaaS / SGLang inference (model0, dp=4)
TRAINER_MODEL0_GPUS=4,5,6,7           ->  Trainer model0 (FSDP, 4 GPUs)
```

Override those env vars to use different GPUs.

## Run

```bash
bash examples/math/qwen3.5-4b-m2po-full/scripts/run_qwen3.5-4b-m2po-full.sh
# delta variant:
bash examples/math/qwen3.5-4b-m2po-delta/scripts/run_qwen3.5-4b-m2po-delta.sh
```

The launcher starts three processes (AstraFlow HTTP service, RaaS/SGLang
inference, FSDP trainer). See `scripts/` for the per-process scripts and
`yaml/` for the experiment / RaaS configs.

## Validation

Trained end-to-end on the stack above; eval rises steadily over training.
Qwen3.5-4B-full, overall metrics across the eval suite:

| metric | step 0 | step 80 | Δ |
|---|---|---|---|
| overall avg@k | 47.8% | 57.4% | +9.6 |
| overall pass@k | 56.5% | 67.4% | +10.9 |

The table above was produced on the predecessor SGLang git build. Both variants
(`full` and `delta`) were subsequently re-validated end-to-end on the pinned
`sglang==0.5.13.post1` release: training completes with no crashes, full
(`shard_copy`) and delta (~7× compressed) weight transfer both function, and
eval holds at the baseline (overall avg@k ≈ 49–51% over a short run) — i.e. the
published-release pin introduces no regression versus the git build.
