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

## Validated environment

These recipes were validated end-to-end on the following stack (8× L40 /
Ada). The model and GDN kernels come from pip dependencies — there is no
hand-patched framework source:

| package | version |
|---|---|
| `torch` | `2.11.0+cu130` |
| `transformers` | `5.8.1` |
| `kernels` | `0.14.1` |
| `sglang` | main/dev with `qwen3_5` support, served with `attention_backend: flashinfer` (validated build `0.5.6.post3.dev5643`) |
| `flash-linear-attention` (`fla`) | `0.5.0` |
| `flashinfer-python` | `0.6.11.post1` |
| attention impl | `sdpa` (set in `experiment.yaml`) |

> **Install note.** The repo's default `pyproject.toml` pins
> (`transformers==5.6.1`, `sglang==0.5.12.post1`) resolve and *load* `qwen3_5`,
> but the recipe was trained on the stack above — install it out of band
> (e.g. a dedicated env). A `pyproject` pin bump is intentionally **not** part
> of this PR: the validated SGLang is a dev build (older than the pinned
> `0.5.12.post1`), so it cannot be pinned to a clean release yet. Bumping the
> pins is deferred to a separate, explicitly-tested PR once a published SGLang
> release with `qwen3_5` support is available.

## GPU layout (default, 8 GPUs)

```
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
