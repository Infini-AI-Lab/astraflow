# Installation (AMD ROCm)

This page covers running AstraFlow on AMD Instinct GPUs (MI300/MI325, gfx942)
under ROCm. For NVIDIA installs, see [Installation](installation.md).

## Prerequisites

- Linux with ROCm 7.0+ kernel driver installed (verify with `rocminfo`)
- AMD Instinct MI300X / MI325X (gfx942 / CDNA3), 8 GPUs per node for the
  default math recipe (RaaS=4 + Trainer=4)
- One of:
  - **Docker** (used in the recipes below), or
  - **enroot + pyxis** under Slurm (for clusters without a Docker daemon)

## Image strategy

The official SGLang ROCm image already ships a ROCm-built PyTorch, SGLang, and
[aiter](https://github.com/ROCm/aiter) (the ROCm attention backend used by
SGLang inference). AstraFlow's ROCm image starts from a version-matched SGLang
ROCm image and layers AstraFlow on top without disturbing the base GPU stack:

- base: `lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x`
  - python 3.10 venv at `/opt/venv`, `torch 2.9.1+rocm7.2.0`, `sglang 0.5.12.post1`,
    `sgl_kernel`, `aiter`, ROCm `apex`.
- the AstraFlow Dockerfile (`docker/Dockerfile.rocm`) adds:
  - AstraFlow + its pure-python deps, constrained so pip cannot upgrade or replace
    the ROCm `torch`/`sglang`/`triton`/`numpy` shipped by the base
  - `megatron-core`, `mbridge`, `torchdata` via `--no-deps`
    (megatron-core's `numpy<2.0.0` pin conflicts with the base's numpy 2.x but
    is fine at runtime)
  - `transformers==5.6.1` (a patch over base's 5.6.0; see top-level `pyproject.toml`)
  - a **Triton-AMD `flash-attn`** for the trainer (see below)

## Option A: Docker build

```bash
cd /path/to/astraflow
docker build -f docker/Dockerfile.rocm -t astraflow:rocm .
```

The build pulls the ~25 GB SGLang ROCm base, installs AstraFlow's deps, and
finishes with an 8 s `flash-attn` install (no CK/CUDA compile — see "Why
flash-attn is required" below). Final image is ~75 GB.

Run the example math recipe:

```bash
# from a node with 8 MI300/MI325 GPUs and HF_TOKEN exported
bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full_amd.sh
```

The launcher wraps `docker run` with the device flags AstraFlow needs:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --network=host --shm-size=512g --ulimit nofile=65536:65536 \
  -v /home:/home -v "$PWD:/opt/astraflow" \
  -e HF_TOKEN -e WANDB_API_KEY \
  astraflow:rocm bash -lc "..."
```

`--device=/dev/kfd --device=/dev/dri --group-add video` give the container
access to the AMD GPUs. `--shm-size=512g` and `--ulimit nofile=65536:65536` are
needed for the same reasons as the CUDA image (see [docker/README.md]).

## Option B: enroot + pyxis (no Docker daemon)

For Slurm clusters that use pyxis but lack a Docker daemon, `examples/_common/build_astraflow_rocm.sh`
performs the equivalent build via two `srun` steps:

```bash
# Default destination: <repo>/.images/astraflow-rocm.sqsh
bash examples/_common/build_astraflow_rocm.sh
```

It (1) imports the SGLang ROCm base into a squashfs via pyxis `--container-save`,
then (2) layers the AstraFlow install into the saved image. Building requires
`--container-remap-root` so pip can write to the image's root-owned `/opt/venv`.

Launch via `srun --container-image=.images/astraflow-rocm.sqsh ...` in your job
script.

## Required runtime environment

The recipe launcher sets these; if you build your own:

```bash
# Triton-AMD flash-attn varlen on gfx942 (required — see below)
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# Allocator + NCCL/RCCL tweaks (also recommended on CUDA)
export NCCL_CUMEM_ENABLE=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# MIOpen needs writable cache + DB paths inside the container
export MIOPEN_USER_DB_PATH=/tmp/miopen/db
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen/cache
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
```

`CUDA_VISIBLE_DEVICES` works as-is — ROCm PyTorch maps it to the visible HIP
devices, so the recipe's `SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3` /
`TRAINER_MODEL0_GPUS=4,5,6,7` split works unchanged.

## Why `flash_attention_2` is required (not just a perf choice)

The FSDP trainer packs multiple sequences into one microbatch and passes
`cu_seq_lens_q/k` (`astraflow/train_worker/engine/fsdp_engine.py`). Transformers
honors those boundaries **only** in the `flash_attention_2` path. Under `sdpa`,
those kwargs are ignored and a single causal mask spans the whole packed
buffer, so packed sub-sequences attend across boundaries; the trainer's
recomputed logprobs then diverge from the rollout, importance weights explode,
and the M2PO policy gradient is corrupted while the task reward still looks
plausible. **`flash_attention_2` is a correctness requirement on ROCm whenever
sequence packing is used.**

The base SGLang ROCm image does not ship the `flash_attn` Python package
(SGLang uses `aiter` for inference). The AstraFlow Dockerfile installs it with
the **Triton-AMD backend**, which JITs Triton kernels on gfx942 instead of
compiling the CK/CUDA backend:

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    pip install flash-attn==2.8.3 --no-build-isolation
```

Install time is ~8 s (no CK compile). `flash_attn.flash_attn_varlen_func`
(what transformers' `flash_attention_2` calls) is available immediately.

## ROCm-specific code adaptations (already in AstraFlow)

Two minor adaptations live in the tree so a fresh ROCm install runs out of
the box; no user action required:

- `astraflow/train_worker/platforms/__init__.py` recognizes ROCm/HIP and
  returns `CudaPlatform` (NCCL maps to RCCL on ROCm PyTorch).
- `astraflow/train_worker/utils/functional/vocab_parallel.py` falls back to
  eager for `_gather_logprobs` on ROCm (inductor codegen of these reductions
  currently fails on gfx942). Override with
  `ASTRAFLOW_FORCE_TORCH_COMPILE=1` if you want to test compile.

## Verify the install

A quick health check inside the container (matches what the launcher does):

```python
import torch, flash_attn, sglang, transformers, astraflow
print('torch    ', torch.__version__, 'hip', torch.version.hip)
print('flash-attn', flash_attn.__version__)
print('sglang   ', sglang.__version__)
print('transformers', transformers.__version__)
print('astraflow OK')
```

Then run the example math recipe with a small `SMOKE_STEPS`:

```bash
SMOKE_STEPS=2 bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full_amd.sh
```

A healthy ROCm run logs `importance_sampling/importance_weight/avg ≈ 1.0000`
on the first PPO step. If you see values far from 1 (e.g. ≈ 0.4) or
`m2po_mean_m2 ≫ 0.01`, something is wrong with the attention path — most
likely `flash_attention_2` is not active. See
`docs/notes/cross-platform-fix.md` for the full diagnosis.

## Known limitations / notes

- **MoE expert parallelism via aiter** is not exercised by the math recipe. The
  base SGLang ROCm image ships the aiter EP kernels (see `sgl-workspace/aiter`).
- **Megatron-LM backend** (Transformer Engine + apex) is not built into the
  ROCm image. Use the FSDP backend on ROCm (the default for the math recipe).
- ROCm `xgrammar` in the base image pins `apache-tvm-ffi>=0.1.9` while the
  installed version is 0.1.8; the warning is harmless for math RL (no
  grammar-constrained decoding is used).
