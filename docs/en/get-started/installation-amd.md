# Installation (AMD ROCm)

Running AstraFlow on AMD Instinct GPUs (MI300X / MI325X, `gfx942`) under
ROCm. For the NVIDIA path, see [Installation](installation.md).

## Prerequisites

- Linux with the ROCm 7.0+ kernel driver installed (verify with `rocminfo`)
- AMD Instinct MI300X / MI325X (CDNA3 / `gfx942`)
- Docker

## Build the image

```bash
cd /path/to/astraflow
docker build -f docker/Dockerfile.rocm -t astraflow:rocm .
```

The build pulls the SGLang ROCm base (~25 GB) and finishes in a few minutes;
the resulting image is ~75 GB.

### What's in the image

`docker/Dockerfile.rocm` starts from the official SGLang ROCm image and layers
AstraFlow on top **without reinstalling the GPU stack**. Specifically:

- **Base** — `lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x`: Python 3.10 venv at
  `/opt/venv`, `torch 2.9.1+rocm7.2.0`, `sglang 0.5.12.post1`, `sgl_kernel`,
  [`aiter`](https://github.com/ROCm/aiter) (the ROCm attention backend SGLang
  uses for inference), ROCm `apex`, ROCm `triton`.
- **Pinned base versions** — a constraints file generated from
  `pip freeze` is passed to every install step so pip cannot replace the
  ROCm-built `torch`, `sglang`, `triton`, `numpy`, etc. with CUDA wheels.
- **AstraFlow + pure-python deps** — `pip install -e .` with the CUDA-only
  pins stripped from `pyproject.toml`. `transformers` is bumped to 5.6.1 (the
  patch fix AstraFlow needs) over the base's 5.6.0.
- **`megatron-core`, `mbridge`, `torchdata`, `torch_memory_saver`** —
  installed `--no-deps` (megatron-core's `numpy<2.0.0` pin conflicts with the
  base's numpy 2.x but is fine at runtime); their import sites are reached on
  the FSDP path.
- **Flash Attention (Triton-AMD backend)** —
  `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install flash-attn==2.8.3`. This
  JITs Triton kernels on `gfx942` with no CK/CUDA compile (seconds, not
  minutes). The Dockerfile sets `ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
  so the trainer's `flash_attention_2` path dispatches to it automatically
  inside the container.

> **`flash_attention_2` is a correctness requirement on ROCm, not a perf
> choice.** AstraFlow's FSDP trainer packs multiple sequences per microbatch
> and passes `cu_seq_lens_q/k`; transformers honors those boundaries only in
> the `flash_attention_2` path. Under `sdpa` the packed sub-sequences attend
> across boundaries, the trainer's recomputed logprobs diverge from the
> rollout, importance weights explode, and the M2PO policy gradient is
> corrupted — while the task reward still looks plausible. The Triton-AMD
> `flash_attn` install above is what keeps this path correct.

## Run

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --network=host --shm-size=512g --ulimit nofile=65536:65536 \
  -v /home:/home \
  -e HF_TOKEN -e WANDB_API_KEY \
  -it astraflow:rocm
```

`--device=/dev/kfd --device=/dev/dri --group-add video` give the container
access to the AMD GPUs. The recipe launcher
(`examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full_amd.sh`)
wraps the same `docker run` and runs the recipe end-to-end.

> **Note on `--shm-size`:** RaaS stages received model weights under
> `/dev/shm/astraflow_weights` during weight transfer. The container default
> (64 MB) and small values like `16g` are far too small for 8B-scale recipes
> and cause `OSError: [Errno 28] No space left on device` during training.
> `--shm-size=512g` is a tmpfs *cap*, not a reservation, so it only consumes
> host RAM as actually used — set it comfortably below host RAM.

> **Note on `--ulimit nofile`:** a recipe run drives many concurrent rollouts
> whose reward workers open a large number of file descriptors. The container
> default (1024) is too low and the reward pool fails with `[Errno 24] Too
> many open files`. Raise it with `--ulimit nofile=65536:65536`.

## Verify installation

```bash
python -c "
import astraflow, torch, transformers
print(f'astraflow:    {astraflow.version.__version__}')
print(f'torch:        {torch.__version__}, hip={torch.version.hip}, GPUs: {torch.cuda.device_count()}')
print(f'transformers: {transformers.__version__}')
"

python -c "
import flash_attn, sglang
from flash_attn import flash_attn_varlen_func
print(f'flash-attn: {flash_attn.__version__} (varlen OK)')
print(f'sglang:     {sglang.__version__}')
"
```

Then run the math recipe with a small smoke step count:

```bash
SMOKE_STEPS=2 bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full_amd.sh
```

A healthy ROCm run logs `importance_sampling/importance_weight/avg ≈ 1.0000`
on the first PPO step. Values far from 1 (e.g. ≈ 0.4) or
`m2po_mean_m2 ≫ 0.01` indicate the attention path is broken (most commonly
`flash_attention_2` is not active).

## Notes

### ROCm-specific code adaptations (already in AstraFlow)

Two minor adaptations live in the tree so a fresh ROCm install works out of
the box — no user action needed:

- `astraflow/train_worker/platforms/__init__.py` recognizes ROCm/HIP and
  returns `CudaPlatform` (NCCL → RCCL is handled by ROCm PyTorch).
- `astraflow/train_worker/utils/functional/vocab_parallel.py` falls back to
  eager for `_gather_logprobs` on ROCm where inductor codegen of these
  reductions currently fails on `gfx942`. Override with
  `ASTRAFLOW_FORCE_TORCH_COMPILE=1` to try compile.

### Recommended runtime env

The recipe launcher sets these for you; if you exec into the container
manually:

```bash
export NCCL_CUMEM_ENABLE=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export MIOPEN_USER_DB_PATH=/tmp/miopen/db
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen/cache
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
```

`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is already set by `ENV` in
`Dockerfile.rocm`. `CUDA_VISIBLE_DEVICES` works as-is — ROCm PyTorch maps it
to HIP devices, so the recipe's
`SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3` / `TRAINER_MODEL0_GPUS=4,5,6,7` split
works unchanged.

### Known limitations

- **Megatron-LM backend** (Transformer Engine + apex) is not built into the
  ROCm image. Use the FSDP backend on ROCm (the math recipe's default).
- The base SGLang ROCm image's `xgrammar` pins `apache-tvm-ffi >= 0.1.9` while
  the installed version is `0.1.8`; the warning is harmless for math RL (no
  grammar-constrained decoding is used).
