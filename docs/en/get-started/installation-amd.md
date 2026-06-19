# Installation (AMD ROCm)

This page covers running AstraFlow on AMD Instinct GPUs (MI300X / MI325X,
`gfx942`) under ROCm. For the NVIDIA path, see [Installation](installation.md).

## Prerequisites

- Linux with the ROCm 7.0+ kernel driver installed (verify with `rocminfo`)
- AMD Instinct MI300X / MI325X (CDNA3 / `gfx942`)
- Docker

The base image and the dependency layout are designed around the official
**SGLang ROCm image** — it already ships a ROCm-built PyTorch 2.9.1, SGLang
0.5.12.post1, `sgl_kernel`, and [`aiter`](https://github.com/ROCm/aiter) (the
ROCm attention backend SGLang uses for inference). AstraFlow's ROCm build
layers on top **without reinstalling those packages** — replacing them would
pull CUDA wheels and break the GPU stack.

## Option A: Custom Installation (inside the SGLang ROCm base)

The manual, step-by-step path — these are the same operations the Dockerfile
runs.

### Step 1: Run the SGLang ROCm base container

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$PWD":/workspace/astraflow -w /workspace/astraflow \
  lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x bash
```

The base provides `python 3.10` in a venv at `/opt/venv` with
`torch 2.9.1+rocm7.2.0`, `sglang 0.5.12.post1`, `sgl_kernel`, `aiter`, ROCm
`apex`, and ROCm `triton 3.5.1+rocm7.2.0`. All steps below run inside this
container.

### Step 2: Pin the base's GPU stack as pip constraints

```bash
python docker/rocm/gen_constraints.py /tmp/rocm-constraints.txt
```

This writes a pip constraints file containing the **exact** versions of
`torch`, `torchvision`, `torchaudio`, `triton`, `sglang`, `sgl-kernel`, and
`numpy` that ship in the base image — using `importlib.metadata` so the long
ROCm local-version strings (e.g. `torch==2.9.1+rocm7.2.0.lw.git7e1940d4`) are
captured verbatim. Every subsequent `pip install` is run under this constraint
file so the GPU stack cannot be replaced.

### Step 3: Strip CUDA-only pins from pyproject and install AstraFlow

`pyproject.toml`'s `[project] dependencies` pins `torch==2.11.0` (and friends).
Strip those plus `megatron-core` and `mbridge` (whose `numpy<2.0.0` pin
conflicts with the base's numpy 2.x), then install:

```bash
python - <<'PY'
import re
STRIP = {"torch","torchaudio","torchvision","torchdata",
        "torch_memory_saver","torch-memory-saver","megatron-core","mbridge"}
norm = lambda s: re.split(r"[<>=!~\[ ]", s.strip(), 1)[0].strip().lower().replace("_","-")
out, in_deps = [], False
for line in open("pyproject.toml").readlines():
    s = line.strip()
    if s.startswith("dependencies = ["): in_deps = True; out.append(line); continue
    if in_deps and s.startswith("]"): in_deps = False; out.append(line); continue
    m = re.match(r'\s*"([^"]+)"', line) if in_deps else None
    if m and norm(m.group(1)) in STRIP: continue
    out.append(line)
open("pyproject.toml","w").writelines(out)
PY

pip install --no-build-isolation -e . -c /tmp/rocm-constraints.txt
pip install --no-deps megatron-core==0.13.1 mbridge==0.13.0 torchdata
pip install torch_memory_saver==0.0.9.post1 -c /tmp/rocm-constraints.txt
```

This installs AstraFlow + its pure-python deps, bumps `transformers` to 5.6.1
(the patch fix AstraFlow needs), and adds `megatron-core` / `mbridge` /
`torchdata` / `torch_memory_saver` whose import sites the FSDP path reaches.

### Step 4: Install Flash Attention (Triton-AMD backend)

The FSDP trainer packs multiple sequences per microbatch and passes
`cu_seq_lens_q/k`; transformers honors those boundaries **only** under
`attn_implementation="flash_attention_2"`. The base SGLang ROCm image does not
ship the `flash_attn` Python package (SGLang uses `aiter` internally), so
install it with the Triton-AMD backend, which JITs Triton kernels on `gfx942`
with no CK/CUDA compile:

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
pip install flash-attn==2.8.3 --no-build-isolation -c /tmp/rocm-constraints.txt
```

Install completes in seconds (no nvcc). At runtime keep
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` exported so `flash_attn` dispatches to
the Triton path.

> **`flash_attention_2` is a correctness requirement on ROCm, not a perf
> choice.** Under `sdpa` (or any non-varlen backend), the trainer's recomputed
> logprobs diverge from the rollout, importance weights explode, and the M2PO
> policy gradient is corrupted — while the task reward still looks plausible.

### Step 5: Verify installation

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

Then run the recipe with a small smoke step count:

```bash
SMOKE_STEPS=2 bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full_amd.sh
```

A healthy ROCm run logs `importance_sampling/importance_weight/avg ≈ 1.0000`
on the first PPO step. Values far from 1 (e.g. ≈ 0.4) or `m2po_mean_m2 ≫ 0.01`
indicate the attention path is broken (most commonly `flash_attention_2` is
not active).

## Option B: Docker

The Dockerfile bakes Steps 1-4 of Option A into an image:

```bash
cd /path/to/astraflow
docker build -f docker/Dockerfile.rocm -t astraflow:rocm .
```

The build pulls the SGLang ROCm base (~25 GB), installs AstraFlow's deps under
the constraints file, and finishes with the Triton-AMD `flash-attn` install.
Final image is ~75 GB.

Run a recipe (or an interactive shell):

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
wraps this `docker run` with the env vars the recipe needs.

> **Note on `--shm-size`:** same reason as the CUDA image — RaaS stages
> received model weights under `/dev/shm/astraflow_weights` during weight
> transfer, and the container default (64 MB) is far too small for 8B-scale
> recipes. `--shm-size=512g` is a tmpfs *cap*, not a reservation.

> **Note on `--ulimit nofile`:** a recipe run drives many concurrent rollouts
> whose reward workers open a large number of file descriptors. The container
> default (1024) is too low; raise it with `--ulimit nofile=65536:65536`.

> **Note on `FLASH_ATTENTION_TRITON_AMD_ENABLE`:** the Dockerfile already sets
> this via `ENV`, so the trainer's `flash_attention_2` path dispatches to the
> Triton-AMD backend automatically inside the container.

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

Set inside the container (the recipe launcher sets these for you):

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE  # set by ENV in Dockerfile.rocm
export NCCL_CUMEM_ENABLE=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export MIOPEN_USER_DB_PATH=/tmp/miopen/db
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen/cache
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
```

`CUDA_VISIBLE_DEVICES` works as-is — ROCm PyTorch maps it to HIP devices, so
the recipe's `SERVICE_CUDA_VISIBLE_DEVICES=0,1,2,3` /
`TRAINER_MODEL0_GPUS=4,5,6,7` split works unchanged.

### Known limitations

- **Megatron-LM backend** (Transformer Engine + apex) is not built into the
  ROCm image. Use the FSDP backend on ROCm (the math recipe's default).
- The base SGLang ROCm image's `xgrammar` pins `apache-tvm-ffi >= 0.1.9` while
  the installed version is `0.1.8`; the warning is harmless for math RL (no
  grammar-constrained decoding is used).
