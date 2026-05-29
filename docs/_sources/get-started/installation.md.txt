# Installation

## Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support

## Option A: Custom Installation (Conda)

These steps install AstraFlow into a local conda environment.

### Step 1: Create conda environment

```bash
conda create -n astraflow python=3.12 -y
conda activate astraflow
```

### Step 2: Install uv (fast pip replacement)

```bash
pip install -U "uv>=0.10"
```

> **uv ≥ 0.10 is required.** `pyproject.toml` uses `[tool.uv]` settings
> (`extra-build-dependencies`, `override-dependencies`) that older uv
> releases don't recognize. When uv hits an unknown `[tool.uv]` key it
> silently ignores the *entire* `[tool.uv]` table, so the
> `transformers==5.6.1` override (which must beat sglang's `==5.6.0` pin)
> is dropped and the install fails with an unsolvable
> `transformers` conflict. The Docker images install the latest uv via the
> official installer and are unaffected.

### Step 3: Install AstraFlow (core + dev tools)

```bash
uv pip install -e ".[dev]"
```

This installs all core dependencies (~260 packages) including PyTorch 2.11.0,
Transformers 5.6.1, Megatron-Core 0.13.1, Ray, W&B, and dev tools (pytest, ruff,
ipython).

### Step 4: Install Flash Attention and SGLang

#### Flash Attention

This is FlashAttention-**2** (`import flash_attn`), used by the FSDP trainer. It
is excluded from uv resolution (see `pyproject.toml` `[tool.uv]`) and built from
source, so it needs the CUDA 13 toolchain and a roomy build-temp directory:

```bash
# nvcc must be on PATH and match torch's CUDA (13.0 for torch 2.11+cu130)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

# nvcc writes GBs of intermediate files to $TMPDIR. Point it at local scratch
# with plenty of space — NOT a small/NFS-quota'd home, or the build fails with
# "nvFatbin error: empty input" or "Disk quota exceeded" from truncated temps.
export TMPDIR=/tmp/fa-build && mkdir -p "$TMPDIR"

uv pip install "flash-attn==2.8.3" --no-build-isolation
```

> On a single-GPU-arch box you can speed up the build and shrink its footprint
> with `FLASH_ATTN_CUDA_ARCHS=<arch> NVCC_THREADS=1` (e.g. `90` for H100, `80`
> for A100, `89` for L40/4090). These are optional — the real requirement is a
> roomy `TMPDIR`.

#### SGLang (inference backend)

Install via the project extra so uv applies the `[tool.uv]` overrides (the
`transformers==5.6.1` pin and the `flash-attn-4` pre-release allowance). SGLang
pulls in FlashAttention-**4** (`flash-attn-4`, a pre-release wheel) automatically
for its own attention backend — you do not install that one yourself.

```bash
uv pip install -e ".[sglang]"
```

### Step 5: Verify installation

```bash
python -c "
import astraflow, torch, transformers
print(f'astraflow:    {astraflow.version.__version__}')
print(f'torch:        {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
print(f'transformers: {transformers.__version__}')
"
```

Verify Flash Attention and SGLang:

```bash
python -c "
import flash_attn, sglang
print(f'flash-attn: {flash_attn.__version__}')
print(f'sglang:     {sglang.__version__}')
"
```

## Option B: Docker

A pre-built image is published on Docker Hub — it skips the from-source steps above
entirely. Requires the NVIDIA Container Toolkit so `--gpus all` works.

```bash
docker run --gpus all --net=host --shm-size=512g -it astraflowai/astraflow:v0.1.0
```

> **Note on `--shm-size`:** this sets the size of the container's `/dev/shm`. A
> recipe run co-locates the trainer, RaaS, and SGLang in a single container, all
> sharing one `/dev/shm` — in particular RaaS stages received model weights under
> `/dev/shm/astraflow_weights` during weight transfer. The container default
> (64 MB) and small values such as `16g` are far too small and cause
> `OSError: [Errno 28] No space left on device` during training on 8B-scale
> recipes. Size `/dev/shm` generously (`512g` above). It is a tmpfs *cap*, not a
> reservation, so it only consumes host RAM as actually used — set it to a value
> comfortably below host RAM.

The image bundles astraflow, SGLang, and flash-attn. Pin a version tag (`v0.1.0`) for
reproducibility; `:latest` tracks the most recent release. See `docker/README.md` for
build details and the NVIDIA Container Toolkit install guide.
