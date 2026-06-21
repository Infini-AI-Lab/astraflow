# Installation

> For AMD ROCm (MI300/MI325) installs, see [Installation (AMD ROCm)](installation-amd.md).

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

### Step 5 (optional): Install the Megatron training backend

Only needed if you want to train with the **Megatron-LM backend** (tensor /
pipeline / expert parallelism, MoE models). The default **FSDP** backend and
all inference need nothing here — skip to Step 6.

> **Prefer Docker?** Skip this entire step with the pre-built
> `astraflowai/astraflow:v0.1.1.megatron` image (see Option B below), which already
> bundles Transformer Engine + apex.

`megatron-core` and `mbridge` are already installed by Step 3. The Megatron
backend additionally uses **Transformer Engine** (fused LayerNorm + sequence
parallelism) and benefits from **apex** (fused LayerNorm / Adam). Both are
compiled from source against the installed PyTorch:

```bash
# nvcc must be on PATH (same CUDA toolchain as the flash-attn build above)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export NVTE_FRAMEWORK=pytorch

# Transformer Engine (required for the Megatron backend with TP/SP).
# The prebuilt transformer-engine wheels link libcublas.so.12 and do NOT load
# on a CUDA 13 install (ImportError: libcublas.so.12). Build TE from source
# against your CUDA 13 toolkit instead; nvidia-mathdx provides the build-time
# cuBLASDx / cuDNN frontend headers.
uv pip install nvidia-mathdx==25.6.0
uv pip install -v --no-build-isolation \
  "git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.13"

# apex (optional — Megatron falls back to Torch Norm / torch Adam if absent).
# APEX_CPP_EXT/APEX_CUDA_EXT select the fused kernels; FORCE_CUDA=1 builds them
# without a visible GPU.
git clone --depth 1 https://github.com/NVIDIA/apex.git /tmp/apex
cd /tmp/apex
FORCE_CUDA=1 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
  uv pip install -v --no-build-isolation .
cd - && rm -rf /tmp/apex
```

> If apex's build complains about a CUDA toolkit vs. PyTorch CUDA minor-version
> mismatch, the difference is safe to ignore — comment out the
> `check_cuda_torch_binary_vs_bare_metal` guard in apex's `setup.py`, or just
> skip apex (Transformer Engine is the only hard requirement). The
> `docker/Dockerfile.sglang.megatron` image automates all of this.

Verify the Megatron extras:

```bash
python -c "
import transformer_engine.pytorch  # noqa: F401
print('transformer-engine: OK')
try:
    from apex.normalization import FusedLayerNorm  # noqa: F401
    print('apex: OK')
except ImportError:
    print('apex: not installed (Torch Norm fallback)')
"
```

### Step 6: Verify installation

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

Pre-built images are published on Docker Hub — they skip the from-source steps above
entirely. Requires the NVIDIA Container Toolkit so `--gpus all` works. Choose the image
by **training backend**:

```bash
# FSDP backend (default) — covers most recipes
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 -it astraflowai/astraflow:v0.1.1

# Megatron-LM backend — adds Transformer Engine + apex (Step 5 above, pre-built in).
# Use this for `backend: megatron` (TP/PP/EP, MoE, large models).
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 -it astraflowai/astraflow:v0.1.1.megatron
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

> **Note on `--ulimit nofile`:** a recipe run drives many concurrent rollouts whose
> reward workers open a large number of file descriptors. The container's default
> `nofile` soft limit (1024) is far too low and the reward pool fails with
> `[Errno 24] Too many open files`. Raise it with `--ulimit nofile=65536:65536`.

The image bundles astraflow, SGLang, and flash-attn. Pin a version tag (`v0.1.1`) for
reproducibility; `:latest` tracks the most recent release. See `docker/README.md` for
build details and the NVIDIA Container Toolkit install guide.
