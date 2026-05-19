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
pip install uv
```

### Step 3: Install AstraFlow (core + dev tools)

```bash
uv pip install -e ".[dev]"
```

This installs all core dependencies (~260 packages) including PyTorch 2.8.0,
Transformers 4.57.1, Megatron-Core 0.13.1, Ray, W&B, and dev tools (pytest, ruff,
ipython).

### Step 4: Install Flash Attention and SGLang

#### Flash Attention

```bash
uv pip install "flash-attn==2.8.3" --no-build-isolation
```

#### SGLang (inference backend)

```bash
uv pip install "sglang==0.5.5.post1"
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
docker run --gpus all --net=host --shm-size=16g -it astraflowai/astraflow:v0.1.0
```

The image bundles astraflow, SGLang, and flash-attn. Pin a version tag (`v0.1.0`) for
reproducibility; `:latest` tracks the most recent release. See `docker/README.md` for
build details and the NVIDIA Container Toolkit install guide.
