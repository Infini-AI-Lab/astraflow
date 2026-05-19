# Installation

## Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support
- Conda (Miniconda or Anaconda)
- Python 3.10, 3.11, or 3.12

## Option A: Docker (fastest)

A pre-built image is published on Docker Hub and skips the steps below entirely.
Requires the NVIDIA Container Toolkit so `--gpus all` works.

```bash
docker run --gpus all --net=host --shm-size=16g -it astraflowai/astraflow:v0.1.0
```

The image bundles astraflow, SGLang, and flash-attn. Pin a version tag (`v0.1.0`) for
reproducibility; `:latest` tracks the most recent release. See `docker/README.md` for
build details and the NVIDIA Container Toolkit install guide.

## Option B: From source

The remaining steps install AstraFlow into a local conda environment.

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

### Step 4: Install pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

### Step 5: Install optional extras

#### Flash Attention

```bash
uv pip install "flash-attn==2.8.3" --no-build-isolation
```

#### vLLM (inference backend)

```bash
uv pip install "vllm==0.11.0"
```

#### SGLang (inference backend)

```bash
uv pip install "sglang==0.5.5.post1"
```

#### Transformer Engine

Transformer Engine requires cuDNN headers on the include path. PyTorch installs cuDNN
via pip, but the headers are not on the system path by default. Set the following
environment variables before building:

```bash
export CUDNN_INCLUDE_DIR=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'include'))")
export CUDNN_LIB_DIR=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))")
export CPATH="$CUDNN_INCLUDE_DIR:$CPATH"
export LIBRARY_PATH="$CUDNN_LIB_DIR:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:$LD_LIBRARY_PATH"
NVTE_FRAMEWORK=pytorch pip install "transformer-engine[pytorch]>=2.13.0" --no-build-isolation
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

Verify optional packages (if installed):

```bash
python -c "
import flash_attn, transformer_engine, vllm, sglang
print(f'flash-attn:          {flash_attn.__version__}')
print(f'transformer-engine:  {transformer_engine.__version__}')
print(f'vllm:                {vllm.__version__}')
print(f'sglang:              {sglang.__version__}')
"
```

## Troubleshooting

### `transformer-engine` fails with `cudnn.h: No such file or directory`

cuDNN is installed via pip (`nvidia-cudnn-cu12`) but the headers are not on the system
include path. Export `CPATH` and `LIBRARY_PATH` as shown in Step 5 before building.

### `nvidia-smi` not found but CUDA works

Some machines have the NVIDIA driver functional but `nvidia-smi` not in PATH. Verify
CUDA via PyTorch instead: `python -c "import torch; print(torch.cuda.is_available())"`.

### CUDA version mismatch between `nvcc` and PyTorch

This is normal. PyTorch ships its own CUDA runtime (e.g., 12.8). The system `nvcc`
version only matters when building custom CUDA extensions.
