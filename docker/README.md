# AstraFlow Docker Images

## Prerequisites

### NVIDIA (CUDA)

- NVIDIA Container Toolkit installed. Install via the [NVIDIA apt guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian), then `sudo systemctl restart docker`.

### AMD (ROCm)

- ROCm-compatible GPU (gfx1100 / 7900 XTX, gfx942 / MI300, gfx950 / MI350). ROCm 7.2+.

## Available Images

| Dockerfile                   | Description                                      | Extras                         |
| ---------------------------- | ------------------------------------------------ | ------------------------------ |
| `Dockerfile.sglang`          | astraflow + SGLang + flash-attn                  | `-e ".[sglang]"`               |
| `Dockerfile.sglang-rocm`     | astraflow + SGLang + flash-attn (ROCm)           | gfx1100 / gfx942 / gfx950      |
| `Dockerfile.sglang.megatron` | `Dockerfile.sglang` + Megatron extras (TE, apex) | builds on `astraflow:sglang` |

All images use Python 3.12 in a [uv](https://docs.astral.sh/uv/)-managed venv at `/opt/venv`.

`Dockerfile.sglang.megatron` is only needed for the **Megatron training backend**
(it layers Transformer Engine and apex on top of the SGLang image). The FSDP
backend and inference do not require it.

## Pull pre-built image

```bash
docker pull astraflowai/astraflow:v0.1.1
```

This image is built from `Dockerfile.sglang` (astraflow + SGLang + flash-attn). Pin a
version tag (`v0.1.1`) for reproducibility; `:latest` tracks the most recent release.

## Build from source

```bash
cd /path/to/astraflow

# CUDA / NVIDIA
docker build -f docker/Dockerfile.sglang -t astraflow:sglang .

# Optional: add the Megatron training backend (Transformer Engine + apex) on top.
docker build -f docker/Dockerfile.sglang.megatron -t astraflow:sglang-megatron .

# ROCm / AMD
docker build -f docker/Dockerfile.sglang-rocm --build-arg GPU_ARCH=gfx1100 -t astraflow:sglang-rocm-gfx1100 .
docker build -f docker/Dockerfile.sglang-rocm --build-arg GPU_ARCH=gfx942  -t astraflow:sglang-rocm-gfx942  .
docker build -f docker/Dockerfile.sglang-rocm --build-arg GPU_ARCH=gfx950  -t astraflow:sglang-rocm-gfx950  .
```

## Quick Start

```bash
# CUDA / NVIDIA — pre-built image
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 -it astraflowai/astraflow:v0.1.1

# CUDA / NVIDIA — locally built
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 -it astraflow:sglang

# ROCm / AMD
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    --net=host --shm-size=16g -it astraflow:sglang-rocm-gfx1100
```

## Notes

- **Venv**: `/opt/venv` activated via `VIRTUAL_ENV` and `PATH` env vars — set in the image.
- **flash-attn**: Excluded from uv resolution (see `[tool.uv]` in `pyproject.toml`). Both Dockerfiles install it separately with `--no-build-isolation`. On AMD, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` enables Triton kernels.
- **Shared memory (`--shm-size`)**: Co-located trainer + RaaS + SGLang share `/dev/shm`. Default 64 MB OOMs. Size generously (`512g` NVIDIA, `16g` ROCm).
- **Open files (`--ulimit nofile`)**: Rollouts open many file descriptors. Default 1024 causes `[Errno 24]`. Use `--ulimit nofile=65536:65536`.

### ROCm (`Dockerfile.sglang-rocm`)

- **Base**: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`.
- **gfx942/gfx950** use upstream `sgl-project/sglang` v0.5.9. **gfx1100** uses the `mjc0608/sglang` fork for 7900 XTX patch since it's not officially supported.
- **megatron-core** is not installed — use FSDP backend on AMD.
