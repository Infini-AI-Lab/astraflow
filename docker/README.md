# AstraFlow Docker Images

## Prerequisites

- NVIDIA Container Toolkit installed so `--gpus all` works. Install via the NVIDIA apt
  guide:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian,
  then restart Docker afterward with `sudo systemctl restart docker`.

## Available Images

| Dockerfile          | Description                     | Extras           |
| ------------------- | ------------------------------- | ---------------- |
| `Dockerfile.sglang` | astraflow + SGLang + flash-attn | `-e ".[sglang]"` |

The image is based on `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` with Python 3.12
managed by [uv](https://docs.astral.sh/uv/).

## Pull pre-built image

A pre-built image is published on Docker Hub — use it to skip the build entirely:

```bash
docker pull astraflowai/astraflow:v0.1.0
```

This image is built from `Dockerfile.sglang` (astraflow + SGLang + flash-attn). Pin a
version tag (`v0.1.0`) for reproducibility; `:latest` tracks the most recent release.

## Build from source

```bash

# NOTE: .dockerignore needs to be at the build context root (where you run docker build .)
cd /path/to/astraflow

docker build -f docker/Dockerfile.sglang -t astraflow:sglang .
```

## Quick Start

```bash
# Run the pre-built image with host network and all GPUs
docker run --gpus all --net=host --shm-size=16g -it astraflowai/astraflow:v0.1.0

# ...or run a locally built image
docker run --gpus all --net=host --shm-size=16g -it astraflow:sglang
```

## Notes

- **flash-attn**: Excluded from uv dependency resolution (see `[tool.uv]` in
  `pyproject.toml`). The Dockerfile installs it explicitly with `--no-build-isolation`.
- **Venv**: The virtualenv at `/opt/venv` is activated via `VIRTUAL_ENV` and `PATH`
  environment variables.
- **Package versions**: Inference backend versions (SGLang, flash-attn) are defined in
  `pyproject.toml` extras — the Dockerfile references the `.[sglang]` extra rather than
  hardcoding versions.
