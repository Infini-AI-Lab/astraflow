# AstraFlow Docker Images

## Prerequisites

- NVIDIA Container Toolkit installed so `--gpus all` works. Install via the NVIDIA apt
  guide:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian,
  then restart Docker afterward with `sudo systemctl restart docker`.

## Available Images

| Dockerfile                   | Description                                      | Extras           |
| ---------------------------- | ------------------------------------------------ | ---------------- |
| `Dockerfile.sglang`          | astraflow + SGLang + flash-attn                  | `-e ".[sglang]"` |
| `Dockerfile.sglang.megatron` | `Dockerfile.sglang` + Megatron extras (TE, apex) | builds on `astraflow:sglang` |

The image is based on `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` with Python 3.12
managed by [uv](https://docs.astral.sh/uv/).

`Dockerfile.sglang.megatron` is only needed for the **Megatron training backend**
(it layers Transformer Engine and apex on top of the SGLang image). The FSDP
backend and inference do not require it.

## Pull pre-built image

Pre-built images are published on Docker Hub — use them to skip the build entirely.
Pick the one that matches your **training backend**:

```bash
# FSDP backend (default) — astraflow + SGLang + flash-attn. Covers most recipes.
docker pull astraflowai/astraflow:v0.1.2

# Megatron-LM backend — the above plus Transformer Engine + apex.
# Only needed when training with `backend: megatron` (TP/PP/EP, MoE, large models).
docker pull astraflowai/astraflow:v0.1.2.megatron
```

`v0.1.2` is built from `Dockerfile.sglang`; `v0.1.2.megatron` from
`Dockerfile.sglang.megatron`. The Megatron image is a strict superset, so if you are
unsure it also runs every FSDP recipe. Pin a version tag for reproducibility;
`:latest` tracks the most recent FSDP release. Both `v0.1.2` images are validated
end-to-end on 8×H100 (400-step math-RL runs incl. eval; FSDP with Qwen3.5-4B and
dense Qwen3-8B, Megatron with Qwen3-8B).

## Build from source

```bash

# NOTE: .dockerignore needs to be at the build context root (where you run docker build .)
cd /path/to/astraflow

docker build -f docker/Dockerfile.sglang -t astraflow:sglang .

# Optional: add the Megatron training backend (Transformer Engine + apex) on top.
docker build -f docker/Dockerfile.sglang.megatron -t astraflow:sglang-megatron .
```

## Quick Start

The recommended workflow: **the image provides the environment** (Python venv,
CUDA 13 toolkit, SGLang, flash-attn, fla kernels); **your local checkout provides
the code**. astraflow is installed *editable* from `/workspace/astraflow`, so
mounting your repo over that path makes the container run your code — code
changes take effect immediately, and you only rebuild the image when the
*environment* changes (dependency pins, CUDA, system libs).

```bash
# Train with YOUR local checkout inside the pre-built environment
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 \
  -v /path/to/astraflow:/workspace/astraflow \
  -v ~/.cache/huggingface:/hf -e HF_HOME=/hf \
  -e WANDB_API_KEY=<your-key> \
  astraflowai/astraflow:v0.1.2 \
  bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full.sh
```

- `-v /path/to/astraflow:/workspace/astraflow` — your repo replaces the baked-in
  code (outputs land in `data-experiments/`/`data-log/` inside your checkout).
- `-v ~/.cache/huggingface:/hf -e HF_HOME=/hf` — reuse your host model/dataset
  cache instead of re-downloading inside the container.

To poke around the image with its baked-in code instead (no mounts), start an
interactive shell:

```bash
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 -it astraflowai/astraflow:v0.1.2

# ...or the Megatron-backend image
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 -it astraflowai/astraflow:v0.1.2.megatron

# ...or run a locally built image
docker run --gpus all --net=host --shm-size=512g --ulimit nofile=65536:65536 -it astraflow:sglang
```

## Notes

- **flash-attn**: Excluded from uv dependency resolution (see `[tool.uv]` in
  `pyproject.toml`). The Dockerfile installs it explicitly with `--no-build-isolation`.
- **Venv**: The virtualenv at `/opt/venv` is activated via `VIRTUAL_ENV` and `PATH`
  environment variables.
- **Package versions**: Inference backend versions (SGLang, flash-attn) are defined in
  `pyproject.toml` extras — the Dockerfile references the `.[sglang]` extra rather than
  hardcoding versions.
- **Shared memory (`--shm-size`)**: A recipe run co-locates the trainer, RaaS, and
  SGLang in one container sharing a single `/dev/shm` (RaaS stages received weights
  under `/dev/shm/astraflow_weights`). The container default (64 MB) and small values
  like `16g` cause `OSError: [Errno 28] No space left on device` during training. Size
  it generously (`512g`); it is a tmpfs cap, not a reservation, so it only uses host
  RAM as actually consumed.
- **Open files (`--ulimit nofile`)**: Training launches many concurrent rollouts whose
  reward workers open a large number of file descriptors. The container's default
  `nofile` soft limit (1024) is too low and the reward pool fails with `[Errno 24] Too
  many open files`. Raise it with `--ulimit nofile=65536:65536` (already in the Quick
  Start commands above).
