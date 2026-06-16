#!/bin/bash
# AMD/ROCm launcher for the qwen3-8b-m2po-full recipe (MI300/MI325, gfx942).
#
# Runs the all-in-one recipe (AstraFlow service + RaaS/SGLang + FSDP trainer, all
# on one 8-GPU node) inside the astraflow ROCm docker image.
#
# Build the image first (on a compute node that has a docker daemon):
#   docker build -f docker/Dockerfile.rocm -t astraflow:rocm .
#
# Then, on a node with 8 MI300/MI325 GPUs:
#   bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full_amd.sh
# Under Slurm (pyxis-free; uses the node's docker daemon):
#   srun --gres=gpu:8 --nodes=1 --ntasks=1 bash examples/math/.../run_qwen3-8b-m2po-full_amd.sh
#
# Env knobs:
#   CONTAINER_IMAGE  docker image tag (default: astraflow:rocm)
#   SMOKE_STEPS      cap total_train_steps for a quick bring-up check
#   HF_TOKEN         HuggingFace token (Qwen3-8B download)
#   WANDB_API_KEY    if unset, W&B runs offline
#   HF_HOME          HF cache (default: $HOME/.cache/huggingface, mounted from host)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-astraflow:rocm}"
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"

# ROCm recipe overrides:
#   (attn stays flash_attention_2 — the recipe default. The trainer packs sequences and
#    relies on flash-attn varlen/cu_seqlens; sdpa silently breaks packed attention and the
#    recomputed logprobs diverge from the rollout. We ship a Triton-AMD flash-attn, enabled
#    via FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE below — so NO attn_impl override on ROCm.)
#   recover.mode=disabled  — no checkpoint writes by default (team policy / smoke runs)
OVERRIDES=(recover.mode=disabled)
# The recipe yaml hardcodes stats_logger.wandb.mode=online (which overrides the
# WANDB_MODE env var). Disable W&B unless an API key is provided.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  OVERRIDES+=(stats_logger.wandb.mode=disabled)
fi
[[ -n "${SMOKE_STEPS:-}" ]] && OVERRIDES+=("total_train_steps=${SMOKE_STEPS}")

# Command run inside the container.
read -r -d '' INNER <<INNER_EOF || true
set -uo pipefail
cd /opt/astraflow
export PYTHONPATH=/opt/astraflow:\${PYTHONPATH:-}
export HF_HOME=${HF_HOME_HOST}
export NCCL_CUMEM_ENABLE=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# Use the Triton-AMD flash-attn backend (gfx942 varlen) for the trainer's
# flash_attention_2 path — required for correct packed-sequence attention.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export MIOPEN_USER_DB_PATH=/tmp/miopen/db MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen/cache
mkdir -p \$MIOPEN_USER_DB_PATH \$MIOPEN_CUSTOM_CACHE_DIR
[[ -z "\${WANDB_API_KEY:-}" ]] && { export WANDB_MODE=offline WANDB_DISABLED=true; }
python -c "import torch;print('torch',torch.__version__,'hip',torch.version.hip,'gpus',torch.cuda.device_count())"
bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full.sh ${OVERRIDES[*]}
INNER_EOF

exec docker run --rm --name astraflow_run \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --network=host --shm-size=512g --ulimit nofile=65536:65536 \
  -e HF_TOKEN="${HF_TOKEN:-}" -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -v /home:/home \
  -v "${REPO_ROOT}:/opt/astraflow" \
  "${CONTAINER_IMAGE}" \
  bash -lc "${INNER}"
