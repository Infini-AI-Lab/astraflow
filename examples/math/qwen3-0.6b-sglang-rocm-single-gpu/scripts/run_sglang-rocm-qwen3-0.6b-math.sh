#!/bin/bash
set -euo pipefail
# All-in-one launcher for AstraFlow v2 math training (Qwen3-0.6B, SGLang, AMD ROCm, single GPU).
#
# Launches 3 processes on a single AMD GPU (default: GPU 0, 7900 XTX / gfx1100):
#   1. AstraFlow HTTP service (CPU-only)
#   2. RaaS inference server  (SGLang, same GPU as trainer)
#   3. Trainer model0         (FSDP, single proc on same GPU)
#
# SGLang is configured with mem_fraction_static=0.8 and
# attention_backend=triton (flashinfer not available on gfx1100).
# Co-location works because Qwen3-0.6B is small enough (~1.2 GB model).
#
# Docker: docker/Dockerfile.sglang-rocm
# Base:   rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
#
# Usage:
#   bash examples/math/qwen3-0.6b-sglang-rocm-single-gpu/scripts/run_sglang-rocm-qwen3-0.6b-math.sh

# =============================================================================
# Part 1: Load env and settings
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
astraflow_load_experiment_env

# =============================================================================
# Part 2: Set up env
# =============================================================================
# Both SGLang and trainer share GPU 0 (7900 XTX / gfx1100, 24 GB VRAM).
export SERVICE_CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0}"
export TRAINER_MODEL0_GPUS="${TRAINER_MODEL0_GPUS:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# Ports / URLs
export RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export WEIGHT_TRANSFER_HTTP_PORT_MODEL0="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"

TRAINER0_NPROC="$(echo "${TRAINER_MODEL0_GPUS}" | awk -F',' '{print NF}')"

astraflow_setup_env

# =============================================================================
# Part 3: Print info and clean up
# =============================================================================
echo "=== AstraFlow v2 (Qwen3-0.6B, math, SGLang, ROCm, single GPU) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "RaaS config         : ${RAAS_CONFIG}"
echo "SGLang GPUs         : ${SERVICE_CUDA_VISIBLE_DEVICES}"
echo "Trainer model0 GPUs : ${TRAINER_MODEL0_GPUS} (FSDP dp${TRAINER0_NPROC})"
echo "RaaS port           : ${RAAS_PORT}"
echo "AstraFlow port      : ${ASTRAFLOW_PORT}"
echo "Sender HTTP model0  : ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}"
echo "WANDB mode          : ${WANDB_MODE:-offline}"
echo "HIP_VISIBLE_DEVICES : ${HIP_VISIBLE_DEVICES}"
echo "=========================================================="

trap astraflow_cleanup_trap EXIT INT TERM
astraflow_kill_stale

# =============================================================================
# Part 4: Launch training
# =============================================================================
echo "[1/3] Starting AstraFlow HTTP service..."
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
  python3 -u -m astraflow \
    --config "${EXPERIMENT_CONFIG}" \
    --port "${ASTRAFLOW_PORT}" \
    --host "${ASTRAFLOW_HOST}" \
    2>&1 | tee "${LOG_DIR}/astraflow.log" &
sleep 5

echo "[2/3] Starting RaaS inference server (SGLang / ROCm)..."
CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES}" \
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
  python3 -u -m astraflow.raas.server \
    --host "${RAAS_HOST}" \
    --port "${RAAS_PORT}" \
    --config "${EXPERIMENT_CONFIG}" \
    --config "${RAAS_CONFIG}" \
    --engine-id "${ENGINE_ID:-default}" \
    --astraflow-url "${ASTRAFLOW_URL}" \
    2>&1 | tee "${LOG_DIR}/raas.log" &
sleep 15

export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"

echo "[3/3] Starting trainer model0..."
CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS}" \
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model0 \
    "$@" \
    2>&1 | tee "${LOG_DIR}/trainer_model0.log"
