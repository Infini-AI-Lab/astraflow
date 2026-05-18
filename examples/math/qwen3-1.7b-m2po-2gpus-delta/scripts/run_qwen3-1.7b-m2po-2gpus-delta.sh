#!/bin/bash
set -euo pipefail
# All-in-one launcher for AstraFlow v2 math training (Qwen3-1.7B, M2PO, TCP).
#
# Launches 3 processes:
#   1. AstraFlow HTTP service (CPU-only)
#   2. RaaS inference server  (SGLang, SERVICE_CUDA_VISIBLE_DEVICES)
#   3. Trainer model0         (math, TRAINER_MODEL0_GPUS)
#
# Usage:
#   bash examples/math/qwen3-1.7b-m2po-2gpus-delta/scripts/run_qwen3-1.7b-m2po-2gpus-delta.sh

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
# Export EXP_NAME and TRIAL_NAME from the experiment YAML.
# Defined in examples/_common/utils.sh.
astraflow_load_experiment_env

# =============================================================================
# Part 2: Set up env
# =============================================================================
# GPU assignments (default: 1 GPU for inference, 1 GPU for training)
export SERVICE_CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0}"
export TRAINER_MODEL0_GPUS="${TRAINER_MODEL0_GPUS:-1}"
# Ports / URLs (each component gets its own port)
export RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export WEIGHT_TRANSFER_HTTP_PORT_MODEL0="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"

TRAINER0_NPROC="$(echo "${TRAINER_MODEL0_GPUS}" | awk -F',' '{print NF}')"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR.
# Defined in examples/_common/utils.sh.
astraflow_setup_env

# =============================================================================
# Part 3: Print info and clean up
# =============================================================================
echo "=== AstraFlow v2 (Qwen3-1.7B, math, M2PO, ctx7k, TCP delta, 2gpus) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "RaaS config         : ${RAAS_CONFIG}"
echo "RaaS GPUs           : ${SERVICE_CUDA_VISIBLE_DEVICES}"
echo "Trainer model0 GPUs : ${TRAINER_MODEL0_GPUS} (FSDP dp${TRAINER0_NPROC})"
echo "RaaS port           : ${RAAS_PORT}"
echo "AstraFlow port      : ${ASTRAFLOW_PORT}"
echo "Sender HTTP model0  : ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}"
echo "WANDB mode          : ${WANDB_MODE:-online}"
echo "=========================================================="

trap astraflow_cleanup_trap EXIT INT TERM

# Kill leftover processes and shared memory from prior runs.
# Defined in examples/_common/utils.sh.
astraflow_kill_stale

# =============================================================================
# Part 4: Launch training
# =============================================================================
echo "[1/3] Starting AstraFlow HTTP service..."
CUDA_VISIBLE_DEVICES="" \
  python3 -u -m astraflow \
    --config "${EXPERIMENT_CONFIG}" \
    --port "${ASTRAFLOW_PORT}" \
    --host "${ASTRAFLOW_HOST}" \
    2>&1 | tee "${LOG_DIR}/astraflow.log" &
sleep 5

echo "[2/3] Starting RaaS inference server (SGLang + TCP receiver)..."
CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES}" \
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
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model0 \
    "$@" \
    2>&1 | tee "${LOG_DIR}/trainer_model0.log"
