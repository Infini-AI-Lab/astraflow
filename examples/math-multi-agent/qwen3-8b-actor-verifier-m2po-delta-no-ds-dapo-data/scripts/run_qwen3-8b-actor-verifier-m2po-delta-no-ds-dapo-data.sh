#!/bin/bash
set -euo pipefail
# All-in-one launcher for AstraFlow v2 training with 2-model math actor-and-verify
# (Qwen3-8B, M2PO, TCP delta, NO dynamic-sampling filter).
#
# Same as qwen3-8b-actor-verifier-m2po-delta-deepscaler-data but with the buffer's filter_function
# left as the default (KeepAllFilter) -- zero-advantage prompts are NOT filtered.
#
# Two-model setup:
#   - model0 (solver):   generates a solution for the math problem
#   - model1 (verifier): approves or rejects the solution (at most once)
#   - If rejected, solver retries once with verifier's full output as context
#
# Launches 4 processes:
#   1. AstraFlow HTTP service (CPU-only)
#   2. RaaS inference server  (SGLang x2, SERVICE_CUDA_VISIBLE_DEVICES)
#   3. Trainer model0         (solver, TRAINER_MODEL0_GPUS)
#   4. Trainer model1         (verifier, TRAINER_MODEL1_GPUS)
#
# Usage:
#   bash examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-no-ds-dapo-data/scripts/run_qwen3-8b-actor-verifier-m2po-delta-no-ds-dapo-data.sh

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
# GPU assignments (default: 4 GPUs for inference, 2+2 for training)
export SERVICE_CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAINER_MODEL0_GPUS="${TRAINER_MODEL0_GPUS:-4,5}"
export TRAINER_MODEL1_GPUS="${TRAINER_MODEL1_GPUS:-6,7}"
# Ports / URLs (each component gets its own port)
export RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
# TCP weight-transfer ports (one per trainer)
export WEIGHT_TRANSFER_HTTP_PORT_MODEL0="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"
export WEIGHT_TRANSFER_HTTP_PORT_MODEL1="${WEIGHT_TRANSFER_HTTP_PORT_MODEL1:-19862}"

TRAINER0_NPROC="$(echo "${TRAINER_MODEL0_GPUS}" | awk -F',' '{print NF}')"
TRAINER1_NPROC="$(echo "${TRAINER_MODEL1_GPUS}" | awk -F',' '{print NF}')"

# Multi-trainer -> cleanup kills ports for both models + handshake.
export ASTRAFLOW_CLEANUP_PORTS="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0} ${WEIGHT_TRANSFER_HTTP_PORT_MODEL1} 21000"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR.
# Defined in examples/_common/utils.sh.
astraflow_setup_env

# =============================================================================
# Part 3: Print info and clean up
# =============================================================================
echo "=== AstraFlow v2 (2-model math actor-and-verify, Qwen3-8B, M2PO, TCP delta, no-filter) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "RaaS config         : ${RAAS_CONFIG}"
echo "RaaS GPUs           : ${SERVICE_CUDA_VISIBLE_DEVICES} (model0 dp=2, model1 dp=2)"
echo "Trainer model0 GPUs : ${TRAINER_MODEL0_GPUS} (solver,   FSDP dp${TRAINER0_NPROC})"
echo "Trainer model1 GPUs : ${TRAINER_MODEL1_GPUS} (verifier, FSDP dp${TRAINER1_NPROC})"
echo "RaaS port           : ${RAAS_PORT}"
echo "AstraFlow port      : ${ASTRAFLOW_PORT}"
echo "Sender HTTP model0  : ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}"
echo "Sender HTTP model1  : ${WEIGHT_TRANSFER_HTTP_PORT_MODEL1}"
echo "WANDB mode          : ${WANDB_MODE:-online}"
echo "================================================================================"

trap astraflow_cleanup_trap EXIT INT TERM

# Kill leftover processes and shared memory from prior runs.
# Defined in examples/_common/utils.sh.
astraflow_kill_stale

# =============================================================================
# Part 4: Launch training
# =============================================================================
echo "[1/4] Starting AstraFlow HTTP service..."
CUDA_VISIBLE_DEVICES="" \
  python3 -u -m astraflow \
    --config "${EXPERIMENT_CONFIG}" \
    --port "${ASTRAFLOW_PORT}" \
    --host "${ASTRAFLOW_HOST}" \
    2>&1 | tee "${LOG_DIR}/astraflow.log" &
sleep 5

echo "[2/4] Starting RaaS inference server (SGLang x2 + TCP receivers)..."
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

echo "[3/4] Starting trainer model0 (solver)..."
CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model0 \
    "$@" \
    2>&1 | tee "${LOG_DIR}/trainer_model0.log" &
sleep 5

echo "[4/4] Starting trainer model1 (verifier)..."
CUDA_VISIBLE_DEVICES="${TRAINER_MODEL1_GPUS}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL1}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER1_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL1:-29542}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model1 \
    "$@" \
    2>&1 | tee "${LOG_DIR}/trainer_model1.log"
