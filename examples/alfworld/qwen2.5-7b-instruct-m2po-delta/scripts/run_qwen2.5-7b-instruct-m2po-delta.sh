#!/bin/bash
set -euo pipefail
# All-in-one launcher for ALFWorld AstraFlow v2 training (Qwen2.5-7B-Instruct, TCP delta).
#
# Launches 4 processes:
#   0. AgentBench ALFWorld task server (AGENTBENCH_PORT, must match YAML workflow_spec.task_server_url)
#   1. AstraFlow HTTP service         (CPU-only)
#   2. RaaS inference server          (SGLang, SERVICE_CUDA_VISIBLE_DEVICES)
#   3. Trainer                        (TRAINER_GPUS)
#
# Usage:
#   bash examples/alfworld/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh

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
# GPU assignments (default: 4 GPUs for inference, 4 for training)
export SERVICE_CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAINER_GPUS="${TRAINER_GPUS:-4,5,6,7}"
# Ports / URLs (each component gets its own port)
export RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT:-19861}"
# AgentBench task server port (must match task_server_url in yaml/experiment.yaml).
export AGENTBENCH_PORT="${AGENTBENCH_PORT:-5000}"

TRAINER_NPROC="$(echo "${TRAINER_GPUS}" | awk -F',' '{print NF}')"

# Single trainer -> override cleanup port list (no MODEL0/MODEL1 pair).
export ASTRAFLOW_CLEANUP_PORTS="${WEIGHT_TRANSFER_HTTP_PORT} 21000"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR.
# Defined in examples/_common/utils.sh.
astraflow_setup_env

# =============================================================================
# Part 3: Print info and clean up
# =============================================================================
echo "=== ALFWorld AstraFlow v2 (TCP delta weight transfer) ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "RaaS config       : ${RAAS_CONFIG}"
echo "RaaS GPUs         : ${SERVICE_CUDA_VISIBLE_DEVICES}"
echo "Trainer GPUs      : ${TRAINER_GPUS} (FSDP dp${TRAINER_NPROC})"
echo "RaaS port         : ${RAAS_PORT}"
echo "AstraFlow port    : ${ASTRAFLOW_PORT}"
echo "Sender HTTP       : ${WEIGHT_TRANSFER_HTTP_PORT}"
echo "AgentBench port   : ${AGENTBENCH_PORT}"
echo "WANDB mode        : ${WANDB_MODE:-online}"
echo "==========================================================="

trap astraflow_cleanup_trap EXIT INT TERM

# Kill leftover processes and shared memory from prior runs.
# Defined in examples/_common/utils.sh.
astraflow_kill_stale

# =============================================================================
# Part 4: Launch training
# =============================================================================
echo "[0/4] Starting AgentBench ALFWorld server..."
bash astraEnv/AgentBench/start-alfworld-server.sh

echo "[1/4] Starting AstraFlow HTTP service..."
CUDA_VISIBLE_DEVICES="" \
  python3 -u -m astraflow \
    --config "${EXPERIMENT_CONFIG}" \
    --port "${ASTRAFLOW_PORT}" \
    --host "${ASTRAFLOW_HOST}" \
    2>&1 | tee "${LOG_DIR}/astraflow.log" &
sleep 5

echo "[2/4] Starting RaaS inference server (SGLang + TCP receiver)..."
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

echo "[3/4] Starting trainer..."
CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT:-29541}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    "$@" \
    2>&1 | tee "${LOG_DIR}/trainer.log"
