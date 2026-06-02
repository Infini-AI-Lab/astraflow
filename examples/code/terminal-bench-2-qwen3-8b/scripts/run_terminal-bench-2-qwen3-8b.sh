#!/bin/bash
set -euo pipefail
# All-in-one launcher for Qwen3-8B training with Terminal-Bench 2.0 eval.
# Harbor is invoked from the "harbor-tb2" conda environment by the workflow.
#
# GPU layout: 4+4 by default on H100 80GB
#   RaaS    : 0,1,2,3
#   Trainer : 4,5,6,7
#
# Usage:
#   bash examples/code/terminal-bench-2-qwen3-8b/scripts/run_terminal-bench-2-qwen3-8b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
astraflow_load_experiment_env

export SERVICE_CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAINER_GPUS="${TRAINER_GPUS:-4,5,6,7}"
TRAINER_NPROC="$(echo "${TRAINER_GPUS}" | awk -F',' '{print NF}')"

RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
RAAS_PORT="${RAAS_PORT:-19190}"
ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"

export WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT:-19862}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${USER:-$(whoami)}/xdg-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/${USER:-$(whoami)}/flashinfer-workspace}"
mkdir -p "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}"

export NCCL_CUMEM_ENABLE=0

if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" && -z "${WANDB_DISABLED:-}" ]]; then
  export WANDB_MODE="offline"
  export WANDB_DISABLED="true"
fi

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"
export ASTRAFLOW_VERIFY_WORK_ROOT="${ASTRAFLOW_VERIFY_WORK_ROOT:-${LOG_DIR}/verify-tmp}"
mkdir -p "${ASTRAFLOW_VERIFY_WORK_ROOT}"

echo "=== AstraFlow v2 + Terminal-Bench 2.0 eval (Qwen3-8B, H100 80GB 4+4) ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "RaaS config       : ${RAAS_CONFIG}"
echo "RaaS GPUs         : ${SERVICE_CUDA_VISIBLE_DEVICES} (model0 dp=4)"
echo "Trainer GPUs      : ${TRAINER_GPUS} (FSDP dp${TRAINER_NPROC})"
echo "RaaS port         : ${RAAS_PORT}"
echo "AstraFlow port    : ${ASTRAFLOW_PORT}"
echo "Harbor env        : harbor-tb2"
echo "Sender HTTP       : ${WEIGHT_TRANSFER_HTTP_PORT}"
echo "WANDB mode        : ${WANDB_MODE:-online}"
echo "========================================================================="

cleanup() {
    trap - EXIT INT TERM
    echo "Shutting down..."
    kill -- -$$ 2>/dev/null || true
    pkill -9 -f astraflow.raas.server 2>/dev/null || true
    pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
    for port in ${WEIGHT_TRANSFER_HTTP_PORT} 21000; do
        lsof -i :"$port" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
    done
    pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
    rm -f /dev/shm/areal_buffer_* /dev/shm/astraflow_* 2>/dev/null || true
    wait 2>/dev/null
    exit 0
}
trap cleanup EXIT INT TERM

echo "Cleaning up stale processes..."
pkill -9 -f astraflow.raas.server 2>/dev/null || true
pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
for port in ${WEIGHT_TRANSFER_HTTP_PORT} 21000; do
    lsof -i :"$port" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
done
pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
pkill -9 -f "compile_worker" 2>/dev/null || true
rm -f /dev/shm/areal_buffer_* /dev/shm/astraflow_* 2>/dev/null || true
sleep 2

echo "[1/3] Starting RaaS inference server..."
CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES}" \
  python3 -u -m astraflow.raas.server \
    --host "${RAAS_HOST}" \
    --port "${RAAS_PORT}" \
    --config "${EXPERIMENT_CONFIG}" \
    --config "${RAAS_CONFIG}" \
    --engine-id "${ENGINE_ID:-default}" \
    --astraflow-url "${ASTRAFLOW_URL}" \
    2>&1 | tee "${LOG_DIR}/raas.log" &
sleep 20

echo "[2/3] Starting AstraFlow HTTP service..."
CUDA_VISIBLE_DEVICES="" \
  python3 -u -m astraflow \
    --config "${EXPERIMENT_CONFIG}" \
    --port "${ASTRAFLOW_PORT}" \
    --host "${ASTRAFLOW_HOST}" \
    2>&1 | tee "${LOG_DIR}/astraflow.log" &
sleep 5

export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"

echo "[3/3] Starting trainer..."
CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" \
ASTRAFLOW_CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT:-29544}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model0 \
    "$@" 2>&1 | tee "${LOG_DIR}/trainer.log"
