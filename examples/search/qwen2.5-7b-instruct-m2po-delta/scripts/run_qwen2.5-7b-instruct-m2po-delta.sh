#!/bin/bash
set -euo pipefail
# All-in-one launcher for AstraFlow v2 ASearcher training (Qwen2.5-7B-Instruct, SGLang, TCP delta).
#
# Launches 3 processes:
#   1. AstraFlow HTTP service (CPU-only)
#   2. RaaS inference server  (SGLang, SERVICE_CUDA_VISIBLE_DEVICES)
#   3. Trainer model0         (asearcher, TRAINER_MODEL0_GPUS)
#
# Usage:
#   bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas.yaml}"
export EXP_NAME="${EXP_NAME:-search}"
export TRIAL_NAME="${TRIAL_NAME:-qwen2.5-7b-instruct-m2po-delta}"

# GPU assignments (default: 4 GPUs for inference, 2 for training; 6,7 idle)
export SERVICE_CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAINER_MODEL0_GPUS="${TRAINER_MODEL0_GPUS:-4,5}"
TRAINER0_NPROC="$(echo "${TRAINER_MODEL0_GPUS}" | awk -F',' '{print NF}')"

RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
RAAS_PORT="${RAAS_PORT:-19190}"
ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"

# TCP weight-transfer port
export WEIGHT_TRANSFER_HTTP_PORT_MODEL0="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"

# Disable NCCL cuMem API to avoid conflicts with SGLang's pre-allocated KV cache.
export NCCL_CUMEM_ENABLE=0

# Use expandable segments to reduce CUDA memory fragmentation during PPO backward.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Required when search_client_type=async-search-access.
export RAG_SERVER_ADDR_DIR="${RAG_SERVER_ADDR_DIR:-${REPO_ROOT}/astraEnv/ASearcher/tmp-log/rag_server_addrs}"
if ! ls "${RAG_SERVER_ADDR_DIR}"/Host*_IP*.txt >/dev/null 2>&1; then
  echo "[ERROR] No RAG server address files found in ${RAG_SERVER_ADDR_DIR}" >&2
  echo "[ERROR] Start retrieval servers first (e.g. astraEnv/ASearcher/scripts/launch_rag_server.sh)." >&2
  exit 1
fi
export JINA_API_KEY="${JINA_API_KEY:-}"

# Auto-fallback when WANDB credentials are absent.
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" && -z "${WANDB_DISABLED:-}" ]]; then
  export WANDB_MODE="offline"
  export WANDB_DISABLED="true"
fi

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== ASearcher AstraFlow v2 (Qwen2.5-7B-Instruct, SGLang, TCP delta) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "RaaS config         : ${RAAS_CONFIG}"
echo "RaaS GPUs           : ${SERVICE_CUDA_VISIBLE_DEVICES}"
echo "Trainer model0 GPUs : ${TRAINER_MODEL0_GPUS} (asearcher, FSDP dp${TRAINER0_NPROC})"
echo "RaaS port           : ${RAAS_PORT}"
echo "AstraFlow port      : ${ASTRAFLOW_PORT}"
echo "Sender HTTP model0  : ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}"
echo "RAG dir             : ${RAG_SERVER_ADDR_DIR}"
echo "WANDB mode          : ${WANDB_MODE:-online}"
echo "========================================================"

cleanup() {
    trap - EXIT INT TERM
    echo "Shutting down..."
    kill -- -$$ 2>/dev/null || true
    pkill -9 -f astraflow.raas.server 2>/dev/null || true
    pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
    for port in ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0} 21000; do
        lsof -i :"$port" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
    done
    pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
    rm -rf /dev/shm/astraflow_* /dev/shm/_delta_timing.log 2>/dev/null || true
    rm -f /dev/shm/sem.mp-* 2>/dev/null || true
    wait 2>/dev/null
    exit 0
}
trap cleanup EXIT INT TERM

echo "Cleaning up stale processes..."
pkill -9 -f astraflow.raas.server 2>/dev/null || true
pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
for port in ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0} 21000; do
    lsof -i :"$port" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
done
pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
pkill -9 -f "compile_worker" 2>/dev/null || true
rm -rf /dev/shm/astraflow_* /dev/shm/_delta_timing.log 2>/dev/null || true
rm -f /dev/shm/sem.mp-* 2>/dev/null || true
sleep 2

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

export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"

echo "[3/3] Starting trainer model0 (asearcher)..."
CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS}" \
ASTRAFLOW_CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model0 \
    "$@" \
    2>&1 | tee "${LOG_DIR}/trainer_model0.log"
