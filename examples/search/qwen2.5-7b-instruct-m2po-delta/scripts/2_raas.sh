#!/bin/bash
set -euo pipefail
# [2/3] Launch RaaS inference server (SGLang + TCP receiver)
#
# Usage (terminal 2, after AstraFlow is ready):
#   bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/2_raas.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas.yaml}"
EXP_NAME="${EXP_NAME:-search}"
TRIAL_NAME="${TRIAL_NAME:-qwen2.5-7b-instruct-m2po-delta}"
export CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
RAAS_PORT="${RAAS_PORT:-19190}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
ASTRAFLOW_URL="${ASTRAFLOW_URL:-http://127.0.0.1:${ASTRAFLOW_PORT}}"

# Disable NCCL cuMem API to avoid conflicts with SGLang's pre-allocated KV cache.
export NCCL_CUMEM_ENABLE=0

# Required when search_client_type=async-search-access (workflow runs inside RaaS).
export RAG_SERVER_ADDR_DIR="${RAG_SERVER_ADDR_DIR:-${REPO_ROOT}/astraEnv/ASearcher/tmp-log/rag_server_addrs}"
export JINA_API_KEY="${JINA_API_KEY:-}"

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== RaaS Inference Server (SGLang + TCP receiver) ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "RaaS config       : ${RAAS_CONFIG}"
echo "GPUs              : ${CUDA_VISIBLE_DEVICES}"
echo "Port              : ${RAAS_PORT}"
echo "AstraFlow URL     : ${ASTRAFLOW_URL}"
echo "======================================================="

python3 -u -m astraflow.raas.server \
  --host "${RAAS_HOST}" \
  --port "${RAAS_PORT}" \
  --config "${EXPERIMENT_CONFIG}" \
  --config "${RAAS_CONFIG}" \
  --engine-id "${ENGINE_ID:-default}" \
  --astraflow-url "${ASTRAFLOW_URL}" \
  2>&1 | tee "${LOG_DIR}/raas.log"
