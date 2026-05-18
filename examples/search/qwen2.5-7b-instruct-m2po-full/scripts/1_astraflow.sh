#!/bin/bash
set -euo pipefail
# [1/3] Launch AstraFlow HTTP service
#
# Usage (terminal 1):
#   bash examples/search/qwen2.5-7b-instruct-m2po-full/scripts/1_astraflow.sh

export CUDA_VISIBLE_DEVICES=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
EXP_NAME="${EXP_NAME:-search}"
TRIAL_NAME="${TRIAL_NAME:-qwen2.5-7b-instruct-m2po-full}"
ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"

# Required when search_client_type=async-search-access.
export RAG_SERVER_ADDR_DIR="${RAG_SERVER_ADDR_DIR:-${REPO_ROOT}/astraEnv/ASearcher/tmp-log/rag_server_addrs}"
export JINA_API_KEY="${JINA_API_KEY:-}"

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== AstraFlow HTTP Service ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "Port              : ${ASTRAFLOW_PORT}"
echo "==============================="

python3 -u -m astraflow \
  --config "${EXPERIMENT_CONFIG}" \
  --port "${ASTRAFLOW_PORT}" \
  --host "${ASTRAFLOW_HOST}" \
  2>&1 | tee "${LOG_DIR}/astraflow.log"
