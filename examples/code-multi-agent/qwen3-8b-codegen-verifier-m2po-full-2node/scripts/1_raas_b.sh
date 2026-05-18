#!/bin/bash
set -euo pipefail
# [1b] Launch RaaS-B on compute-node-0 GPUs 0..3 (model0 dp=3 + model1 dp=1).
#
# Usage (on compute-node-0):
#   bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/1_raas_b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas_b.yaml}"
EXP_NAME="${EXP_NAME:-code-multi-agent}"
TRIAL_NAME="${TRIAL_NAME:-qwen3-8b-codegen-verifier-m2po-full-2node}"
export CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
RAAS_PORT="${RAAS_PORT:-19191}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
ASTRAFLOW_URL="${ASTRAFLOW_URL:-http://127.0.0.1:${ASTRAFLOW_PORT}}"

export NCCL_CUMEM_ENABLE=0

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== RaaS-B (compute-node-0, model0 dp=3 + model1 dp=1) ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "RaaS config       : ${RAAS_CONFIG}"
echo "GPUs              : ${CUDA_VISIBLE_DEVICES}"
echo "Port              : ${RAAS_PORT}"
echo "AstraFlow URL     : ${ASTRAFLOW_URL}"
echo "============================================================"

python3 -u -m astraflow.raas.server \
  --host "${RAAS_HOST}" \
  --port "${RAAS_PORT}" \
  --config "${EXPERIMENT_CONFIG}" \
  --config "${RAAS_CONFIG}" \
  --engine-id "${ENGINE_ID:-raas_b}" \
  --astraflow-url "${ASTRAFLOW_URL}" \
  2>&1 | tee "${LOG_DIR}/raas_b.log"
