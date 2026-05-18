#!/bin/bash
set -euo pipefail
# [1a] Launch RaaS-A on compute-node-1 (8 GPUs: model0 dp=6 + model1 dp=2).
#
# Prerequisite: the AstraFlow HTTP service is already running on compute-node-0.
# ASTRAFLOW_HOST_EXTERNAL is REQUIRED -- the hostname/IP of compute-node-0,
# reachable from this node. run_*.sh on compute-node-0 prints the exact value.
#
# Usage (on compute-node-1):
#   ASTRAFLOW_HOST_EXTERNAL=<compute-node-0 host> \
#     bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/1_raas_a.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas_a.yaml}"
EXP_NAME="${EXP_NAME:-code-multi-agent}"
TRIAL_NAME="${TRIAL_NAME:-qwen3-8b-codegen-verifier-m2po-full-2node}"
export CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
RAAS_PORT="${RAAS_PORT:-19190}"
# REQUIRED: hostname/IP of compute-node-0 (the node running AstraFlow),
# reachable from this node. run_*.sh on compute-node-0 prints the exact value.
: "${ASTRAFLOW_HOST_EXTERNAL:?Set ASTRAFLOW_HOST_EXTERNAL to the compute-node-0 host (see the run_*.sh banner)}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
ASTRAFLOW_URL="${ASTRAFLOW_URL:-http://${ASTRAFLOW_HOST_EXTERNAL}:${ASTRAFLOW_PORT}}"

export NCCL_CUMEM_ENABLE=0

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== RaaS-A (compute-node-1, model0 dp=6 + model1 dp=2) ==="
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
  --engine-id "${ENGINE_ID:-raas_a}" \
  --astraflow-url "${ASTRAFLOW_URL}" \
  2>&1 | tee "${LOG_DIR}/raas_a.log"
