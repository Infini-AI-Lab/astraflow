#!/bin/bash
set -euo pipefail
# [2] Launch AstraFlow HTTP service on compute-node-0. Binds 0.0.0.0 so
# RaaS-A on compute-node-1 can register via its --astraflow-url.
#
# Usage (on compute-node-0):
#   bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/2_astraflow.sh

export CUDA_VISIBLE_DEVICES=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
EXP_NAME="${EXP_NAME:-code-multi-agent}"
TRIAL_NAME="${TRIAL_NAME:-qwen3-8b-codegen-verifier-m2po-full-2node}"
ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== AstraFlow HTTP Service (compute-node-0) ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "Bind              : ${ASTRAFLOW_HOST}:${ASTRAFLOW_PORT}"
echo "================================================="

python3 -u -m astraflow \
  --config "${EXPERIMENT_CONFIG}" \
  --port "${ASTRAFLOW_PORT}" \
  --host "${ASTRAFLOW_HOST}" \
  2>&1 | tee "${LOG_DIR}/astraflow.log"
