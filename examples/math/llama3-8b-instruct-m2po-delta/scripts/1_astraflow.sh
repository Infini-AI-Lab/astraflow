#!/bin/bash
set -euo pipefail
# [1/3] Launch AstraFlow HTTP service
#
# Usage (terminal 1):
#   bash examples/math/llama3-8b-instruct-m2po-delta/scripts/1_astraflow.sh

export CUDA_VISIBLE_DEVICES=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
# Export EXP_NAME and TRIAL_NAME from the experiment YAML.
astraflow_load_experiment_env

export ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR. Defined in examples/_common/utils.sh.
astraflow_setup_env

echo "=== AstraFlow HTTP Service ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "Port              : ${ASTRAFLOW_PORT}"
echo "==============================="

python3 -u -m astraflow \
  --config "${EXPERIMENT_CONFIG}" \
  --port "${ASTRAFLOW_PORT}" \
  --host "${ASTRAFLOW_HOST}" \
  2>&1 | tee "${LOG_DIR}/astraflow.log"
