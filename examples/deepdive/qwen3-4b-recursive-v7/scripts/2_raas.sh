#!/bin/bash
set -euo pipefail
# [2/3] Launch RaaS inference server (SGLang + TCP receiver)
#
# Usage (terminal 2, after AstraFlow is ready):
#   bash examples/textcraft/qwen3-4b-recursive/scripts/2_raas.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
# Export EXP_NAME and TRIAL_NAME from the experiment YAML.
astraflow_load_experiment_env

export CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="${ASTRAFLOW_URL:-http://127.0.0.1:${ASTRAFLOW_PORT}}"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR. Defined in examples/_common/utils.sh.
astraflow_setup_env

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
