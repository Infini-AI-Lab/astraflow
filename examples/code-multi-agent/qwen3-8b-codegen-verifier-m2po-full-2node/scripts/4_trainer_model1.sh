#!/bin/bash
set -euo pipefail
# [4] Trainer for model1 (testcase generator) on compute-node-0 GPU 6,7 (FSDP dp=2).
#
# Usage (on compute-node-0, after AstraFlow is up and both RaaS are registered):
#   bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/4_trainer_model1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
EXP_NAME="${EXP_NAME:-code-multi-agent}"
TRIAL_NAME="${TRIAL_NAME:-qwen3-8b-codegen-verifier-m2po-full-2node}"
export CUDA_VISIBLE_DEVICES="${TRAINER_MODEL1_GPUS:-6,7}"
TRAINER_NPROC_PER_NODE="${TRAINER_NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}"

ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
RAAS_PORT="${RAAS_PORT:-19191}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"
export ASTRAFLOW_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

export WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL1:-19862}"

export NCCL_CUMEM_ENABLE=0

if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" && -z "${WANDB_DISABLED:-}" ]]; then
  export WANDB_MODE="offline"
  export WANDB_DISABLED="true"
fi

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== Trainer model1 — testcase generator (2-node, TCP full) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES}"
echo "AstraFlow           : ${ASTRAFLOW_URL}"
echo "RaaS (local)        : ${ASTRAFLOW_RAAS_URL}"
echo "Sender HTTP         : ${WEIGHT_TRANSFER_HTTP_PORT} (bound on 0.0.0.0)"
echo "WANDB mode          : ${WANDB_MODE:-online}"
echo "================================================================"

torchrun --nnodes 1 --nproc-per-node "${TRAINER_NPROC_PER_NODE}" \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL1:-29542}" \
  examples/launch_trainer.py \
  --config "${EXPERIMENT_CONFIG}" \
  --trainer trainer_model1 \
  "$@" 2>&1 | tee "${LOG_DIR}/trainer_model1.log"
