#!/bin/bash
set -euo pipefail
# [3/3] Launch Trainer for model0 -- asearcher (TCP, sender_agent on local_rank 0)
#
# Usage (terminal 3, after AstraFlow and RaaS are ready):
#   bash examples/search/qwen2.5-7b-instruct-m2po-full/scripts/3_trainer_model0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
EXP_NAME="${EXP_NAME:-search}"
TRIAL_NAME="${TRIAL_NAME:-qwen2.5-7b-instruct-m2po-full}"
export CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS:-4,5}"
TRAINER_NPROC_PER_NODE="${TRAINER_NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')}"

RAAS_PORT="${RAAS_PORT:-19190}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"

export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"
export ASTRAFLOW_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# sender_agent (in trainer) listens on this HTTP port
export WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"

# Disable NCCL cuMem API to avoid conflicts with SGLang's pre-allocated KV cache
export NCCL_CUMEM_ENABLE=0

# Auto-fallback when WANDB credentials are absent
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" && -z "${WANDB_DISABLED:-}" ]]; then
  export WANDB_MODE="offline"
  export WANDB_DISABLED="true"
fi

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== Trainer model0 -- asearcher (TCP) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES}"
echo "AstraFlow           : ${ASTRAFLOW_URL}"
echo "RaaS                : ${ASTRAFLOW_RAAS_URL}"
echo "Sender HTTP         : ${WEIGHT_TRANSFER_HTTP_PORT}"
echo "WANDB mode          : ${WANDB_MODE:-online}"
echo "=========================================="

torchrun --nnodes 1 --nproc-per-node "${TRAINER_NPROC_PER_NODE}" \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
  examples/launch_trainer.py \
  --config "${EXPERIMENT_CONFIG}" \
  --trainer trainer_model0 \
  "$@" 2>&1 | tee "${LOG_DIR}/trainer_model0.log"
