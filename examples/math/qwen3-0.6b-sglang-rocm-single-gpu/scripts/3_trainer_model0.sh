#!/bin/bash
set -euo pipefail
# [3/3] Launch Trainer for model0 (FSDP, single proc, AMD ROCm)
#
# Usage (terminal 3, after AstraFlow and RaaS are ready):
#   bash examples/math/qwen3-0.6b-sglang-rocm-single-gpu/scripts/3_trainer_model0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
astraflow_load_experiment_env

export CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS:-0}"
export HIP_VISIBLE_DEVICES="${TRAINER_HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
TRAINER0_NPROC="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"

export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"

export WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"

astraflow_setup_env

echo "=== Trainer model0 (FSDP, single GPU, ROCm) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "GPU                 : ${CUDA_VISIBLE_DEVICES} (HIP: ${HIP_VISIBLE_DEVICES})"
echo "                    : FSDP dp${TRAINER0_NPROC}"
echo "AstraFlow           : ${ASTRAFLOW_URL}"
echo "RaaS                : ${ASTRAFLOW_RAAS_URL}"
echo "Sender HTTP         : ${WEIGHT_TRANSFER_HTTP_PORT}"
echo "WANDB mode          : ${WANDB_MODE:-offline}"
echo "================================================"

torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
  examples/launch_trainer.py \
  --config "${EXPERIMENT_CONFIG}" \
  --trainer trainer_model0 \
  "$@" 2>&1 | tee "${LOG_DIR}/trainer_model0.log"
