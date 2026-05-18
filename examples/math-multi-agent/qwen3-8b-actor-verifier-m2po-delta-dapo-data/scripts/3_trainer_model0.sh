#!/bin/bash
set -euo pipefail
# [3/4] Launch Trainer for model0 -- solver (TCP, sender_agent on local_rank 0)
#
# Usage (terminal, after AstraFlow and RaaS are ready):
#   bash examples/math-multi-agent/qwen3-8b-actor-verifier-m2po-delta-dapo-data/scripts/3_trainer_model0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
# Export EXP_NAME and TRIAL_NAME from the experiment YAML.
astraflow_load_experiment_env

export CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS:-4,5}"
TRAINER0_NPROC="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"

export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="http://127.0.0.1:${ASTRAFLOW_PORT}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_PORT}"

# sender_agent (in trainer) listens on this HTTP port
export WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR. Defined in examples/_common/utils.sh.
astraflow_setup_env

echo "=== Trainer model0 -- solver (TCP) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES} (FSDP dp${TRAINER0_NPROC})"
echo "AstraFlow           : ${ASTRAFLOW_URL}"
echo "RaaS                : ${ASTRAFLOW_RAAS_URL}"
echo "Sender HTTP         : ${WEIGHT_TRANSFER_HTTP_PORT}"
echo "WANDB mode          : ${WANDB_MODE:-online}"
echo "=============================================="

torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
  examples/launch_trainer.py \
  --config "${EXPERIMENT_CONFIG}" \
  --trainer trainer_model0 \
  "$@" 2>&1 | tee "${LOG_DIR}/trainer_model0.log"
