#!/bin/bash
set -euo pipefail
# All-in-one launcher for the compute-node-0 side of the 2-node recipe:
#   AstraFlow service -> RaaS-B -> Trainer model0 -> Trainer model1
#
# NOTE: "compute-node-0" / "compute-node-1" are PLACEHOLDERS for your two nodes
# (compute-node-0 = this node; compute-node-1 = the RaaS-A node). Substitute your
# real hostnames -- or just copy the launch command this script prints below.
#
# Before running this, launch RaaS-A on compute-node-1 manually:
#   ssh compute-node-1
#   cd <repo>
#   ASTRAFLOW_HOST_EXTERNAL=compute-node-0 \
#     bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/1_raas_a.sh
#
# Ports that must be reachable from compute-node-1 to compute-node-0:
#   8000  (AstraFlow HTTP)
#   19861 (sender HTTP, trainer model0)
#   19862 (sender HTTP, trainer model1)
#   21000 (weight-transfer handshake)
#
# Usage (on compute-node-0):
#   bash examples/code-multi-agent/qwen3-8b-codegen-verifier-m2po-full-2node/scripts/run_qwen3-8b-codegen-verifier-m2po-full-2node.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_B_CONFIG="${RAAS_B_CONFIG:-${YAML_DIR}/raas_b.yaml}"
export EXP_NAME="${EXP_NAME:-code-multi-agent}"
export TRIAL_NAME="${TRIAL_NAME:-qwen3-8b-codegen-verifier-m2po-full-2node}"

# GPU assignments on compute-node-0.
export RAAS_B_GPUS="${RAAS_B_GPUS:-0,1,2,3}"
export TRAINER_MODEL0_GPUS="${TRAINER_MODEL0_GPUS:-4,5}"
export TRAINER_MODEL1_GPUS="${TRAINER_MODEL1_GPUS:-6,7}"
TRAINER0_NPROC="$(echo "${TRAINER_MODEL0_GPUS}" | awk -F',' '{print NF}')"
TRAINER1_NPROC="$(echo "${TRAINER_MODEL1_GPUS}" | awk -F',' '{print NF}')"

RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
RAAS_B_PORT="${RAAS_B_PORT:-19191}"
ASTRAFLOW_HOST="${ASTRAFLOW_HOST:-0.0.0.0}"
ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
ASTRAFLOW_URL_LOCAL="http://127.0.0.1:${ASTRAFLOW_PORT}"
# Hostname of THIS node (compute-node-0); printed in the RaaS-A launch command
# below so it can be copy-pasted on compute-node-1.
ASTRAFLOW_NODE_HOST="$(hostname -f 2>/dev/null || hostname)"

export WEIGHT_TRANSFER_HTTP_PORT_MODEL0="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"
export WEIGHT_TRANSFER_HTTP_PORT_MODEL1="${WEIGHT_TRANSFER_HTTP_PORT_MODEL1:-19862}"

export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" && -z "${WANDB_DISABLED:-}" ]]; then
  export WANDB_MODE="offline"
  export WANDB_DISABLED="true"
fi

LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== AstraFlow v2 — 2-node MRaaS (Qwen3-8B, full, code 2-agent v2) ==="
echo "This launches compute-node-0 side only (AstraFlow, RaaS-B, 2 trainers)."
echo "Launch RaaS-A on compute-node-1 FIRST (see banner below)."
echo
echo "RaaS-B GPUs         : ${RAAS_B_GPUS} (model0 dp=3, model1 dp=1, port ${RAAS_B_PORT})"
echo "Trainer model0 GPUs : ${TRAINER_MODEL0_GPUS} (FSDP dp${TRAINER0_NPROC}, sender :${WEIGHT_TRANSFER_HTTP_PORT_MODEL0})"
echo "Trainer model1 GPUs : ${TRAINER_MODEL1_GPUS} (FSDP dp${TRAINER1_NPROC}, sender :${WEIGHT_TRANSFER_HTTP_PORT_MODEL1})"
echo "AstraFlow           : ${ASTRAFLOW_HOST}:${ASTRAFLOW_PORT}"
echo
echo "  >>> On compute-node-1 (your RaaS-A node), run:"
echo "      ASTRAFLOW_HOST_EXTERNAL=${ASTRAFLOW_NODE_HOST} \\"
echo "        bash ${SCRIPT_DIR}/scripts/1_raas_a.sh"
echo "      (if ${ASTRAFLOW_NODE_HOST} is unreachable from compute-node-1, use this node's IP)"
echo "  >>> Expected RaaS-A registration: engine-id=raas_a at http://compute-node-1:19190"
echo "=========================================================================="

cleanup() {
    trap - EXIT INT TERM
    echo "Shutting down compute-node-0 processes..."
    kill -- -$$ 2>/dev/null || true
    pkill -9 -f astraflow.raas.server 2>/dev/null || true
    pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
    for port in ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0} ${WEIGHT_TRANSFER_HTTP_PORT_MODEL1} 21000; do
        lsof -i :"$port" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
    done
    pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
    rm -rf /dev/shm/astraflow_* /dev/shm/_delta_timing.log 2>/dev/null || true
    rm -f /dev/shm/sem.mp-* 2>/dev/null || true
    wait 2>/dev/null
    exit 0
}
trap cleanup EXIT INT TERM

echo "Cleaning up stale processes on compute-node-0..."
pkill -9 -f astraflow.raas.server 2>/dev/null || true
pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
for port in ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0} ${WEIGHT_TRANSFER_HTTP_PORT_MODEL1} 21000; do
    lsof -i :"$port" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true
done
pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
pkill -9 -f "compile_worker" 2>/dev/null || true
rm -rf /dev/shm/astraflow_* /dev/shm/_delta_timing.log 2>/dev/null || true
rm -f /dev/shm/sem.mp-* 2>/dev/null || true
sleep 2

echo "[1/4] Starting AstraFlow HTTP service on ${ASTRAFLOW_HOST}:${ASTRAFLOW_PORT}..."
CUDA_VISIBLE_DEVICES="" \
  python3 -u -m astraflow \
    --config "${EXPERIMENT_CONFIG}" \
    --port "${ASTRAFLOW_PORT}" \
    --host "${ASTRAFLOW_HOST}" \
    2>&1 | tee "${LOG_DIR}/astraflow.log" &
sleep 15

echo "[2/4] Starting RaaS-B (local, GPUs ${RAAS_B_GPUS}, port ${RAAS_B_PORT})..."
CUDA_VISIBLE_DEVICES="${RAAS_B_GPUS}" \
  python3 -u -m astraflow.raas.server \
    --host "${RAAS_HOST}" \
    --port "${RAAS_B_PORT}" \
    --config "${EXPERIMENT_CONFIG}" \
    --config "${RAAS_B_CONFIG}" \
    --engine-id "raas_b" \
    --astraflow-url "${ASTRAFLOW_URL_LOCAL}" \
    2>&1 | tee "${LOG_DIR}/raas_b.log" &
sleep 10

echo ">>> Waiting for RaaS-A on compute-node-1 to register with AstraFlow..."
echo ">>> (If you haven't launched it yet, do so now.)"
sleep 5

export ASTRAFLOW_URL="${ASTRAFLOW_URL_LOCAL}"
export ASTRAFLOW_RAAS_URL="http://127.0.0.1:${RAAS_B_PORT}"

echo "[3/4] Starting trainer model0 (code generator)..."
CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS}" \
ASTRAFLOW_CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model0 \
    2>&1 | tee "${LOG_DIR}/trainer_model0.log" &
sleep 3

echo "[4/4] Starting trainer model1 (testcase generator)..."
CUDA_VISIBLE_DEVICES="${TRAINER_MODEL1_GPUS}" \
ASTRAFLOW_CUDA_VISIBLE_DEVICES="${TRAINER_MODEL1_GPUS}" \
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL1}" \
  torchrun --nnodes 1 --nproc-per-node "${TRAINER1_NPROC}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL1:-29542}" \
    examples/launch_trainer.py \
    --config "${EXPERIMENT_CONFIG}" \
    --trainer trainer_model1 \
    2>&1 | tee "${LOG_DIR}/trainer_model1.log"
