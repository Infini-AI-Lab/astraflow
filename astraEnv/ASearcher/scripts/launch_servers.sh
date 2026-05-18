#!/bin/bash
set -xeuo pipefail

# Usage:
#   bash astraEnv/ASearcher/scripts/launch_servers.sh <gpu_ids> <num_servers>
#   e.g. bash astraEnv/ASearcher/scripts/launch_servers.sh 6,7 1
#
# Optional env vars:
#   START_PORT       (default: 7000)
#   RETRIEVER_NAME   (default: e5)
#   RETRIEVER_PATH   (default: intfloat/e5-base-v2)
#   TOPK             (default: 3)
#   USE_FAISS_GPU    (default: 1, set 0 to disable)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu_ids> <num_servers>"
  echo "  e.g. $0 6,7 1"
  exit 1
fi

GPU_IDS="$1"
NUM_SERVERS="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WIKI2018_WORK_DIR="${PROJECT_ROOT}/data/wiki2018"
RAG_SERVER_ADDR_DIR="${RAG_SERVER_ADDR_DIR:-${PROJECT_ROOT}/tmp-log/rag_server_addrs}"
START_PORT="${START_PORT:-7000}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_PATH="${RETRIEVER_PATH:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"
USE_FAISS_GPU="${USE_FAISS_GPU:-1}"

INDEX_FILE="${WIKI2018_WORK_DIR}/e5.index/e5_Flat.index"
CORPUS_FILE="${WIKI2018_WORK_DIR}/wiki_corpus.jsonl"
PAGES_FILE="${WIKI2018_WORK_DIR}/wiki_webpages.jsonl"

for f in "${INDEX_FILE}" "${CORPUS_FILE}" "${PAGES_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required file: ${f}"
    exit 1
  fi
done

mkdir -p "${RAG_SERVER_ADDR_DIR}"
find "${RAG_SERVER_ADDR_DIR}" -maxdepth 1 -type f -name "Host*_IP*.txt" -delete

FAISS_GPU_FLAG=()
if [[ "${USE_FAISS_GPU}" == "1" ]]; then
  FAISS_GPU_FLAG+=(--faiss_gpu)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

PIDS=()

cleanup() {
  echo ""
  echo "Stopping all RAG servers..."
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait
  echo "All servers stopped."
  exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting ${NUM_SERVERS} RAG server(s) on GPU(s) ${GPU_IDS}, ports ${START_PORT}–$((START_PORT + NUM_SERVERS - 1))"

if [[ "${NUM_SERVERS}" -eq 1 ]]; then
  exec python3 "${PROJECT_ROOT}/tools/local_retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --pages_path "${PAGES_FILE}" \
    --topk "${TOPK}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_PATH}" \
    "${FAISS_GPU_FLAG[@]}" \
    --port "${START_PORT}" \
    --save-address-to "${RAG_SERVER_ADDR_DIR}"
fi

for ((i = 0; i < NUM_SERVERS; i++)); do
  port=$((START_PORT + i))
  python3 "${PROJECT_ROOT}/tools/local_retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --pages_path "${PAGES_FILE}" \
    --topk "${TOPK}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_PATH}" \
    "${FAISS_GPU_FLAG[@]}" \
    --port "${port}" \
    --save-address-to "${RAG_SERVER_ADDR_DIR}" &
  PIDS+=($!)
  echo "Launched RAG server on port ${port} (PID $!)"
done

wait
