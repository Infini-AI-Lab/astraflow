#!/bin/bash
set -euo pipefail

# Usage:
#   export WIKI2018_WORK_DIR=/path/to/wiki2018
#   export RAG_SERVER_ADDR_DIR=astraEnv/ASearcher/tmp-log/rag_server_addrs
#   bash astraEnv/ASearcher/scripts/launch_multi_local_server.sh <start_port> <num_servers>
#
# Optional env vars:
#   RETRIEVER_NAME (default: e5)
#   RETRIEVER_PATH (default: intfloat/e5-base-v2)
#   TOPK (default: 3)
#   USE_FAISS_GPU (default: 0, set 1 to enable --faiss_gpu)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <start_port> <num_servers>"
  exit 1
fi

START_PORT="$1"
NUM_SERVERS="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WIKI2018_WORK_DIR="${WIKI2018_WORK_DIR:-${PROJECT_ROOT}/data/wiki2018}"
RAG_SERVER_ADDR_DIR="${RAG_SERVER_ADDR_DIR:-${PROJECT_ROOT}/tmp-log/rag_server_addrs}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_PATH="${RETRIEVER_PATH:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"
USE_FAISS_GPU="${USE_FAISS_GPU:-1}"

if [[ -z "${WIKI2018_WORK_DIR}" ]]; then
  echo "WIKI2018_WORK_DIR is not set."
  echo "Example: export WIKI2018_WORK_DIR=/path/to/wiki2018"
  exit 1
fi

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

echo "Clearing previous server address files in ${RAG_SERVER_ADDR_DIR}"
find "${RAG_SERVER_ADDR_DIR}" -maxdepth 1 -type f -name "Host*_IP*.txt" -delete

FAISS_GPU_FLAG=()
if [[ "${USE_FAISS_GPU}" == "1" ]]; then
  FAISS_GPU_FLAG+=(--faiss_gpu)
fi

export CUDA_VISIBLE_DEVICES=0,1
# export USE_FAISS_GPU=1

echo "Starting ${NUM_SERVERS} RAG servers from port ${START_PORT}"
for ((i = 0; i < NUM_SERVERS; i++)); do
  port=$((START_PORT + i))
  log_file="${RAG_SERVER_ADDR_DIR}/rag_server_${port}.log"

  nohup python3 "${PROJECT_ROOT}/tools/local_retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --pages_path "${PAGES_FILE}" \
    --topk "${TOPK}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_PATH}" \
    "${FAISS_GPU_FLAG[@]}" \
    --port "${port}" \
    --save-address-to "${RAG_SERVER_ADDR_DIR}" \
    >"${log_file}" 2>&1 &

  echo "Launched RAG server on port ${port}, log: ${log_file}"
done

echo "Done. Address files:"
ls -1 "${RAG_SERVER_ADDR_DIR}"/Host*_IP*.txt 2>/dev/null || true
