#!/bin/bash
set -euo pipefail

# Launch RAG retrieval server in the foreground. Ctrl+C to stop.
# Run from the ASearcher directory.
#
# Usage:
#   bash scripts/launch_rag_server.sh <gpu_ids>
#   e.g. bash scripts/launch_rag_server.sh 6,7
#
# Optional env vars:
#   PORT             (default: 7000)
#   RETRIEVER_NAME   (default: e5)
#   RETRIEVER_PATH   (default: intfloat/e5-base-v2)
#   TOPK             (default: 3)
#   USE_FAISS_GPU    (default: 1, set 0 to disable)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gpu_ids>"
  echo "  e.g. $0 6,7"
  exit 1
fi

GPU_IDS="$1"

PORT="${PORT:-7000}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_PATH="${RETRIEVER_PATH:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"
USE_FAISS_GPU="${USE_FAISS_GPU:-1}"

WIKI2018_WORK_DIR="./data/wiki2018"
RAG_SERVER_ADDR_DIR="${RAG_SERVER_ADDR_DIR:-./tmp-log/rag_server_addrs}"

INDEX_FILE="${WIKI2018_WORK_DIR}/e5.index/e5_Flat.index"
CORPUS_FILE="${WIKI2018_WORK_DIR}/wiki_corpus.jsonl"
PAGES_FILE="${WIKI2018_WORK_DIR}/wiki_webpages.jsonl"

for f in "${INDEX_FILE}" "${CORPUS_FILE}" "${PAGES_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required file: ${f}"
    echo "Make sure you run this script from the ASearcher directory."
    exit 1
  fi
done

mkdir -p "${RAG_SERVER_ADDR_DIR}"

FAISS_GPU_FLAG=()
if [[ "${USE_FAISS_GPU}" == "1" ]]; then
  FAISS_GPU_FLAG+=(--faiss_gpu)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "Starting RAG server on GPU(s) ${GPU_IDS}, port ${PORT}"

exec python3 ./tools/local_retrieval_server.py \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --pages_path "${PAGES_FILE}" \
  --topk "${TOPK}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --retriever_model "${RETRIEVER_PATH}" \
  "${FAISS_GPU_FLAG[@]}" \
  --port "${PORT}" \
  --save-address-to "${RAG_SERVER_ADDR_DIR}"
