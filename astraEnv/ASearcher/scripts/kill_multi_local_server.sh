#!/bin/bash
set -euo pipefail

# Usage:
#   bash astraEnv/ASearcher/scripts/kill_multi_local_server.sh <start_port> <num_servers>
#
# This script only terminates processes listening on the target ports when the
# command line contains "local_retrieval_server.py".

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <start_port> <num_servers>"
  exit 1
fi

START_PORT="$1"
NUM_SERVERS="$2"

if ! [[ "${START_PORT}" =~ ^[0-9]+$ ]] || ! [[ "${NUM_SERVERS}" =~ ^[0-9]+$ ]]; then
  echo "Error: start_port and num_servers must be positive integers."
  exit 1
fi

if [[ "${NUM_SERVERS}" -le 0 ]]; then
  echo "Error: num_servers must be greater than 0."
  exit 1
fi

if ! command -v lsof >/dev/null 2>&1 && ! command -v ss >/dev/null 2>&1; then
  echo "Error: need either 'lsof' or 'ss' to find listening processes."
  exit 1
fi

find_listening_pids() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    return
  fi

  ss -ltnp "sport = :${port}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u || true
}

killed_count=0
not_found_count=0
skipped_count=0
failed_count=0

echo "Stopping RAG servers from port ${START_PORT}, count ${NUM_SERVERS}"
for ((i = 0; i < NUM_SERVERS; i++)); do
  port=$((START_PORT + i))
  mapfile -t pids < <(find_listening_pids "${port}")

  if [[ ${#pids[@]} -eq 0 ]]; then
    echo "[port ${port}] No listening process found"
    not_found_count=$((not_found_count + 1))
    continue
  fi

  for pid in "${pids[@]}"; do
    cmdline="$(ps -p "${pid}" -o args= 2>/dev/null || true)"

    if [[ -z "${cmdline}" ]]; then
      echo "[port ${port}] PID ${pid} disappeared before termination"
      not_found_count=$((not_found_count + 1))
      continue
    fi

    if [[ "${cmdline}" != *"local_retrieval_server.py"* ]]; then
      echo "[port ${port}] Skipped PID ${pid} (not local_retrieval_server.py)"
      skipped_count=$((skipped_count + 1))
      continue
    fi

    kill "${pid}" 2>/dev/null || true

    for _ in {1..10}; do
      if ! ps -p "${pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.2
    done

    if ps -p "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" 2>/dev/null || true
      sleep 0.2
    fi

    if ps -p "${pid}" >/dev/null 2>&1; then
      echo "[port ${port}] Failed to stop PID ${pid}"
      failed_count=$((failed_count + 1))
    else
      echo "[port ${port}] Stopped PID ${pid}"
      killed_count=$((killed_count + 1))
    fi
  done
done

echo "Summary: killed=${killed_count}, not_found=${not_found_count}, skipped=${skipped_count}, failed=${failed_count}"

if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
