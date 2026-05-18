#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTBENCH_DIR="${AGENTBENCH_DIR:-$SCRIPT_DIR}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$AGENTBENCH_DIR/launch-webshop-server.sh}"
AGENTBENCH_PORT="${AGENTBENCH_PORT:-5000}"
AGENTBENCH_CONDA_ENV="${AGENTBENCH_CONDA_ENV:-agent-bench}"
AGENTBENCH_CONDA_SH="${AGENTBENCH_CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-600}"
WAIT_INTERVAL_SECS="${WAIT_INTERVAL_SECS:-2}"
NODE_NAME="$(hostname -s)"
AGENTBENCH_LOG_DIR="${AGENTBENCH_LOG_DIR:-$AGENTBENCH_DIR/data-log}"
AGENTBENCH_LOG_FILE="${AGENTBENCH_LOG_FILE:-$AGENTBENCH_LOG_DIR/agentbench_webshop_server_${NODE_NAME}.log}"
AGENTBENCH_PODMAN_SUFFIX="${AGENTBENCH_PODMAN_SUFFIX:-$NODE_NAME}"
if [[ -z "$AGENTBENCH_PODMAN_SUFFIX" ]]; then
    AGENTBENCH_PODMAN_SUFFIX="$(id -u)"
fi
AGENTBENCH_PODMAN_ROOT="${AGENTBENCH_PODMAN_ROOT:-/tmp/agentbench-podman-root-${AGENTBENCH_PODMAN_SUFFIX}}"
AGENTBENCH_PODMAN_RUNROOT="${AGENTBENCH_PODMAN_RUNROOT:-/tmp/agentbench-podman-runroot-${AGENTBENCH_PODMAN_SUFFIX}}"
AGENTBENCH_PODMAN_TMPDIR="${AGENTBENCH_PODMAN_TMPDIR:-/tmp/agentbench-podman-tmp-${AGENTBENCH_PODMAN_SUFFIX}}"
AGENTBENCH_PODMAN_STORAGE_OPT="${AGENTBENCH_PODMAN_STORAGE_OPT:-overlay.ignore_chown_errors=true}"
AGENTBENCH_MIN_FREE_GB="${AGENTBENCH_MIN_FREE_GB:-8}"
AGENTBENCH_WEBSHOP_IMAGE="${AGENTBENCH_WEBSHOP_IMAGE:-docker.io/longinyu/agentbench-webshop:latest}"

require_free_space() {
    local path="$1"
    local min_gb="$2"
    local available_kb required_kb

    available_kb="$(df -Pk "$path" | awk 'NR==2 {print $4}')"
    required_kb="$((min_gb * 1024 * 1024))"
    if (( available_kb < required_kb )); then
        echo "Insufficient disk for AgentBench Podman paths on $path." >&2
        echo "Available: $((available_kb / 1024 / 1024)) GiB, required: ${min_gb} GiB." >&2
        exit 1
    fi
}

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "Launcher not found: $LAUNCH_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$AGENTBENCH_CONDA_SH" ]]; then
    echo "Conda init script not found: $AGENTBENCH_CONDA_SH" >&2
    exit 1
fi

source "$AGENTBENCH_CONDA_SH"
set +u  # conda deactivate hooks may reference unbound variables (e.g. CUDAARCHS_BACKUP)
conda activate "$AGENTBENCH_CONDA_ENV"
set -u

cd "$AGENTBENCH_DIR"

if ! command -v podman >/dev/null 2>&1; then
    echo "podman not found in active environment: $AGENTBENCH_CONDA_ENV" >&2
    exit 1
fi

mkdir -p "$AGENTBENCH_PODMAN_ROOT" "$AGENTBENCH_PODMAN_RUNROOT" "$AGENTBENCH_PODMAN_TMPDIR"
require_free_space "$AGENTBENCH_PODMAN_TMPDIR" "$AGENTBENCH_MIN_FREE_GB"

export AGENTBENCH_PODMAN_ROOT
export AGENTBENCH_PODMAN_RUNROOT
export AGENTBENCH_PODMAN_TMPDIR
export AGENTBENCH_PODMAN_STORAGE_OPT
export TMPDIR="$AGENTBENCH_PODMAN_TMPDIR"
export TMP="$AGENTBENCH_PODMAN_TMPDIR"
export TEMP="$AGENTBENCH_PODMAN_TMPDIR"

echo "Pulling WebShop image: $AGENTBENCH_WEBSHOP_IMAGE"
podman \
    --root "$AGENTBENCH_PODMAN_ROOT" \
    --runroot "$AGENTBENCH_PODMAN_RUNROOT" \
    --tmpdir "$AGENTBENCH_PODMAN_TMPDIR" \
    --storage-opt "$AGENTBENCH_PODMAN_STORAGE_OPT" \
    pull "$AGENTBENCH_WEBSHOP_IMAGE"

if curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1:${AGENTBENCH_PORT}/api/health" >/dev/null 2>&1; then
    echo "AgentBench server already healthy on port ${AGENTBENCH_PORT}. Nothing to launch."
    exit 0
fi

mkdir -p "$AGENTBENCH_LOG_DIR"

echo "Starting AgentBench Webshop server..."
echo "Server log: $AGENTBENCH_LOG_FILE"
echo "Podman root:    $AGENTBENCH_PODMAN_ROOT"
echo "Podman runroot: $AGENTBENCH_PODMAN_RUNROOT"
echo "Podman tmpdir:  $AGENTBENCH_PODMAN_TMPDIR"
nohup bash "$LAUNCH_SCRIPT" >>"$AGENTBENCH_LOG_FILE" 2>&1 &
SERVER_PID=$!

echo "Waiting for server health endpoint on port ${AGENTBENCH_PORT}..."
START_TS="$(date +%s)"
while true; do
    if curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1:${AGENTBENCH_PORT}/api/health" >/dev/null 2>&1; then
        echo "AgentBench server is ready (pid=${SERVER_PID})."
        break
    fi

    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "Server process exited before becoming healthy (pid=${SERVER_PID})." >&2
        echo "Check log: $AGENTBENCH_LOG_FILE" >&2
        exit 1
    fi

    NOW_TS="$(date +%s)"
    if (( NOW_TS - START_TS >= WAIT_TIMEOUT_SECS )); then
        echo "Timed out after ${WAIT_TIMEOUT_SECS}s waiting for AgentBench on port ${AGENTBENCH_PORT}." >&2
        echo "Server process is still running (pid=${SERVER_PID})." >&2
        echo "Check log: $AGENTBENCH_LOG_FILE" >&2
        exit 1
    fi

    sleep "$WAIT_INTERVAL_SECS"
done
