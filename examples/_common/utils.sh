# Shared helpers for AstraFlow v2 recipe launchers.
#
# Sourced by examples/**/scripts/run_*.sh to avoid duplicating boilerplate
# across recipes. Public functions:
#
#   astraflow_load_experiment_env  -- parse ${EXPERIMENT_CONFIG} and export
#                                     EXP_NAME=experiment.experiment_name
#                                     TRIAL_NAME=experiment.trial_name
#                                     (pre-set env vars take precedence).
#   astraflow_setup_env            -- apply NCCL+PYTORCH tweaks, WANDB
#                                     fallback, and set LOG_DIR (+ mkdir).
#                                     Model- and component-specific
#                                     settings (ports, URLs, trainer nproc,
#                                     banner) stay in each recipe so users
#                                     can see and tweak them directly.
#   astraflow_kill_stale           -- pre-launch sweep: kill leftover
#                                     processes and shared-memory from a
#                                     prior run.
#   astraflow_cleanup_trap         -- trap handler for EXIT/INT/TERM: kill
#                                     the process group and clean up the
#                                     same state.
#
# Usage in a launcher:
#
#   source "${REPO_ROOT}/examples/_common/utils.sh"
#   astraflow_load_experiment_env
#   astraflow_setup_env
#   trap astraflow_cleanup_trap EXIT INT TERM
#   astraflow_kill_stale
#
# Port list: by default covers ${WEIGHT_TRANSFER_HTTP_PORT_MODEL0} and 21000.
# Override by exporting ASTRAFLOW_CLEANUP_PORTS="p1 p2 ..." (space-separated)
# before sourcing or calling these functions -- e.g. multi-model recipes
# should include all WEIGHT_TRANSFER_HTTP_PORT_MODEL* values.

astraflow_load_experiment_env() {
    local cfg="${1:-${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is not set}}"
    local py_out
    py_out="$(python3 -c '
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))["experiment"]
print(cfg["experiment_name"])
print(cfg["trial_name"])
' "${cfg}")"
    export EXP_NAME="${EXP_NAME:-$(printf '%s\n' "${py_out}" | sed -n 1p)}"
    export TRIAL_NAME="${TRIAL_NAME:-$(printf '%s\n' "${py_out}" | sed -n 2p)}"
}

astraflow_setup_env() {
    # NCCL cuMem conflicts with SGLang's pre-allocated KV cache.
    export NCCL_CUMEM_ENABLE=0
    # Expandable segments reduce CUDA memory fragmentation during PPO backward.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" && -z "${WANDB_DISABLED:-}" ]]; then
        export WANDB_MODE="offline"
        export WANDB_DISABLED="true"
    fi

    export LOG_DIR="data-log/${EXP_NAME}/${TRIAL_NAME}"
    mkdir -p "${LOG_DIR}"
}

_astraflow_cleanup_ports() {
    if [[ -n "${ASTRAFLOW_CLEANUP_PORTS:-}" ]]; then
        echo "${ASTRAFLOW_CLEANUP_PORTS}"
    else
        echo "${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-} 21000"
    fi
}

_astraflow_kill_ports() {
    local ports
    ports="$(_astraflow_cleanup_ports)"
    for port in ${ports}; do
        [[ -z "${port}" ]] && continue
        lsof -i :"${port}" 2>/dev/null | awk 'NR>1{print $2}' | sort -u \
            | xargs kill -9 2>/dev/null || true
    done
}

_astraflow_rm_shm() {
    rm -rf /dev/shm/astraflow_* /dev/shm/_delta_timing.log 2>/dev/null || true
    rm -f /dev/shm/sem.mp-* 2>/dev/null || true
}

astraflow_kill_stale() {
    echo "Cleaning up stale processes..."
    pkill -9 -f astraflow.raas.server 2>/dev/null || true
    pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
    pkill -9 -f "sglang::scheduler" 2>/dev/null || true
    _astraflow_kill_ports
    pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
    pkill -9 -f "compile_worker" 2>/dev/null || true
    _astraflow_rm_shm
    sleep 2
}

astraflow_cleanup_trap() {
    trap - EXIT INT TERM
    echo "Shutting down..."
    kill -- -$$ 2>/dev/null || true
    pkill -9 -f astraflow.raas.server 2>/dev/null || true
    pkill -9 -f astraflow.raas.entrypoint 2>/dev/null || true
    _astraflow_kill_ports
    pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
    _astraflow_rm_shm
    wait 2>/dev/null
    exit 0
}

# Qwen3.5 (Gated-DeltaNet) on Hopper: the GDN backward must use fla's tilelang
# kernel (fla blocks its triton path on sm_90 as numerically wrong, fla#640),
# and the tilelang JIT needs a full CUDA toolkit (nvcc + CCCL headers; the
# pip-shipped nvcc has none). Called by the Qwen3.5 recipe launchers before
# starting the trainer. No-op on non-Hopper GPUs and when the user has already
# set the variables.
astraflow_setup_qwen35_hopper_env() {
    if nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q '^9\.'; then
        export FLA_TILELANG="${FLA_TILELANG:-1}"
        if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
            export CUDA_HOME=/usr/local/cuda
            export PATH="${CUDA_HOME}/bin:${PATH}"
        fi
        echo "Hopper (sm_90) detected: FLA_TILELANG=${FLA_TILELANG} CUDA_HOME=${CUDA_HOME:-unset}"
    fi
}
