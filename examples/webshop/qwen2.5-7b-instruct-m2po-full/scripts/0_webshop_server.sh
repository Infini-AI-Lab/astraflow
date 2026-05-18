#!/bin/bash
set -euo pipefail
# [0/4] Launch AgentBench Webshop Server
#
# Usage:
#   bash examples/webshop/qwen2.5-7b-instruct-m2po-full/scripts/0_webshop_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
# Export EXP_NAME and TRIAL_NAME from the experiment YAML.
astraflow_load_experiment_env

# If you change AGENTBENCH_PORT, also update task_server_url in yaml/experiment.yaml.
export AGENTBENCH_PORT="${AGENTBENCH_PORT:-5000}"
export AGENTBENCH_LOG_DIR="${AGENTBENCH_LOG_DIR:-${REPO_ROOT}/data-log/${EXP_NAME}/${TRIAL_NAME}}"
mkdir -p "${AGENTBENCH_LOG_DIR}"

echo "=== AgentBench Webshop Server ==="
echo "Port        : ${AGENTBENCH_PORT}"
echo "Log dir     : ${AGENTBENCH_LOG_DIR}"
echo "================================="

bash astraEnv/AgentBench/start-webshop-server.sh
