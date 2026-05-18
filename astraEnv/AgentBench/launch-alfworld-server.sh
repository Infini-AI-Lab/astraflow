#!/usr/bin/env bash
set -xeuo pipefail

python -m src.server.adapter_pool_controller alfworld-std     --config "configs/tasks/alfworld.yaml"     --port 5000     --startup-timeout 300     --num-adapters 32     --max-restarts 10     --log-dir "adapter_alfworld_logs"     --strategy "round-robin"     --restart-after-episodes 32     --container-runtime "podman"
