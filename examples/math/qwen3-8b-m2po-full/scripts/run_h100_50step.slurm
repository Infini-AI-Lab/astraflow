#!/bin/bash
#SBATCH --job-name=astraflow_50step
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --exclude=liquid-gpu-014
#SBATCH --time=06:00:00
#SBATCH --output=/home/changyi/astraflow/logs/50step-%j.out
#SBATCH --error=/home/changyi/astraflow/logs/50step-%j.out

set -euo pipefail
set -x

mkdir -p /home/changyi/astraflow/logs

set +u
source /home/changyi/miniconda3/etc/profile.d/conda.sh
conda activate astraflow
set -u

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE=online
export MASTER_ADDR=127.0.0.1
export TRIAL_NAME="qwen3-8b-m2po-50step-h100"

if [ -d "/cm/shared/apps/cuda12.9/toolkit/12.9.1" ]; then
  export CUDA_HOME="/cm/shared/apps/cuda12.9/toolkit/12.9.1"
elif [ -d "/cm/shared/apps/cuda13.0/toolkit/13.0.2" ]; then
  export CUDA_HOME="/cm/shared/apps/cuda13.0/toolkit/13.0.2"
else
  export CUDA_HOME="${CONDA_PREFIX:-/usr/local/cuda}"
fi
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2

echo "NODE: $(hostname)"
nvidia-smi --query-gpu=name,memory.free --format=csv | head -3

cd /home/changyi/astraflow
# Pass 50-step + recover-off overrides via "$@" passthrough on launch_trainer.py
bash examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full.sh \
  total_train_steps=50 \
  recover.mode=disabled \
  evaluator.freq_steps=10
