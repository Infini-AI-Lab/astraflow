#!/bin/bash
# Build the AstraFlow ROCm container image as an enroot squashfs (.sqsh) on a
# Slurm cluster that runs enroot/pyxis (no docker daemon on the login node, and
# unprivileged overlay mounts are blocked there — so all container work goes
# through pyxis on a compute node).
#
# Two srun steps:
#   1. import the SGLang ROCm base (docker://) and save it           -> ${BASE_SQSH}
#   2. layer astraflow + deps into it and save the result            -> ${OUT_SQSH}
#      (runs docker/rocm/build_in_container.sh inside the container)
#
# --container-remap-root is required: the install writes into the image's
# root-owned /opt/venv and bakes the source into /opt/astraflow.
#
# Usage:
#   bash examples/_common/build_astraflow_rocm.sh
# Env:
#   REPO_ROOT        astraflow checkout (default: this repo)
#   IMAGE_DIR        where .sqsh files go (default: ${REPO_ROOT}/.images)
#   BASE_IMAGE       docker ref of base (default: lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x)
#   SLURM_PARTITION  partition (default: gpuworker)
#   BUILD_GRES       gres for the build allocation (default: gpu:1)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IMAGE_DIR="${IMAGE_DIR:-${REPO_ROOT}/.images}"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x}"
BASE_SQSH="${BASE_SQSH:-${IMAGE_DIR}/sglang-rocm-base.sqsh}"
OUT_SQSH="${OUT_SQSH:-${IMAGE_DIR}/astraflow-rocm.sqsh}"
PART="${SLURM_PARTITION:-gpuworker}"
GRES="${BUILD_GRES:-gpu:1}"
mkdir -p "${IMAGE_DIR}"

# ---- stage 1: import + save base ----
if [[ ! -f "${BASE_SQSH}" ]]; then
  echo "[build] (1/2) importing base ${BASE_IMAGE} -> ${BASE_SQSH}"
  srun --partition="${PART}" --nodes=1 --ntasks=1 --gres="${GRES}" \
    --time="${BUILD_TIME:-01:00:00}" \
    --container-image="docker://${BASE_IMAGE}" \
    --container-save="${BASE_SQSH}" \
    --container-remap-root \
    true
else
  echo "[build] (1/2) base already present: ${BASE_SQSH}"
fi

# ---- stage 2: layer astraflow ----
echo "[build] (2/2) layering astraflow deps -> ${OUT_SQSH}"
srun --partition="${PART}" --nodes=1 --ntasks=1 --gres="${GRES}" \
  --time="${BUILD_TIME:-01:00:00}" \
  --container-image="${BASE_SQSH}" \
  --container-save="${OUT_SQSH}" \
  --container-mounts="${REPO_ROOT}:/src:ro" \
  --container-workdir="/src" \
  --container-remap-root \
  bash -lc "ASTRAFLOW_SRC=/src ASTRAFLOW_BAKE=/opt/astraflow bash docker/rocm/build_in_container.sh"

echo "[build] DONE -> ${OUT_SQSH}"
echo "[build] run with: srun --container-image=${OUT_SQSH} ..."
