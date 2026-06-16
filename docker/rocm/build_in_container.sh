#!/bin/bash
# Install astraflow + its ROCm-safe deps INSIDE the SGLang ROCm base container.
# This is the enroot/pyxis equivalent of the RUN steps in docker/Dockerfile.rocm,
# factored out so both build paths run identical logic.
#
# Expects:
#   - running inside lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x (venv at /opt/venv)
#   - ASTRAFLOW_SRC = path to the astraflow checkout (default: /workspace/astraflow)
set -euo pipefail

export VIRTUAL_ENV=/opt/venv
export PATH="/opt/venv/bin:${PATH}"
export PIP_NO_INPUT=1

SRC="${ASTRAFLOW_SRC:-/workspace/astraflow}"

# When SRC is a runtime mount (enroot/pyxis), copy it into the image rootfs so the
# editable install does not dangle once the mount goes away. For docker builds the
# source is already COPY'd into the rootfs, so leave ASTRAFLOW_BAKE unset.
if [[ -n "${ASTRAFLOW_BAKE:-}" && "${ASTRAFLOW_BAKE}" != "${SRC}" ]]; then
  echo "[build] baking source ${SRC} -> ${ASTRAFLOW_BAKE}"
  mkdir -p "${ASTRAFLOW_BAKE}"
  cp -a "${SRC}/." "${ASTRAFLOW_BAKE}/"
  SRC="${ASTRAFLOW_BAKE}"
fi
cd "${SRC}"

echo "[build] python: $(python -V)  pip: $(pip -V)"
echo "[build] base torch: $(python -c 'import torch;print(torch.__version__, torch.version.hip)')"

echo "[build] 1/4 generate ROCm constraints (protect base GPU stack)"
python docker/rocm/gen_constraints.py /tmp/rocm-constraints.txt
cat /tmp/rocm-constraints.txt

echo "[build] 2/4 prune CUDA-only pins from pyproject"
python docker/rocm/prune_pyproject.py pyproject.toml

echo "[build] 3/4 install astraflow + pure-python deps (no GPU extras)"
pip install --no-build-isolation -e . -c /tmp/rocm-constraints.txt

echo "[build] 3b/4 megatron-core + mbridge + torchdata (--no-deps)"
# megatron-core/mbridge: bypass their numpy<2 pin. torchdata: provides
# torchdata.stateful_dataloader (imported by astraflow.dataflow); --no-deps keeps
# the ROCm torch untouched.
pip install --no-deps megatron-core==0.13.1 mbridge==0.13.0 torchdata

echo "[build] 4/4 torch_memory_saver (real, else no-op shim)"
pip install torch_memory_saver==0.0.9.post1 -c /tmp/rocm-constraints.txt \
  || { echo "[tms] real package unavailable on ROCm — installing no-op shim";
       python docker/rocm/install_tms_shim.py; }

echo "[build] sanity import check"
python - <<'PY'
import torch, sglang, transformers, megatron.core, mbridge, torch_memory_saver
print("torch", torch.__version__, "hip", torch.version.hip)
print("sglang", sglang.__version__, "transformers", transformers.__version__)
import astraflow
from astraflow.train_worker.platforms import current_platform  # noqa: F401
print("astraflow import OK")
PY
echo "[build] DONE"
