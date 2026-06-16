#!/usr/bin/env python
"""Remove the CUDA-only / GPU-stack pins from astraflow's pyproject [project]
dependencies so an editable install resolves against the ROCm base image's
pre-installed packages instead of pulling CUDA wheels from PyPI.

Stripped packages (provided by / managed in the ROCm base image):
  torch, torchaudio, torchvision, torchdata  -> ROCm torch from base
  torch_memory_saver                         -> installed separately (best-effort / shim)

Everything else (transformers==5.6.1, megatron-core, mbridge, the pure-python
utils) is kept and installed normally.

Idempotent: re-running on an already-pruned file is a no-op.

Usage: prune_pyproject.py [path]   (default: ./pyproject.toml)
"""
import re
import sys

STRIP = {
    "torch",
    "torchaudio",
    "torchvision",
    "torchdata",
    "torch_memory_saver",
    "torch-memory-saver",
    # Installed separately with --no-deps: megatron-core 0.13.1 declares an overly
    # conservative numpy<2.0.0 that conflicts with the ROCm base's numpy 2.2.6
    # (which the ROCm torch was built against). megatron-core/mbridge import fine
    # on numpy 2.x; their real deps (torch, transformers, accelerate, safetensors,
    # einops) are already in the base image.
    "megatron-core",
    "mbridge",
}


def dist_name(spec: str) -> str:
    # "torch==2.11.0" / "torch_memory_saver==0.0.9.post1" -> normalized base name
    name = re.split(r"[<>=!~\[ ]", spec.strip(), maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml"
    with open(path) as f:
        lines = f.readlines()

    out, removed, in_deps = [], [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("dependencies = ["):
            in_deps = True
            out.append(line)
            continue
        if in_deps:
            if stripped.startswith("]"):
                in_deps = False
                out.append(line)
                continue
            m = re.match(r'\s*"([^"]+)"', line)
            if m and dist_name(m.group(1)).replace("_", "-") in {
                s.replace("_", "-") for s in STRIP
            }:
                removed.append(m.group(1))
                continue
        out.append(line)

    with open(path, "w") as f:
        f.writelines(out)
    print(f"[prune_pyproject] removed from dependencies: {removed or 'nothing'}")


if __name__ == "__main__":
    main()
