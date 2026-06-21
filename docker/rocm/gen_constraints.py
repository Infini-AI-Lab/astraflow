#!/usr/bin/env python
"""Emit a pip constraints file pinning the ROCm-critical packages to the versions
already installed in the base image.

These are the packages we must NOT let pip touch while it resolves astraflow's
dependency tree (a stray upgrade pulls a CUDA wheel and breaks the GPU stack).
We read the installed versions at build time rather than hardcoding the long
ROCm local-version strings (e.g. ``2.9.1+rocm7.2.0.lw.git...``).

transformers is deliberately excluded so pip is free to bump 5.6.0 -> 5.6.1.

Usage: gen_constraints.py [out_path]   (default: ./rocm-constraints.txt)
"""
import sys
from importlib import metadata

# Packages provided by the ROCm base image that must stay exactly as shipped.
PROTECT = [
    "torch",
    "torchaudio",
    "torchvision",
    "torchao",
    "triton",
    "pytorch-triton-rocm",
    "sglang",
    "sgl-kernel",
    "sglang-router",
    "numpy",
]


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "rocm-constraints.txt"
    lines = []
    for name in PROTECT:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            # Not present in this base image — nothing to protect.
            continue
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[gen_constraints] wrote {len(lines)} pins to {out}")


if __name__ == "__main__":
    main()
