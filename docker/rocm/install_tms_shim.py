#!/usr/bin/env python
"""Install a no-op ``torch_memory_saver`` shim into the active site-packages.

astraflow imports ``torch_memory_saver`` at module load on the FSDP path
(astraflow/train_worker/engine/megatron_engine.py), but only *uses* it for the
Megatron CPU-offload path (pause/resume around weight transfer). On ROCm the real
extension (which hooks cudaMalloc) may not build; this shim satisfies the import
and the no-op offload so FSDP training runs. If you later need real offload on
ROCm, replace this with a working torch_memory_saver build.
"""
import os
import sysconfig

SHIM = '''"""No-op torch_memory_saver shim for ROCm (installed by astraflow Dockerfile.rocm)."""
from contextlib import contextmanager


class _TorchMemorySaver:
    """Minimal stand-in: offload/onload become no-ops; region() is a pass-through."""

    def __init__(self):
        self.hook_mode = None

    def pause(self, *a, **k):
        return None

    def resume(self, *a, **k):
        return None

    @contextmanager
    def region(self, *a, **k):
        yield

    def disable(self, *a, **k):
        return None


torch_memory_saver = _TorchMemorySaver()
TorchMemorySaver = _TorchMemorySaver


def configure_subprocess(*a, **k):
    from contextlib import nullcontext
    return nullcontext()


__version__ = "0.0.0+rocm-shim"
'''


def main() -> None:
    site = sysconfig.get_paths()["purelib"]
    target = os.path.join(site, "torch_memory_saver.py")
    with open(target, "w") as f:
        f.write(SHIM)
    print(f"[tms-shim] wrote no-op torch_memory_saver to {target}")


if __name__ == "__main__":
    main()
