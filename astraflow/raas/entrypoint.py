"""SGLang launch entrypoint with inference-engine patches pre-applied."""
import os
import sys


def _norm_fallback_for_non_hopper():
    """Force flashinfer's CUDA-JIT RMSNorm on non-Hopper GPUs.

    flashinfer 0.6.x routes RMSNorm through a CuTe-DSL kernel that has no
    Ada/Ampere implementation and JITs into an incompatible nvidia-cutlass-dsl
    (crashing with a GPUModuleOp TypeError on sm_89/sm_80). FLASHINFER_USE_CUDA_NORM=1
    selects its CUDA-JIT norm instead. Hopper (sm_90+) ships prebuilt cubins and is
    unaffected. Must run before sglang/flashinfer is imported; uses NVML so it does
    not create a CUDA context in this launcher process.
    """
    if os.environ.get("FLASHINFER_USE_CUDA_NORM"):
        return
    try:
        import pynvml

        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        idx = int(vis.split(",")[0]) if vis and vis.split(",")[0].isdigit() else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        major, _ = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        pynvml.nvmlShutdown()
        if major < 9:
            os.environ["FLASHINFER_USE_CUDA_NORM"] = "1"
    except Exception:
        pass


# Must run before sglang/flashinfer import (apply_patches may import sglang).
_norm_fallback_for_non_hopper()

from astraflow.raas.patch import apply_patches

# Apply patches at module level so they also run in spawned child processes.
apply_patches()

if __name__ == '__main__':
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
