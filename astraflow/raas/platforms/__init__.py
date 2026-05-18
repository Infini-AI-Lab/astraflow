from __future__ import annotations

import torch
from transformers.utils.import_utils import is_torch_npu_available

import astraflow.raas.utils.logging as logging

from .cpu import CpuPlatform
from .cuda import CudaPlatform
from .npu import NPUPlatform
from .platform import Platform

logger = logging.getLogger("Platform init")

is_npu_available = is_torch_npu_available()


def _init_platform() -> Platform:
    """Detect hardware and return the appropriate Platform instance."""
    if torch.cuda.is_available():
        logger.info(f"Detected CUDA device: {torch.cuda.get_device_name()}")
        logger.info("Initializing CUDA platform.")
        return CudaPlatform()
    elif is_npu_available:
        from torch_npu.contrib import transfer_to_npu

        _ = transfer_to_npu.is_available()
        logger.info("Initializing NPU platform.")
        return NPUPlatform()
    else:
        logger.info("No accelerator detected. Initializing CPU platform.")
        return CpuPlatform()


class _LazyPlatform:
    """Defers platform detection until first attribute access."""

    def __init__(self):
        self._platform: Platform | None = None
        self._initialized = False

    def _ensure_initialized(self) -> Platform:
        if not self._initialized:
            self._platform = _init_platform()
            self._initialized = True
        assert self._platform is not None
        return self._platform

    def __getattr__(self, name: str):
        return getattr(self._ensure_initialized(), name)

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._ensure_initialized(), name, value)

    def __repr__(self) -> str:
        if self._initialized:
            return f"LazyPlatform({self._platform!r})"
        return "LazyPlatform(uninitialized)"


current_platform: Platform | _LazyPlatform = _LazyPlatform()

__all__ = ["Platform", "current_platform", "is_npu_available"]
