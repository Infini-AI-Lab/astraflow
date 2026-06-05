"""Monkey-patch infrastructure for inference backends.

Submodules provide backend-specific patches:
- ``sglang`` — SGLang server patches (CLI args, rollout registration)
"""

import logging
import os
import types
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


class BasePatch:
    def __init__(self):
        self._patched_functions = {}

    def _mark_as_patched(self, func: Callable, identifier: str):
        marker = f"__patched_{identifier}__"
        setattr(func, marker, True)

    def _is_patched(self, func: Callable, identifier: str) -> bool:
        marker = f"__patched_{identifier}__"
        return hasattr(func, marker)

    def apply(self) -> bool:
        raise NotImplementedError


class PatchManager:
    def __init__(self):
        self.patches: List[BasePatch] = []

    def register(self, patch: BasePatch):
        self.patches.append(patch)
        return self

    def apply_all(self) -> Dict[str, bool]:
        results = {}
        for patch in self.patches:
            patch_name = patch.__class__.__name__
            try:
                success = patch.apply()
                results[patch_name] = success
                if success:
                    logger.info(f"Successfully applied patch: {patch_name}")
                else:
                    logger.warning(f"Patch {patch_name} did not apply (may be already applied or not needed)")
            except Exception as e:
                logger.error(f"Failed to apply patch {patch_name}: {e}")
                results[patch_name] = False
        return results


def _log_patch_results(results: Dict[str, bool]):
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    if successful:
        logger.info(f"Successfully applied patches: {', '.join(successful)}")
    if failed:
        logger.warning(f"Failed or skipped patches: {', '.join(failed)}")


def _env_enabled() -> bool:
    return os.getenv("ASTRAFLOW_AUTOPATCH", "false").lower() in ("true", "1")


def _validate_patch_results(results: Dict[str, bool], strict: bool) -> None:
    if all(results.values()):
        return
    failed = sorted([name for name, success in results.items() if not success])
    msg = (
        "SGLang autopatch strict mode is enabled, but some patches failed: "
        + ", ".join(failed)
    )
    if strict:
        logger.error(msg)
        raise RuntimeError(msg)
    logger.warning(msg)


def _run_sglang_patches(strict: bool) -> bool:
    from astraflow.raas.patch.sglang import (
        HttpServerPatch,
        ServerArgsPatch,
    )

    manager = PatchManager()
    manager.register(ServerArgsPatch())
    manager.register(HttpServerPatch())

    results = manager.apply_all()
    _log_patch_results(results)
    _validate_patch_results(results, strict=strict)
    return all(results.values())


try:
    from wrapt.importer import when_imported

    @when_imported("sglang")
    def _patch_sglang(module: types.ModuleType) -> None:
        if not _env_enabled():
            logger.debug("Disabled by ASTRAFLOW_AUTOPATCH")
            return

        logger.info("Auto-applying patches to sglang...")
        _run_sglang_patches(strict=False)

except ImportError:
    logger.warning("wrapt not installed, autopatch disabled")


def apply_patches(strict: bool = False):
    """Manually apply all SGLang patches."""
    return _run_sglang_patches(strict=strict)
