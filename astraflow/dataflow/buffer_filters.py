"""Buffer filtering algorithms for AstraFlow rollout buffers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

BufferFilterFn = Callable[[dict[str, Any], dict[str, Any]], bool]


class KeepAllFilter:
    """Filter that keeps all examples (no filtering)."""

    def __call__(self, example: dict[str, Any], metadata: dict[str, Any]) -> bool:
        return True


class FilterZeroAdvFilter:
    """Filter that drops examples where ``zero_adv`` equals 1."""

    def __call__(self, example: dict[str, Any], metadata: dict[str, Any]) -> bool:
        del example
        return metadata.get("zero_adv") != 1


FILTER_REGISTRY: dict[str, type[BufferFilterFn]] = {
    "keep_all": KeepAllFilter,
    "filter_zero_adv": FilterZeroAdvFilter,
}


def get_filter(filter_name: str | None) -> BufferFilterFn | None:
    """Get a filter function from the registry by name."""
    if filter_name is None:
        return None

    if filter_name not in FILTER_REGISTRY:
        available = ", ".join(FILTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown filter name '{filter_name}'. Available filters: {available}"
        )

    return FILTER_REGISTRY[filter_name]()


__all__ = [
    "BufferFilterFn",
    "FILTER_REGISTRY",
    "FilterZeroAdvFilter",
    "KeepAllFilter",
    "get_filter",
]
