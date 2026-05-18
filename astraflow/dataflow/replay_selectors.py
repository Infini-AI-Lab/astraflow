"""Selection strategies for AstraFlow replay retrieval."""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

ReplaySelectionFn = Callable[[int, int], list[int]]


def select_latest(buffer_size: int, batch_size: int) -> list[int]:
    """Select the latest indices from replay storage."""
    count = min(batch_size, buffer_size)
    start = max(0, buffer_size - count)
    return list(range(start, buffer_size))


REPLAY_SELECTION_REGISTRY: dict[str, ReplaySelectionFn] = {
    "latest": select_latest,
}


def get_replay_selection(selection_name: str | None) -> ReplaySelectionFn | None:
    """Get a replay-selection function from the registry by name."""
    if selection_name is None:
        return None

    if selection_name not in REPLAY_SELECTION_REGISTRY:
        available = ", ".join(REPLAY_SELECTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown replay selection name '{selection_name}'. "
            f"Available selections: {available}"
        )

    return REPLAY_SELECTION_REGISTRY[selection_name]


__all__ = [
    "REPLAY_SELECTION_REGISTRY",
    "ReplaySelectionFn",
    "get_replay_selection",
    "select_latest",
]
