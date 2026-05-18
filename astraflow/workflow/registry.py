"""Workflow and reward registries with decorator-based registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

WORKFLOW_REGISTRY: dict[str, type] = {}
REWARD_REGISTRY: dict[str, Callable] = {}


def register_workflow(name: str):
    """Decorator to register a workflow class by name."""

    def decorator(cls):
        if name in WORKFLOW_REGISTRY:
            raise ValueError(
                f"Workflow {name!r} is already registered by {WORKFLOW_REGISTRY[name]}."
            )
        WORKFLOW_REGISTRY[name] = cls
        return cls

    return decorator


def register_reward(name: str):
    """Decorator to register a reward function by name."""

    def decorator(fn):
        if name in REWARD_REGISTRY:
            raise ValueError(
                f"Reward {name!r} is already registered by {REWARD_REGISTRY[name]}."
            )
        REWARD_REGISTRY[name] = fn
        return fn

    return decorator


def get_workflow(name: str) -> type:
    """Look up a registered workflow class by name."""
    if name not in WORKFLOW_REGISTRY:
        raise KeyError(
            f"Unknown workflow {name!r}. "
            f"Registered: {sorted(WORKFLOW_REGISTRY.keys())}"
        )
    return WORKFLOW_REGISTRY[name]


def get_reward(name: str) -> Callable:
    """Look up a registered reward function by name."""
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward {name!r}. "
            f"Registered: {sorted(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]
