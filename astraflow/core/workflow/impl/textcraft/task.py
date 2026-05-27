"""Minimal Task dataclass for TextCraft.

Adapted from ``platoon.envs.base.Task`` but stripped to the fields used by
our recursive workflow. We do not need ``fork_strategy`` or ``SubTask``
because env.fork() takes a fresh Task directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A TextCraft crafting task.

    Attributes
    ----------
    goal:
        Natural-language goal string, e.g. ``"Craft the following items: 4x wooden_pickaxe"``.
    id:
        Stable identifier (e.g. ``"textcraft.train.42"``) — used as prompt_id.
    max_steps:
        Step budget for this task (root + all descendants share this pool).
    misc:
        Extra fields. We always populate:
          - ``target_items: dict[str, int]`` — what the agent must craft
          - ``initial_inventory: dict[str, int]`` — starting inventory
          - ``gold_trajectory: list[dict]`` — reference solve trace (eval only)
    """

    goal: str | None = None
    id: str | None = None
    max_steps: int | None = None
    misc: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(
            goal=d.get("goal"),
            id=d.get("id"),
            max_steps=d.get("max_steps"),
            misc=d.get("misc", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "id": self.id,
            "max_steps": self.max_steps,
            "misc": self.misc,
        }
