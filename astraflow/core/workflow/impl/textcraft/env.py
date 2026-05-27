"""TextCraft environment — stateful, in-process, forkable.

Adapted from platoon's ``TextCraftCodeExecutor`` / ``TextCraftEnv`` but
stripped of the IPython sandbox and CodeAct abstractions. This is a plain
Python object the recursive_agent workflow constructs per episode and
dispatches actions against.

State:
  - ``inventory: dict[str, int]`` — mutable; shared by reference across
    parent/child forks.
  - ``recipe_db: RecipeDatabase`` — read-only, shared across forks.

Action methods return text observations. ``fork(child_task)`` returns a
new TextCraftEnv whose inventory is the SAME dict object as the parent
(matches platoon's `_share_inventory=True` pattern).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from astraflow.core.workflow.impl.textcraft.recipe_loader import RecipeDatabase
from astraflow.core.workflow.impl.textcraft.task import Task


DEFAULT_RECIPES_DIR = Path(__file__).parent / "recipes"

# Module-level singleton recipe DB; ~860 recipes, ~50MB resident.
# Loaded on first use, shared across all envs in the process.
_RECIPE_DB_CACHE: dict[str, RecipeDatabase] = {}


def get_default_recipe_db() -> RecipeDatabase:
    """Return the process-global RecipeDatabase, loading on first call."""
    key = str(DEFAULT_RECIPES_DIR)
    if key not in _RECIPE_DB_CACHE:
        _RECIPE_DB_CACHE[key] = RecipeDatabase(DEFAULT_RECIPES_DIR)
    return _RECIPE_DB_CACHE[key]


class TextCraftEnv:
    """Stateful crafting environment.

    Parameters
    ----------
    task:
        The Task this env serves. Reads ``task.misc["initial_inventory"]``
        if no explicit inventory is passed; root episodes typically pass
        nothing, sub-agents pass the parent's inventory dict by reference.
    recipe_db:
        Shared RecipeDatabase. ``None`` → use the process-global one.
    inventory:
        Pre-existing inventory dict. If non-None it is **aliased** (not
        copied) — that's the whole point of fork().
    """

    def __init__(
        self,
        task: Task,
        recipe_db: RecipeDatabase | None = None,
        inventory: dict[str, int] | None = None,
    ):
        self.task = task
        self.recipe_db = recipe_db if recipe_db is not None else get_default_recipe_db()
        if inventory is not None:
            # Alias — mutations visible to whoever else holds this dict.
            self.inventory = inventory
        else:
            # Fresh copy of the task's initial inventory.
            init = task.misc.get("initial_inventory", {}) if task.misc else {}
            self.inventory = dict(init)
        self.finished: bool = False
        self.finish_message: str | None = None
        # Subagent telemetry for delegation shaping. Reset per env (not
        # forked) — each agent tracks its own immediate children.
        self.subagent_launched: int = 0
        self.subagent_succeeded: float = 0.0

    # ------------------------------------------------------------------ fork

    def fork(self, child_task: Task) -> "TextCraftEnv":
        """Return a child env that shares this env's inventory by reference."""
        return TextCraftEnv(
            task=child_task,
            recipe_db=self.recipe_db,  # shared
            inventory=self.inventory,  # ALIASED — mutations propagate
        )

    # ------------------------------------------------------------------ actions

    def get_info(self, items: list[str]) -> str:
        """Return JSON-text recipe info for each requested item."""
        records = []
        for item in items:
            recipes = self.recipe_db.get_recipes_for_item(item)
            records.append({
                "item": item,
                "can_craft": self.recipe_db.can_craft(item),
                "is_base": self.recipe_db.is_base_item(item),
                "in_inventory": self.inventory.get(item, 0),
                "crafting_depth": self.recipe_db.get_crafting_depth(item),
                "recipes": [
                    {
                        "ingredients": dict(r.ingredients),
                        "result_count": r.result_count,
                    }
                    for r in recipes
                ],
            })
        return json.dumps(records, separators=(",", ":"))

    def view_inventory(self) -> str:
        """Return JSON-text snapshot of current inventory."""
        # Drop zero-counts for cleanliness.
        clean = {k: v for k, v in self.inventory.items() if v > 0}
        return json.dumps(clean, separators=(",", ":"))

    def craft(
        self, ingredients: dict[str, int], target: tuple[str, int] | list
    ) -> str:
        """Consume ingredients, add target item.

        Validates against the recipe DB:
          - target item must be craftable
          - ingredients must match a known recipe (modulo tag resolution)
          - target_count must be divisible by recipe.result_count
          - all ingredients present in sufficient quantity

        On success: mutates inventory in place, returns "OK: ..." text.
        On failure: returns "ERROR: ..." text. Inventory unchanged.
        """
        # Accept tuple or list (JSON gives us list).
        if isinstance(target, (list, tuple)) and len(target) == 2:
            target_item, target_count = target[0], int(target[1])
        else:
            return f"ERROR: target must be [item_name, count], got {target!r}"

        if target_count <= 0:
            return f"ERROR: target count must be positive, got {target_count}"

        recipes = self.recipe_db.get_recipes_for_item(target_item)
        if not recipes:
            return f"ERROR: no recipe for {target_item!r} (is_base={self.recipe_db.is_base_item(target_item)})"

        # Find a matching recipe. A recipe matches if its ingredient set
        # equals the requested ingredient set scaled by N (where N is the
        # number of times the recipe must run to produce target_count).
        # Tag ingredients accept any item that satisfies the tag.
        matched_recipe = None
        n_crafts = 0
        for recipe in recipes:
            if target_count % recipe.result_count != 0:
                continue
            n = target_count // recipe.result_count
            # Build the expected ingredient bag for n applications.
            expected = {k: v * n for k, v in recipe.ingredients.items()}
            # Resolve tags against provided ingredients.
            resolved = self._match_ingredients(expected, ingredients)
            if resolved is None:
                continue
            if resolved == dict(ingredients):
                matched_recipe = recipe
                n_crafts = n
                break

        if matched_recipe is None:
            recipe_summary = "; ".join(
                f"{r.result_count}x {target_item} <- {dict(r.ingredients)}"
                for r in recipes[:3]
            )
            return (
                f"ERROR: ingredients {dict(ingredients)} don't match any recipe "
                f"for {target_count}x {target_item}. Known recipes: {recipe_summary}"
            )

        # Validate inventory has enough of each ingredient.
        for ing, count in ingredients.items():
            if self.inventory.get(ing, 0) < count:
                return (
                    f"ERROR: need {count}x {ing}, have {self.inventory.get(ing, 0)}. "
                    f"Inventory: {self.view_inventory()}"
                )

        # Consume.
        for ing, count in ingredients.items():
            self.inventory[ing] = self.inventory.get(ing, 0) - count
            if self.inventory[ing] <= 0:
                del self.inventory[ing]

        # Produce.
        self.inventory[target_item] = self.inventory.get(target_item, 0) + target_count

        return f"OK: crafted {target_count}x {target_item}. Inventory: {self.view_inventory()}"

    def _match_ingredients(
        self, expected: dict[str, int], provided: dict[str, int]
    ) -> dict[str, int] | None:
        """Resolve tag-typed ingredients in ``expected`` against ``provided``.

        Returns the expected bag with tags rewritten to whichever concrete
        item from ``provided`` satisfies them. Returns None if any tag
        cannot be satisfied or counts don't line up.
        """
        resolved: dict[str, int] = {}
        provided_remaining = dict(provided)
        # First copy over any non-tag expected items.
        for k, v in expected.items():
            if not k.startswith("tag:"):
                resolved[k] = v
                provided_remaining[k] = provided_remaining.get(k, 0) - v
                if provided_remaining[k] < 0:
                    return None
                if provided_remaining[k] == 0:
                    del provided_remaining[k]
        # Now resolve tags by consuming from provided_remaining.
        for k, v in expected.items():
            if not k.startswith("tag:"):
                continue
            tag_name = k[len("tag:") :]
            satisfying = set(self.recipe_db.get_items_for_tag(tag_name))
            # Find any provided item that satisfies the tag with enough count.
            picked = None
            for prov_item, prov_count in provided_remaining.items():
                if prov_item in satisfying and prov_count >= v:
                    picked = prov_item
                    break
            if picked is None:
                return None
            resolved[picked] = resolved.get(picked, 0) + v
            provided_remaining[picked] -= v
            if provided_remaining[picked] == 0:
                del provided_remaining[picked]
        return resolved

    # ------------------------------------------------------------------ finish

    def finish(self, message: str) -> None:
        self.finished = True
        self.finish_message = message

    # ------------------------------------------------------------------ eval

    def evaluate(self) -> tuple[float, dict[str, Any]]:
        """Return (score, info_dict) for reward computation.

        Score = 1.0 iff every target_item is present in inventory at >=
        the requested count. Else 0.0 (binary, matches platoon).
        """
        targets: dict[str, int] = self.task.misc.get("target_items", {}) if self.task.misc else {}
        if not targets:
            return 0.0, {"reason": "no target_items"}

        satisfied = 0
        for item, count in targets.items():
            if self.inventory.get(item, 0) >= count:
                satisfied += 1

        score = 1.0 if satisfied == len(targets) else 0.0
        info = {
            "satisfied": satisfied,
            "total_targets": len(targets),
            "subagent_launched": self.subagent_launched,
            "subagent_succeeded": self.subagent_succeeded,
        }
        return score, info
