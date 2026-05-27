"""Task generation for TextCraft crafting tasks."""

import argparse
import json
import pathlib
import random
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from astraflow.core.workflow.impl.textcraft.task import Task

from .recipe_loader import RecipeDatabase


def get_tag_satisfying_items(recipe_db: RecipeDatabase, tag: str) -> List[str]:
    """
    Get items that satisfy a tag requirement.
    Uses the recipe database to find items that satisfy the tag.
    """
    # Remove "tag:" prefix if present
    tag_name = tag.replace("tag:", "")

    # Get items from the recipe database's tag mapping
    items = recipe_db.get_items_for_tag(tag_name)

    if items:
        return items

    # Fallback: if no items found, try pattern matching
    tag_lower = tag_name.lower()

    # Pattern matching for common cases
    if "planks" in tag_lower:
        # Find all planks items
        planks_items = [item for item in recipe_db.all_items if item.endswith("_planks")]
        if planks_items:
            return planks_items

    if ("log" in tag_lower or "stem" in tag_lower) and "planks" not in tag_lower:
        # Find all log/stem items
        log_items = [
            item
            for item in recipe_db.all_items
            if "_log" in item or "_stem" in item or item.endswith("_log") or item.endswith("_stem")
        ]
        if log_items:
            return log_items

    if "wood" in tag_lower and "planks" not in tag_lower:
        # Find all wood items
        wood_items = [item for item in recipe_db.all_items if "_wood" in item or item.endswith("_wood")]
        if wood_items:
            return wood_items

    # Return empty list if nothing found
    return []


def extract_base_materials_needed(
    recipe_db: RecipeDatabase,
    target_items: Dict[str, int],
    initial_inventory: Dict[str, int],
) -> Dict[str, int]:
    """
    Extract all base materials (non-craftable items) needed to craft target items.
    Returns a dictionary mapping base item names to required counts.
    """
    needed_base_items: Dict[str, int] = {}
    inventory = initial_inventory.copy()

    def collect_base_materials(item: str, count: int, visited: Optional[Set[str]] = None):
        """Recursively collect base materials needed."""
        if visited is None:
            visited = set()

        if item in visited:
            return
        visited.add(item)

        # If we already have enough in inventory, no need to collect
        if inventory.get(item, 0) >= count:
            return

        # If it's a base item, add to needed
        if recipe_db.is_base_item(item):
            needed = count - inventory.get(item, 0)
            if needed > 0:
                needed_base_items[item] = needed_base_items.get(item, 0) + needed
            return

        # If it's craftable, collect ingredients
        if recipe_db.can_craft(item):
            recipes = recipe_db.get_recipes_for_item(item)
            if recipes:
                recipe = recipes[0]
                needed_crafts = (count - inventory.get(item, 0) + recipe.result_count - 1) // recipe.result_count

                for ing_item, ing_count in recipe.ingredients.items():
                    if ing_item.startswith("tag:"):
                        # For tags, find items that satisfy the tag and collect their base materials
                        tag_name = ing_item.replace("tag:", "")
                        satisfying_items = get_tag_satisfying_items(recipe_db, tag_name)
                        if satisfying_items:
                            # Use the first satisfying item as representative
                            # Collect base materials for that item
                            collect_base_materials(satisfying_items[0], needed_crafts * ing_count, visited.copy())
                        # If no satisfying items found, skip (will need manual handling)
                        continue
                    collect_base_materials(ing_item, needed_crafts * ing_count, visited.copy())

    for item, count in target_items.items():
        collect_base_materials(item, count)

    return needed_base_items


def solve_crafting_task(
    recipe_db: RecipeDatabase,
    target_items: Dict[str, int],
    initial_inventory: Dict[str, int],
    max_depth: int = 20,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, int]]]:
    """
    Solve a crafting task by finding a valid sequence of crafting steps.

    Returns:
        Tuple of (trajectory, required_base_materials) or None if no solution found.
        - trajectory: List of crafting steps
        - required_base_materials: Dict of base items needed
        Each step is: {"action": "craft", "target": ("item", count), "ingredients": {...}}
    """
    # Simulate inventory
    inventory = initial_inventory.copy()
    trajectory = []
    visited_steps: Set[Tuple[str, int]] = set()  # Track (item, count) we've tried to craft

    def can_craft_item(item: str, count: int) -> bool:
        """Check if we can craft an item with current inventory."""
        if inventory.get(item, 0) >= count:
            return True

        recipes = recipe_db.get_recipes_for_item(item)
        if not recipes:
            return False

        # Try first recipe
        recipe = recipes[0]

        # Check if we have all ingredients
        for ing_item, ing_count in recipe.ingredients.items():
            if ing_item.startswith("tag:"):
                # For tags, we'd need to check if any matching item exists
                # For now, skip tag validation (could be improved)
                continue

            # Calculate how many times we need to craft
            needed = count
            available = inventory.get(ing_item, 0)

            # If we need more than available, check if we can craft the ingredient
            if available < needed * ing_count:
                # Try to craft the ingredient recursively
                ingredient_needed = (needed * ing_count) - available
                # Calculate how many of the ingredient we need to craft
                crafts_needed = (ingredient_needed + recipe.result_count - 1) // recipe.result_count
                if not can_craft_item(ing_item, crafts_needed):
                    return False

        return True

    def craft_item(item: str, count: int) -> bool:
        """Attempt to craft an item, recursively crafting ingredients if needed."""
        # Check if we already have enough
        if inventory.get(item, 0) >= count:
            return True

        # Check if we're currently crafting this (prevent infinite recursion)
        step_key = (item, count)
        if step_key in visited_steps:
            return False
        visited_steps.add(step_key)

        recipes = recipe_db.get_recipes_for_item(item)
        if not recipes:
            return False

        recipe = recipes[0]

        # Calculate how many times we need to craft
        needed = count - inventory.get(item, 0)
        crafts_needed = (needed + recipe.result_count - 1) // recipe.result_count

        # Calculate total crafts needed upfront
        total_crafts_needed = crafts_needed

        # Calculate total ingredient needs including tag requirements
        # We need to look ahead to see what tags will consume
        ingredient_needs: Dict[str, int] = {}
        for ing_item, ing_count_per_craft in recipe.ingredients.items():
            total_needed_for_this_ing = total_crafts_needed * ing_count_per_craft

            if ing_item.startswith("tag:"):
                # For tags, find what items satisfy it and account for their consumption
                tag_name = ing_item.replace("tag:", "")
                satisfying_items = recipe_db.get_items_for_tag(tag_name)

                # We'll need to consume `total_needed_for_this_ing` of SOME satisfying item
                # Add this to our needs (we'll resolve which specific item later)
                # For now, mark that we need this tag satisfied
                ingredient_needs[f"tag:{tag_name}"] = total_needed_for_this_ing
            else:
                # Concrete item
                ingredient_needs[ing_item] = ingredient_needs.get(ing_item, 0) + total_needed_for_this_ing

        # First, ensure we have all concrete ingredients
        # Track what we have as we craft to detect if recursive crafts consume things
        inventory_snapshot = {}

        for ing_item, total_needed in ingredient_needs.items():
            if ing_item.startswith("tag:"):
                # Handle tags separately after concrete items
                continue

            available = inventory.get(ing_item, 0)

            if available < total_needed:
                if recipe_db.can_craft(ing_item):
                    # Take snapshot of inventory BEFORE this craft
                    before_craft = inventory.copy()

                    # Craft enough to have at least `total_needed` items total
                    if not craft_item(ing_item, total_needed):
                        return False

                    # Update our snapshot for this ingredient
                    inventory_snapshot[ing_item] = inventory.get(ing_item, 0)

                    # After crafting this ingredient, check if inventory of OTHER concrete ingredients decreased
                    # (meaning the recursive craft consumed them via tags)
                    for other_ing, other_needed in ingredient_needs.items():
                        if other_ing.startswith("tag:") or other_ing == ing_item:
                            continue

                        # Check if this ingredient was consumed relative to our snapshot
                        # Use the most recent snapshot value, or the value before this craft
                        before = inventory_snapshot.get(other_ing, before_craft.get(other_ing, 0))
                        now = inventory.get(other_ing, 0)
                        if now < before:
                            # Some was consumed - need to craft more to restore it
                            if recipe_db.can_craft(other_ing):
                                if not craft_item(other_ing, other_needed):
                                    return False
                                # Update the snapshot
                                inventory_snapshot[other_ing] = inventory.get(other_ing, 0)
                else:
                    # Base item, can't craft it
                    return False
            else:
                # Already have enough, record in snapshot
                inventory_snapshot[ing_item] = inventory.get(ing_item, 0)

        # Now handle tag requirements - craft fresh items specifically for tags
        # to avoid conflicts with concrete ingredient needs
        for ing_item, total_needed in ingredient_needs.items():
            if not ing_item.startswith("tag:"):
                continue

            tag_name = ing_item.replace("tag:", "")
            satisfying_items = recipe_db.get_items_for_tag(tag_name)

            # Check if we have enough of ANY satisfying item that's NOT also needed as a concrete ingredient
            # Prefer using items that aren't explicitly required by this recipe
            concrete_needs = set(k for k in ingredient_needs.keys() if not k.startswith("tag:"))

            # Available items that don't conflict with concrete needs
            non_conflicting_available = {
                item: inventory.get(item, 0)
                for item in satisfying_items
                if item not in concrete_needs and inventory.get(item, 0) > 0
            }
            total_non_conflicting = sum(non_conflicting_available.values())

            if total_non_conflicting >= total_needed:
                # We have enough non-conflicting items, we're good
                continue

            # Need to craft more satisfying items
            # Choose one to craft (prefer craftable items that aren't in concrete_needs)
            craftable_satisfying = [
                item for item in satisfying_items if recipe_db.can_craft(item) and item not in concrete_needs
            ]

            if not craftable_satisfying:
                # Fall back to any craftable satisfying item
                craftable_satisfying = [item for item in satisfying_items if recipe_db.can_craft(item)]

            if craftable_satisfying:
                # Try each craftable item until one succeeds
                # (some may fail due to missing base materials)
                crafted_successfully = False
                for chosen in craftable_satisfying:
                    needed_more = total_needed - total_non_conflicting
                    if craft_item(chosen, needed_more):
                        crafted_successfully = True
                        break

                if not crafted_successfully:
                    # None of the craftable items worked - check if we have any satisfying items in inventory
                    total_available = sum(inventory.get(item, 0) for item in satisfying_items)
                    if total_available < total_needed:
                        return False
            else:
                # Check if we have ANY satisfying items (even if conflicting)
                total_available = sum(inventory.get(item, 0) for item in satisfying_items)
                if total_available < total_needed:
                    # No craftable satisfying items and not enough in inventory - unsolvable
                    return False

        # Now craft the item the required number of times
        # Consume ingredients and track what we actually consume
        step_ingredients = {}
        for ing_item, ing_count_per_craft in recipe.ingredients.items():
            total_needed = total_crafts_needed * ing_count_per_craft

            if ing_item.startswith("tag:"):
                # Resolve tag to a concrete item
                tag_name = ing_item.replace("tag:", "")
                satisfying_items = recipe_db.get_items_for_tag(tag_name)

                # Choose which item to consume: prefer items we have enough of and aren't craftable
                # (to avoid consuming intermediate crafted items when possible)
                available_items = [
                    (sat_item, inventory.get(sat_item, 0))
                    for sat_item in satisfying_items
                    if inventory.get(sat_item, 0) >= total_needed
                ]

                if available_items:
                    # Sort by: not craftable first, then by quantity (more is better)
                    available_items.sort(key=lambda x: (recipe_db.can_craft(x[0]), -x[1]))
                    chosen_item = available_items[0][0]

                    # Consume the chosen item
                    inventory[chosen_item] -= total_needed
                    if inventory[chosen_item] <= 0:
                        del inventory[chosen_item]

                    # Record the concrete item in trajectory
                    step_ingredients[chosen_item] = total_needed
                else:
                    # This shouldn't happen if we ensured ingredients are available
                    # But keep the tag as fallback
                    step_ingredients[f"tag:{tag_name}"] = total_needed
            else:
                # Concrete item - consume it
                total_consumed = total_needed
                inventory[ing_item] = inventory.get(ing_item, 0) - total_consumed
                if inventory[ing_item] <= 0:
                    del inventory[ing_item]
                step_ingredients[ing_item] = total_consumed

        # Add crafted items
        crafted_count = total_crafts_needed * recipe.result_count
        inventory[item] = inventory.get(item, 0) + crafted_count

        trajectory.append(
            {
                "action": "craft",
                "target": (item, total_crafts_needed),
                "ingredients": step_ingredients,
                "result_count": crafted_count,
            }
        )

        # Remove from visited_steps to allow re-crafting if this item gets consumed by tags
        visited_steps.discard(step_key)

        return True

    # Calculate total needs accounting for dependencies between targets
    # If target A is an ingredient for target B, we need extra of A
    total_needs = target_items.copy()
    for target_item, target_count in target_items.items():
        recipes = recipe_db.get_recipes_for_item(target_item)
        if recipes:
            recipe = recipes[0]
            crafts_needed = (target_count + recipe.result_count - 1) // recipe.result_count

            for ing_item, ing_count_per_craft in recipe.ingredients.items():
                if ing_item.startswith("tag:"):
                    continue

                # If this ingredient is also a target, we need more of it
                if ing_item in total_needs:
                    total_needs[ing_item] = total_needs.get(ing_item, 0) + (crafts_needed * ing_count_per_craft)

    # Try to craft each item according to total needs
    for item, count in total_needs.items():
        if not craft_item(item, count):
            return None  # Failed to find solution

    # Extract required base materials
    required_base_materials = extract_base_materials_needed(recipe_db, target_items, initial_inventory)

    return trajectory, required_base_materials


def create_textcraft_datasets(
    seed: int = 42,
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
    recipes_dir: Optional[pathlib.Path] = None,
    min_depth: int = 2,
    max_depth: int = 5,
    min_target_items: int = 1,
    max_target_items: int = 3,
) -> Tuple[List[Task], List[Task]]:
    """
    Create train and validation datasets for TextCraft tasks.

    Args:
        seed: Random seed
        num_samples_train: Number of training samples
        num_samples_val: Number of validation samples
        recipes_dir: Path to recipes directory
        min_depth: Minimum crafting depth for hierarchical tasks
        max_depth: Maximum crafting depth
        min_target_items: Minimum number of target items per task
        max_target_items: Maximum number of target items per task

    Returns:
        Tuple of (train_tasks, val_tasks)
    """
    random.seed(seed)

    if recipes_dir is None:
        recipes_dir = pathlib.Path(__file__).parent / "recipes"

    recipe_db = RecipeDatabase(recipes_dir)

    # Find hierarchical recipes
    hierarchical_items = recipe_db.find_hierarchical_recipes(min_depth=min_depth, max_depth=max_depth)

    if not hierarchical_items:
        raise ValueError(f"No hierarchical recipes found with depth {min_depth}-{max_depth}")

    train_tasks: List[Task] = []
    val_tasks: List[Task] = []

    # Track seen tasks to prevent duplicates
    # Use a tuple of (sorted target_items, sorted initial_inventory) as the key
    seen_tasks: Set[Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...]]] = set()

    # Generate tasks
    total_needed = num_samples_train + num_samples_val
    train_count = 0
    val_count = 0

    attempts = 0
    max_attempts = total_needed * 100  # Increased attempts to account for deduplication

    while (train_count < num_samples_train or val_count < num_samples_val) and attempts < max_attempts:
        attempts += 1

        # Select target item(s)
        num_targets = random.randint(min_target_items, max_target_items)
        target_items_list = random.sample(hierarchical_items, min(num_targets, len(hierarchical_items)))

        # Create target dictionary with random counts (1-3)
        target_items: Dict[str, int] = {}
        for item in target_items_list:
            target_items[item] = random.randint(1, 3)

        # First, extract what base materials are needed (without solving)
        required_base_materials = extract_base_materials_needed(recipe_db, target_items, {})
        if not required_base_materials:
            continue  # No base materials found, skip this task

        # Now generate initial inventory that includes all required base materials
        # plus some extra random base items
        base_items = list(recipe_db.base_items)
        if not base_items:
            continue

        initial_inventory: Dict[str, int] = {}

        # Add required base materials (with some buffer)
        for item, count in required_base_materials.items():
            # Add required amount plus some buffer (10-50% extra)
            buffer = random.randint(0, max(1, count // 2))
            initial_inventory[item] = count + buffer

        # Solve with initial inventory to get trajectory and identify tag requirements
        solve_result = solve_crafting_task(recipe_db, target_items, initial_inventory)
        if solve_result is None:
            continue  # Task not solvable with these base materials

        gold_trajectory, _ = solve_result

        # Check gold trajectory for tag-based ingredients and ensure they're satisfied
        tag_requirements: Dict[str, int] = {}
        for step in gold_trajectory:
            for ing_item, ing_count in step.get("ingredients", {}).items():
                if ing_item.startswith("tag:"):
                    tag_name = ing_item.replace("tag:", "")
                    tag_requirements[tag_name] = max(tag_requirements.get(tag_name, 0), ing_count)

        # Add items that satisfy tag requirements
        for tag_name, ing_count in tag_requirements.items():
            satisfying_items = get_tag_satisfying_items(recipe_db, tag_name)
            if satisfying_items:
                # Pick a random satisfying item (or use first)
                representative = random.choice(satisfying_items)
                # Check if we already have a satisfying item in inventory
                has_satisfying = any(item in initial_inventory for item in satisfying_items)
                if not has_satisfying:
                    # Add enough to satisfy the tag requirement
                    initial_inventory[representative] = ing_count + random.randint(0, max(1, ing_count // 2))
                else:
                    # Ensure we have enough of the satisfying item
                    for item in satisfying_items:
                        if item in initial_inventory:
                            if initial_inventory[item] < ing_count:
                                initial_inventory[item] = ing_count + random.randint(0, max(1, ing_count // 2))
                            break

        # Add some additional random base items
        num_extra_items = random.randint(2, 5)
        for _ in range(num_extra_items):
            item = random.choice(base_items)
            if item not in initial_inventory:  # Don't duplicate required items
                initial_inventory[item] = random.randint(1, 5)

        # Create a unique key for this task (sorted tuples for deterministic hashing)
        target_key = tuple(sorted(target_items.items()))
        inventory_key = tuple(sorted(initial_inventory.items()))
        task_key = (target_key, inventory_key)

        # Skip if we've seen this exact task before
        if task_key in seen_tasks:
            continue

        # Mark as seen
        seen_tasks.add(task_key)

        # Re-solve to get the final trajectory with complete inventory (including tag items)
        final_solve_result = solve_crafting_task(recipe_db, target_items, initial_inventory)
        if final_solve_result is None:
            continue  # Task not solvable with final inventory (shouldn't happen)

        # Use the trajectory from solving with the complete initial inventory
        actual_trajectory, _ = final_solve_result

        # Create task
        target_str = ", ".join([f"{count}x {item}" for item, count in target_items.items()])
        goal = f"Craft the following items: {target_str}"

        task = Task(
            goal=goal,
            max_steps=50,  # Allow enough steps for hierarchical crafting
            misc={
                "target_items": target_items,
                "initial_inventory": initial_inventory,
                "gold_trajectory": actual_trajectory,  # Use trajectory from actual inventory
            },
            id=f"textcraft.train.{train_count}" if train_count < num_samples_train else f"textcraft.val.{val_count}",
        )

        # Split deterministically based on task content (using target_items only for consistency)
        task_hash = hash(str(target_key))
        is_val = (task_hash % 10) < (num_samples_val * 10 / total_needed)

        if is_val and val_count < num_samples_val:
            val_tasks.append(task)
            val_count += 1
        elif train_count < num_samples_train:
            train_tasks.append(task)
            train_count += 1

    if train_count < num_samples_train or val_count < num_samples_val:
        raise RuntimeError(
            f"Failed to generate enough unique tasks. Generated {train_count}/{num_samples_train} train and "
            f"{val_count}/{num_samples_val} val tasks after {attempts} attempts. "
            f"Consider reducing the number of samples or increasing max_attempts."
        )

    return train_tasks, val_tasks


TRAIN_DATA: Optional[List[str]] = None
VAL_DATA: Optional[List[str]] = None
TASKS: Dict[str, Task] = {}


def get_task_ids(
    split: Literal["train", "val"],
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
) -> List[str]:
    """Get task IDs for a split."""
    if split == "train":
        return [f"textcraft.train.{i}" for i in range(num_samples_train)]
    elif split == "val":
        return [f"textcraft.val.{i}" for i in range(num_samples_val)]
    else:
        raise ValueError(f"Invalid split: {split}")


def load_task_from_disk(id: str) -> Task:
    """Load a task from disk."""
    parent = pathlib.Path(__file__).parent

    if id.startswith("textcraft.train."):
        global TRAIN_DATA
        if TRAIN_DATA is None:
            file = parent / "textcraft_train.jsonl"
            if file.exists():
                TRAIN_DATA = file.read_text().splitlines()
            else:
                raise FileNotFoundError(f"Training data file not found: {file}")
        idx = int(id.split(".")[2])
        if idx >= len(TRAIN_DATA):
            raise IndexError(f"Task index {idx} out of range for training data")
        return Task.from_dict(json.loads(TRAIN_DATA[idx]))

    elif id.startswith("textcraft.val."):
        global VAL_DATA
        if VAL_DATA is None:
            file = parent / "textcraft_val.jsonl"
            if file.exists():
                VAL_DATA = file.read_text().splitlines()
            else:
                raise FileNotFoundError(f"Validation data file not found: {file}")
        idx = int(id.split(".")[2])
        if idx >= len(VAL_DATA):
            raise IndexError(f"Task index {idx} out of range for validation data")
        return Task.from_dict(json.loads(VAL_DATA[idx]))

    else:
        raise ValueError(f"Invalid task id: {id}")


def get_task(id: str) -> Task:
    """Get a task by ID (with caching)."""
    global TASKS
    if id in TASKS:
        return TASKS[id]
    task = load_task_from_disk(id)
    TASKS[id] = task
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=1000,
        help="Number of validation/test samples to generate",
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=2,
        help="Minimum crafting depth for hierarchical tasks",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum crafting depth",
    )
    args = parser.parse_args()

    train_data, val_data = create_textcraft_datasets(
        num_samples_train=args.num_samples,
        num_samples_val=args.eval_size,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    print(f"Generated {len(train_data)} training tasks and {len(val_data)} validation tasks")

    parent_dir = pathlib.Path(__file__).parent

    # Save training data
    train_file = parent_dir / "textcraft_train.jsonl"
    with open(train_file, "w") as f:
        for task in train_data:
            json.dump(asdict(task), f)
            f.write("\n")

    # Save validation data
    val_file = parent_dir / "textcraft_val.jsonl"
    with open(val_file, "w") as f:
        for task in val_data:
            json.dump(asdict(task), f)
            f.write("\n")

    print(f"Saved {len(train_data)} training samples to {train_file}")
    print(f"Saved {len(val_data)} validation samples to {val_file}")
