"""Recipe loader for Minecraft crafting recipes."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def extract_item_name(item_str: str) -> str:
    """Extract item name from minecraft:item_name format."""
    if ":" in item_str:
        return item_str.split(":")[1]
    return item_str


class Recipe:
    """Represents a Minecraft crafting recipe."""

    def __init__(self, recipe_data: dict, filename: str):
        self.filename = filename
        self.type = recipe_data.get("type", "")

        # Handle result - can be dict or list
        result = recipe_data.get("result")
        if isinstance(result, list):
            # Some recipes have list results, use first one
            result = result[0]
        if isinstance(result, str):
            # Some recipes have string results
            self.result_item = extract_item_name(result)
            self.result_count = 1
        elif isinstance(result, dict):
            if "item" in result:
                self.result_item = extract_item_name(result["item"])
                self.result_count = result.get("count", 1)
            else:
                raise ValueError(f"Invalid result format in {filename}: {result}")
        else:
            raise ValueError(f"Invalid result format in {filename}: {result}")

        # Parse ingredients based on recipe type
        self.ingredients: Dict[str, int] = {}

        if self.type == "minecraft:crafting_shapeless":
            # Shapeless: list of ingredients
            for ing in recipe_data.get("ingredients", []):
                item_name = None
                if "item" in ing:
                    item_name = extract_item_name(ing["item"])
                elif "tag" in ing:
                    # For tags, we'll use the tag name as-is for now
                    item_name = f"tag:{extract_item_name(ing['tag'])}"

                if item_name:
                    self.ingredients[item_name] = self.ingredients.get(item_name, 0) + 1

        elif self.type == "minecraft:crafting_shaped":
            # Shaped: pattern + key
            pattern = recipe_data.get("pattern", [])
            key = recipe_data.get("key", {})

            for row in pattern:
                for char in row:
                    if char != " " and char in key:
                        ing = key[char]
                        item_name = None
                        if "item" in ing:
                            item_name = extract_item_name(ing["item"])
                        elif "tag" in ing:
                            item_name = f"tag:{extract_item_name(ing['tag'])}"

                        if item_name:
                            self.ingredients[item_name] = self.ingredients.get(item_name, 0) + 1

        # Normalize item names (remove minecraft: prefix if present)
        normalized_ingredients = {}
        for item, count in self.ingredients.items():
            normalized_ingredients[item] = count
        self.ingredients = normalized_ingredients

    def __repr__(self) -> str:
        return f"Recipe({self.result_item} <- {self.ingredients})"


class RecipeDatabase:
    """Database of all Minecraft recipes."""

    def __init__(self, recipes_dir: Path):
        self.recipes_dir = Path(recipes_dir)
        self.recipes: Dict[str, List[Recipe]] = defaultdict(list)
        self.all_items: Set[str] = set()
        self.all_tags: Set[str] = set()  # Track all tags found in recipes
        self.tag_to_items: Dict[str, Set[str]] = defaultdict(set)  # Map tags to items that satisfy them
        self._load_all_recipes()
        self._build_dependency_graph()
        self._build_tag_mappings()

    def _load_all_recipes(self):
        """Load all recipe JSON files."""
        for recipe_file in self.recipes_dir.glob("*.json"):
            try:
                with open(recipe_file, "r") as f:
                    recipe_data = json.load(f)

                # Only load crafting recipes (shaped and shapeless)
                recipe_type = recipe_data.get("type", "")
                if recipe_type not in ["minecraft:crafting_shaped", "minecraft:crafting_shapeless"]:
                    continue  # Skip smelting, stonecutting, etc.

                recipe = Recipe(recipe_data, recipe_file.stem)
                self.recipes[recipe.result_item].append(recipe)
                self.all_items.add(recipe.result_item)
                # Add ingredients to all_items and track tags
                for ing in recipe.ingredients.keys():
                    if ing.startswith("tag:"):
                        tag_name = ing.replace("tag:", "")
                        self.all_tags.add(tag_name)
                        # DO NOT add result item to tag mapping here - this causes incorrect mappings
                        # (e.g., warped_planks crafted from tag:warped_stems shouldn't satisfy tag:warped_stems)
                        # Pattern matching in _build_tag_mappings will handle correct mappings
                    else:
                        self.all_items.add(ing)
            except Exception:
                # Silently skip recipes we can't parse
                continue

    def _build_dependency_graph(self):
        """Build a dependency graph for finding hierarchical recipes."""
        # Items that can be crafted (have recipes)
        self.craftable_items = set(self.recipes.keys())

        # Items that are base materials (no recipes)
        self.base_items = self.all_items - self.craftable_items

    def _build_tag_mappings(self):
        """Build mappings from tags to items that satisfy them by analyzing recipes."""
        # For each tag, find items that can satisfy it
        # An item satisfies a tag if it matches tag patterns (e.g., "planks" tag -> items ending in "_planks")
        # NOTE: We do NOT add result items to tag mappings - if an item is crafted FROM a tag,
        # it doesn't satisfy that tag (e.g., warped_planks is crafted from tag:warped_stems,
        # but warped_planks doesn't satisfy tag:warped_stems)

        # First, clear and rebuild tag mappings more carefully
        self.tag_to_items.clear()

        # DO NOT add result items to tag mappings - this was causing incorrect mappings
        # (e.g., warped_planks being added to tag:warped_stems)

        # Now add items that match tag patterns more precisely
        for tag_name in self.all_tags:
            tag_lower = tag_name.lower()

            # Generic "planks" tag - all planks items
            if tag_lower == "planks":
                for item in self.all_items:
                    if item.endswith("_planks"):
                        self.tag_to_items[tag_name].add(item)

            # Generic "logs" tag - all log/stem items
            elif tag_lower == "logs":
                for item in self.all_items:
                    if item.endswith("_log") or item.endswith("_stem") or item == "log" or item == "stem":
                        self.tag_to_items[tag_name].add(item)

            # Specific wood type logs (e.g., "birch_logs")
            elif tag_lower.endswith("_logs"):
                wood_type = tag_lower.replace("_logs", "")
                # Find items that match this specific wood type
                for item in self.all_items:
                    if item.startswith(wood_type + "_log") or item.startswith(wood_type + "_stem"):
                        self.tag_to_items[tag_name].add(item)

            # Specific wood type stems (e.g., "crimson_stems")
            elif tag_lower.endswith("_stems"):
                wood_type = tag_lower.replace("_stems", "")
                for item in self.all_items:
                    if item.startswith(wood_type + "_stem") or item.startswith(wood_type + "_log"):
                        self.tag_to_items[tag_name].add(item)

            # Wooden slabs
            elif "wooden_slabs" in tag_lower or tag_lower == "wooden_slab":
                for item in self.all_items:
                    if "_slab" in item and ("planks" in item or "wood" in item):
                        self.tag_to_items[tag_name].add(item)

            # Stone tool materials
            elif "stone_tool" in tag_lower:
                stone_items = ["cobblestone", "blackstone", "cobbled_deepslate", "stone"]
                for item in stone_items:
                    if item in self.all_items:
                        self.tag_to_items[tag_name].add(item)

            # Stone crafting materials
            elif "stone_crafting" in tag_lower:
                stone_items = ["cobblestone", "stone", "smooth_stone"]
                for item in stone_items:
                    if item in self.all_items:
                        self.tag_to_items[tag_name].add(item)

            # Coals
            elif tag_lower == "coals":
                coal_items = ["coal", "charcoal"]
                for item in coal_items:
                    if item in self.all_items:
                        self.tag_to_items[tag_name].add(item)

            # Soul fire base blocks
            elif "soul_fire" in tag_lower:
                soul_items = ["soul_sand", "soul_soil"]
                for item in soul_items:
                    if item in self.all_items:
                        self.tag_to_items[tag_name].add(item)

            # Wool
            elif tag_lower == "wool":
                for item in self.all_items:
                    if item.endswith("_wool"):
                        self.tag_to_items[tag_name].add(item)

    def get_items_for_tag(self, tag_name: str) -> List[str]:
        """Get all items that satisfy a given tag."""
        return sorted(list(self.tag_to_items.get(tag_name, set())))

    def get_recipes_for_item(self, item: str) -> List[Recipe]:
        """Get all recipes that produce the given item."""
        return self.recipes.get(item, [])

    def can_craft(self, item: str) -> bool:
        """Check if an item can be crafted."""
        return item in self.recipes

    def is_base_item(self, item: str) -> bool:
        """Check if an item is a base material (cannot be crafted)."""
        return item in self.base_items

    def get_crafting_depth(self, item: str, visited: Optional[Set[str]] = None) -> int:
        """Get the maximum crafting depth required to craft an item."""
        if visited is None:
            visited = set()

        if item in visited:
            return 0  # Cycle detected

        if self.is_base_item(item):
            return 0

        if item not in self.recipes:
            return 0

        visited.add(item)
        max_depth = 0

        for recipe in self.recipes[item]:
            for ingredient in recipe.ingredients.keys():
                if ingredient.startswith("tag:"):
                    continue  # Skip tags for depth calculation
                depth = self.get_crafting_depth(ingredient, visited.copy())
                max_depth = max(max_depth, depth + 1)

        return max_depth

    def find_hierarchical_recipes(self, min_depth: int = 2, max_depth: int = 5) -> List[str]:
        """Find items that require hierarchical crafting (multiple steps)."""
        hierarchical = []
        for item in self.craftable_items:
            depth = self.get_crafting_depth(item)
            if min_depth <= depth <= max_depth:
                hierarchical.append(item)
        return hierarchical

    def get_recipe_chain(self, item: str, visited: Optional[Set[str]] = None) -> List[Tuple[str, Recipe]]:
        """Get a chain of recipes needed to craft an item."""
        if visited is None:
            visited = set()

        if item in visited or self.is_base_item(item):
            return []

        if item not in self.recipes:
            return []

        visited.add(item)
        recipes = self.recipes[item]
        if not recipes:
            return []

        # Use the first recipe (could be improved to select best)
        recipe = recipes[0]
        chain = [(item, recipe)]

        for ingredient in recipe.ingredients.keys():
            if ingredient.startswith("tag:"):
                continue
            sub_chain = self.get_recipe_chain(ingredient, visited.copy())
            chain.extend(sub_chain)

        return chain
