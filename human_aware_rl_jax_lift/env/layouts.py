"""Layout parsing utilities from legacy overcooked .layout files."""

import ast
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import jax.numpy as jnp
import numpy as np

from .state import (
    TERRAIN_COUNTER,
    TERRAIN_DISH,
    TERRAIN_EMPTY,
    TERRAIN_ONION,
    TERRAIN_POT,
    TERRAIN_SERVE,
    TERRAIN_TOMATO,
    Terrain,
)

_DEFAULT_LEGACY_LAYOUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "human_aware_rl"
    / "overcooked_ai"
    / "overcooked_ai_py"
    / "data"
    / "layouts"
)
_PACKAGE_LAYOUT_DIR = Path(__file__).resolve().parent / "data" / "layouts"
LAYOUT_DIR_ENV = "OVERCOOKED_LAYOUT_DIR"

CHAR_TO_TERRAIN = {
    " ": TERRAIN_EMPTY,
    "X": TERRAIN_COUNTER,
    "P": TERRAIN_POT,
    "O": TERRAIN_ONION,
    "T": TERRAIN_TOMATO,
    "D": TERRAIN_DISH,
    "S": TERRAIN_SERVE,
}


def _pad_positions(positions: List[Tuple[int, int]], length: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.zeros((length, 2), dtype=np.int32)
    mask = np.zeros((length,), dtype=np.bool_)
    if positions:
        n = min(len(positions), length)
        arr[:n] = np.array(positions[:n], dtype=np.int32)
        mask[:n] = True
    return arr, mask


def _extract_grid_lines(grid_str: str) -> List[str]:
    return [line.strip() for line in grid_str.splitlines() if line.strip()]


def parse_layout(layout_name: str) -> Terrain:
    """Parse a legacy `.layout` file into a static Terrain struct."""
    env_override = os.environ.get(LAYOUT_DIR_ENV)
    if env_override:
        base_dir = Path(env_override)
    elif _PACKAGE_LAYOUT_DIR.exists():
        base_dir = _PACKAGE_LAYOUT_DIR
    else:
        base_dir = _DEFAULT_LEGACY_LAYOUT_DIR
    layout_path = base_dir / f"{layout_name}.layout"
    data = ast.literal_eval(layout_path.read_text())
    lines = _extract_grid_lines(data["grid"])

    h, w = len(lines), len(lines[0])
    grid = np.zeros((h, w), dtype=np.int32)
    walkable = np.zeros((h, w), dtype=np.bool_)

    players = [None, None]
    pots, counters, onions, tomatoes, dishes, serves = [], [], [], [], [], []

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if ch in ("1", "2"):
                idx = int(ch) - 1
                players[idx] = (x, y)
                grid[y, x] = TERRAIN_EMPTY
                walkable[y, x] = True
                continue

            terrain = CHAR_TO_TERRAIN[ch]
            grid[y, x] = terrain
            walkable[y, x] = terrain == TERRAIN_EMPTY

            if terrain == TERRAIN_POT:
                pots.append((x, y))
            elif terrain == TERRAIN_COUNTER:
                counters.append((x, y))
            elif terrain == TERRAIN_ONION:
                onions.append((x, y))
            elif terrain == TERRAIN_TOMATO:
                tomatoes.append((x, y))
            elif terrain == TERRAIN_DISH:
                dishes.append((x, y))
            elif terrain == TERRAIN_SERVE:
                serves.append((x, y))

    max_pots = max(len(pots), 1)
    max_counters = max(len(counters), 1)
    max_onions = max(len(onions), 1)
    max_tomatoes = max(len(tomatoes), 1)
    max_dishes = max(len(dishes), 1)
    max_serves = max(len(serves), 1)

    pot_pos, pot_mask = _pad_positions(pots, max_pots)
    ctr_pos, ctr_mask = _pad_positions(counters, max_counters)
    onion_pos, onion_mask = _pad_positions(onions, max_onions)
    tomato_pos, tomato_mask = _pad_positions(tomatoes, max_tomatoes)
    dish_pos, dish_mask = _pad_positions(dishes, max_dishes)
    serve_pos, serve_mask = _pad_positions(serves, max_serves)

    cook_time = int(data.get("cook_time", 20))
    num_items = int(data.get("num_items_for_soup", 3))
    delivery_reward = int(data.get("delivery_reward", 20))

    return Terrain(
        grid=jnp.array(grid, dtype=jnp.int32),
        walkable_mask=jnp.array(walkable),
        player_start=jnp.array(players, dtype=jnp.int32),
        pot_positions=jnp.array(pot_pos, dtype=jnp.int32),
        pot_mask=jnp.array(pot_mask),
        counter_positions=jnp.array(ctr_pos, dtype=jnp.int32),
        counter_mask=jnp.array(ctr_mask),
        onion_disp_positions=jnp.array(onion_pos, dtype=jnp.int32),
        onion_disp_mask=jnp.array(onion_mask),
        tomato_disp_positions=jnp.array(tomato_pos, dtype=jnp.int32),
        tomato_disp_mask=jnp.array(tomato_mask),
        dish_disp_positions=jnp.array(dish_pos, dtype=jnp.int32),
        dish_disp_mask=jnp.array(dish_mask),
        serve_positions=jnp.array(serve_pos, dtype=jnp.int32),
        serve_mask=jnp.array(serve_mask),
        cook_time=cook_time,
        num_items_for_soup=num_items,
        delivery_reward=delivery_reward,
    )


def parse_paper_layouts() -> dict[str, Terrain]:
    """Load the five legacy paper layouts from `human_aware_rl`."""
    names = ["simple", "unident_s", "random0", "random1", "random3"]
    return {name: parse_layout(name) for name in names}
