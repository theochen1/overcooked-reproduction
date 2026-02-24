"""Static-shape JAX state containers for Overcooked."""

from typing import Tuple

import jax.numpy as jnp
from flax import struct

# Terrain encoding
TERRAIN_EMPTY = 0
TERRAIN_COUNTER = 1
TERRAIN_POT = 2
TERRAIN_ONION = 3
TERRAIN_TOMATO = 4
TERRAIN_DISH = 5
TERRAIN_SERVE = 6

# Object encoding
OBJ_NONE = 0
OBJ_ONION = 1
OBJ_TOMATO = 2
OBJ_DISH = 3
OBJ_SOUP = 4

# Soup encoding
SOUP_NONE = 0
SOUP_ONION = 1
SOUP_TOMATO = 2

# Legacy action encoding
ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_STAY = 4
ACTION_INTERACT = 5

DIR_VECS = jnp.array(
    [
        [0, -1],  # NORTH
        [0, 1],   # SOUTH
        [1, 0],   # EAST
        [-1, 0],  # WEST
        [0, 0],   # STAY
        [0, 0],   # INTERACT
    ],
    dtype=jnp.int32,
)


@struct.dataclass
class Terrain:
    """Static terrain description for one layout."""

    grid: jnp.ndarray  # [H, W] int32 terrain code
    walkable_mask: jnp.ndarray  # [H, W] bool
    player_start: jnp.ndarray  # [2, 2] int32

    pot_positions: jnp.ndarray  # [max_pots, 2] int32
    pot_mask: jnp.ndarray  # [max_pots] bool

    counter_positions: jnp.ndarray  # [max_counters, 2] int32
    counter_mask: jnp.ndarray  # [max_counters] bool

    onion_disp_positions: jnp.ndarray  # [max_onion_disp, 2]
    onion_disp_mask: jnp.ndarray  # [max_onion_disp]
    tomato_disp_positions: jnp.ndarray  # [max_tomato_disp, 2]
    tomato_disp_mask: jnp.ndarray  # [max_tomato_disp]
    dish_disp_positions: jnp.ndarray  # [max_dish_disp, 2]
    dish_disp_mask: jnp.ndarray  # [max_dish_disp]
    serve_positions: jnp.ndarray  # [max_serve, 2]
    serve_mask: jnp.ndarray  # [max_serve]

    cook_time: int
    num_items_for_soup: int
    delivery_reward: int


@struct.dataclass
class OvercookedState:
    """Static-shape Overcooked state for jit/vmap-able transitions."""

    player_pos: jnp.ndarray  # [2, 2] int32
    player_or: jnp.ndarray  # [2] int32 in {0,1,2,3}

    held_obj: jnp.ndarray  # [2] int32 object id
    held_soup: jnp.ndarray  # [2, 3] int32: (soup_type, num_items, cook_time) if held_obj == OBJ_SOUP

    pot_state: jnp.ndarray  # [max_pots, 3] int32: (soup_type, num_items, cook_time), soup_type=0 => empty
    counter_obj: jnp.ndarray  # [max_counters] int32 object id
    counter_soup: jnp.ndarray  # [max_counters, 3] int32 soup payload for counter soups

    timestep: jnp.ndarray  # scalar int32
    done: jnp.ndarray  # scalar bool


def make_initial_state(terrain: Terrain) -> OvercookedState:
    """Construct a default empty initial state from terrain metadata."""

    max_pots = terrain.pot_positions.shape[0]
    max_counters = terrain.counter_positions.shape[0]

    return OvercookedState(
        player_pos=terrain.player_start.astype(jnp.int32),
        player_or=jnp.array([1, 1], dtype=jnp.int32),  # SOUTH by legacy default
        held_obj=jnp.zeros((2,), dtype=jnp.int32),
        held_soup=jnp.zeros((2, 3), dtype=jnp.int32),
        pot_state=jnp.zeros((max_pots, 3), dtype=jnp.int32),
        counter_obj=jnp.zeros((max_counters,), dtype=jnp.int32),
        counter_soup=jnp.zeros((max_counters, 3), dtype=jnp.int32),
        timestep=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False),
    )


def in_bounds(grid: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    """Check if pos is inside a [H, W] grid."""
    h, w = grid.shape
    return (pos[0] >= 0) & (pos[1] >= 0) & (pos[0] < w) & (pos[1] < h)


def pos_to_yx(pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert (x, y) to (y, x) for array indexing."""
    return pos[1], pos[0]
