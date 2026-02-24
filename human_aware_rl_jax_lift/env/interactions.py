"""Interaction semantics for legacy Overcooked terrain types."""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

from .state import (
    OBJ_DISH,
    OBJ_NONE,
    OBJ_ONION,
    OBJ_SOUP,
    OBJ_TOMATO,
    OvercookedState,
    SOUP_ONION,
    SOUP_TOMATO,
    TERRAIN_COUNTER,
    TERRAIN_DISH,
    TERRAIN_ONION,
    TERRAIN_POT,
    TERRAIN_SERVE,
    TERRAIN_TOMATO,
    Terrain,
)


def _match_position_index(positions: jnp.ndarray, mask: jnp.ndarray, target: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    eq = jnp.all(positions == target[None, :], axis=1) & mask
    idx = jnp.argmax(eq.astype(jnp.int32))
    return idx, jnp.any(eq)


def _object_type_to_soup_type(obj_type: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(obj_type == OBJ_ONION, SOUP_ONION, SOUP_TOMATO)


def resolve_interacts(
    terrain: Terrain,
    state: OvercookedState,
    joint_action: jnp.ndarray,
    placement_in_pot_rew: float,
    dish_pickup_rew: float,
    soup_pickup_rew: float,
) -> Tuple[OvercookedState, jnp.ndarray, jnp.ndarray]:
    """
    Resolve legacy INTERACT logic in player order.

    Returns updated_state, sparse_reward, shaped_reward.
    """
    sparse_reward = jnp.array(0.0, dtype=jnp.float32)
    shaped_reward = jnp.array(0.0, dtype=jnp.float32)

    def _step_one_player(carry, player_idx):
        st, sparse_r, shaped_r = carry
        action = joint_action[player_idx]
        do_interact = action == 5  # ACTION_INTERACT

        pos = st.player_pos[player_idx]
        ori = st.player_or[player_idx]
        interact_pos = pos + jnp.array(
            [[0, -1], [0, 1], [1, 0], [-1, 0]],
            dtype=jnp.int32,
        )[ori]

        h, w = terrain.grid.shape
        x, y = interact_pos[0], interact_pos[1]
        in_bounds = (x >= 0) & (y >= 0) & (x < w) & (y < h)
        terrain_type = jnp.where(in_bounds, terrain.grid[y, x], jnp.array(-1, dtype=jnp.int32))

        held_obj = st.held_obj[player_idx]

        # Counter: place/pick up
        ctr_idx, ctr_exists = _match_position_index(terrain.counter_positions, terrain.counter_mask, interact_pos)
        ctr_obj = st.counter_obj[ctr_idx]
        can_place_on_counter = (held_obj != OBJ_NONE) & ctr_exists & (ctr_obj == OBJ_NONE)
        can_pick_from_counter = (held_obj == OBJ_NONE) & ctr_exists & (ctr_obj != OBJ_NONE)

        st = st.replace(
            counter_obj=st.counter_obj.at[ctr_idx].set(jnp.where(can_place_on_counter, held_obj, ctr_obj)),
            held_obj=st.held_obj.at[player_idx].set(
                jnp.where(
                    can_place_on_counter,
                    OBJ_NONE,
                    jnp.where(can_pick_from_counter, ctr_obj, held_obj),
                )
            ),
        )

        # Onion/Tomato/Dish pickup
        onion_pick = (terrain_type == TERRAIN_ONION) & (st.held_obj[player_idx] == OBJ_NONE)
        tomato_pick = (terrain_type == TERRAIN_TOMATO) & (st.held_obj[player_idx] == OBJ_NONE)
        dish_pick = (terrain_type == TERRAIN_DISH) & (st.held_obj[player_idx] == OBJ_NONE)
        st = st.replace(
            held_obj=st.held_obj.at[player_idx].set(
                jnp.where(
                    onion_pick,
                    OBJ_ONION,
                    jnp.where(tomato_pick, OBJ_TOMATO, jnp.where(dish_pick, OBJ_DISH, st.held_obj[player_idx])),
                )
            )
        )
        shaped_r = shaped_r + jnp.where(dish_pick, dish_pickup_rew, 0.0)

        # Pot interactions
        pot_idx, pot_exists = _match_position_index(terrain.pot_positions, terrain.pot_mask, interact_pos)
        pot = st.pot_state[pot_idx]
        p_soup_type, p_num_items, p_cook_time = pot[0], pot[1], pot[2]
        player_obj = st.held_obj[player_idx]

        # Ingredient into pot
        is_ing = (player_obj == OBJ_ONION) | (player_obj == OBJ_TOMATO)
        pot_empty = p_soup_type == 0
        ing_type = _object_type_to_soup_type(player_obj)
        same_type = p_soup_type == ing_type
        can_add_ing = pot_exists & is_ing & ((pot_empty) | (same_type & (p_num_items < terrain.num_items_for_soup)))

        new_soup_type = jnp.where(pot_empty, ing_type, p_soup_type)
        new_num_items = jnp.where(can_add_ing, jnp.minimum(p_num_items + 1, terrain.num_items_for_soup), p_num_items)
        new_pot = jnp.array([new_soup_type, new_num_items, jnp.where(can_add_ing, 0, p_cook_time)], dtype=jnp.int32)
        st = st.replace(
            pot_state=st.pot_state.at[pot_idx].set(jnp.where(can_add_ing, new_pot, pot)),
            held_obj=st.held_obj.at[player_idx].set(jnp.where(can_add_ing, OBJ_NONE, player_obj)),
        )
        shaped_r = shaped_r + jnp.where(can_add_ing, placement_in_pot_rew, 0.0)

        # Dish picks up ready soup
        player_obj = st.held_obj[player_idx]
        pot = st.pot_state[pot_idx]
        can_pick_soup = (
            pot_exists
            & (player_obj == OBJ_DISH)
            & (pot[0] != 0)
            & (pot[1] == terrain.num_items_for_soup)
            & (pot[2] >= terrain.cook_time)
        )
        st = st.replace(
            held_obj=st.held_obj.at[player_idx].set(jnp.where(can_pick_soup, OBJ_SOUP, player_obj)),
            held_soup=st.held_soup.at[player_idx].set(
                jnp.where(can_pick_soup, pot, st.held_soup[player_idx])
            ),
            pot_state=st.pot_state.at[pot_idx].set(jnp.where(can_pick_soup, jnp.zeros((3,), dtype=jnp.int32), pot)),
        )
        shaped_r = shaped_r + jnp.where(can_pick_soup, soup_pickup_rew, 0.0)

        # Serve soup
        can_serve = (terrain_type == TERRAIN_SERVE) & (st.held_obj[player_idx] == OBJ_SOUP)
        st = st.replace(
            held_obj=st.held_obj.at[player_idx].set(jnp.where(can_serve, OBJ_NONE, st.held_obj[player_idx])),
            held_soup=st.held_soup.at[player_idx].set(
                jnp.where(can_serve, jnp.zeros((3,), dtype=jnp.int32), st.held_soup[player_idx])
            ),
        )
        sparse_r = sparse_r + jnp.where(can_serve, float(terrain.delivery_reward), 0.0)

        # Apply only when action is INTERACT
        st = tree_util.tree_map(
            lambda a, b: jnp.where(do_interact, a, b),
            st,
            carry[0],
        )
        sparse_r = jnp.where(do_interact, sparse_r, carry[1])
        shaped_r = jnp.where(do_interact, shaped_r, carry[2])
        return (st, sparse_r, shaped_r), None

    (state, sparse_reward, shaped_reward), _ = jax.lax.scan(
        _step_one_player, (state, sparse_reward, shaped_reward), jnp.arange(2, dtype=jnp.int32)
    )
    return state, sparse_reward, shaped_reward
