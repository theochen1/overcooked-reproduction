"""Pure JAX transition function for legacy Overcooked dynamics."""

from typing import Dict, Tuple

import jax.numpy as jnp

from .collisions import resolve_player_collisions
from .interactions import resolve_interacts
from .reward_shaping import DEFAULT_REW_SHAPING_PARAMS
from .state import (
    ACTION_INTERACT,
    ACTION_STAY,
    DIR_VECS,
    OvercookedState,
    Terrain,
    in_bounds,
    pos_to_yx,
)


def _move_if_direction(terrain: Terrain, pos: jnp.ndarray, ori: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy movement semantics: interact doesn't move, stay keeps orientation."""
    is_interact = action == ACTION_INTERACT
    is_stay = action == ACTION_STAY

    proposed = pos + DIR_VECS[action]
    valid = in_bounds(terrain.grid, proposed) & terrain.walkable_mask[pos_to_yx(proposed)]
    new_pos = jnp.where((~is_interact) & valid, proposed, pos)
    new_ori = jnp.where((~is_interact) & (~is_stay), action, ori)
    return new_pos, new_ori


def _resolve_movement(terrain: Terrain, state: OvercookedState, joint_action: jnp.ndarray) -> OvercookedState:
    old_pos = state.player_pos
    old_or = state.player_or
    p0, o0 = _move_if_direction(terrain, old_pos[0], old_or[0], joint_action[0])
    p1, o1 = _move_if_direction(terrain, old_pos[1], old_or[1], joint_action[1])

    new_pos = jnp.stack([p0, p1], axis=0)
    new_pos = resolve_player_collisions(old_pos, new_pos)
    new_or = jnp.array([o0, o1], dtype=jnp.int32)
    return state.replace(player_pos=new_pos, player_or=new_or)


def _step_environment_effects(terrain: Terrain, state: OvercookedState) -> OvercookedState:
    """Increment cook time for full soups in pots, capped at cook_time."""
    soup_type = state.pot_state[:, 0]
    num_items = state.pot_state[:, 1]
    cook_time = state.pot_state[:, 2]
    is_full = num_items == terrain.num_items_for_soup
    is_nonempty = soup_type != 0
    can_cook = is_nonempty & is_full & (cook_time < terrain.cook_time) & terrain.pot_mask
    new_cook = jnp.where(can_cook, cook_time + 1, cook_time)
    return state.replace(pot_state=state.pot_state.at[:, 2].set(new_cook))


def step(
    terrain: Terrain,
    state: OvercookedState,
    joint_action: jnp.ndarray,
    reward_shaping_params: Dict[str, float] | None = None,
) -> Tuple[OvercookedState, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Legacy transition order:
      1) resolve_interacts
      2) resolve_movement
      3) step_environment_effects
    """
    rsp = DEFAULT_REW_SHAPING_PARAMS if reward_shaping_params is None else reward_shaping_params
    st, sparse_reward, shaped_reward = resolve_interacts(
        terrain=terrain,
        state=state,
        joint_action=joint_action.astype(jnp.int32),
        placement_in_pot_rew=float(rsp["PLACEMENT_IN_POT_REW"]),
        dish_pickup_rew=float(rsp["DISH_PICKUP_REWARD"]),
        soup_pickup_rew=float(rsp["SOUP_PICKUP_REWARD"]),
    )
    st = _resolve_movement(terrain, st, joint_action.astype(jnp.int32))
    st = _step_environment_effects(terrain, st)
    st = st.replace(timestep=st.timestep + jnp.array(1, dtype=jnp.int32))
    info = {
        "shaped_r_by_agent": jnp.array([shaped_reward, shaped_reward], dtype=jnp.float32),
    }
    return st, sparse_reward.astype(jnp.float32), shaped_reward.astype(jnp.float32), info
