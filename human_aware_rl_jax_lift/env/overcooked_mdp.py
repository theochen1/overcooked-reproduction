"""Pure JAX transition function for legacy Overcooked dynamics."""

from typing import Dict, Optional, Tuple

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

# Base reward values (before shaping factor scaling)
_BASE_PLACEMENT_REW = float(DEFAULT_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"])
_BASE_DISH_REW      = float(DEFAULT_REW_SHAPING_PARAMS["DISH_PICKUP_REWARD"])
_BASE_SOUP_REW      = float(DEFAULT_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"])


def _move_if_direction(terrain: Terrain, pos: jnp.ndarray, ori: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy movement semantics: interact doesn't move, stay keeps orientation."""
    is_interact = action == ACTION_INTERACT
    is_stay = action == ACTION_STAY

    proposed = pos + DIR_VECS[action]
    h, w = terrain.grid.shape
    clipped = jnp.array(
        [
            jnp.clip(proposed[0], 0, w - 1),
            jnp.clip(proposed[1], 0, h - 1),
        ],
        dtype=jnp.int32,
    )
    valid = in_bounds(terrain.grid, proposed) & terrain.walkable_mask[pos_to_yx(clipped)]
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
    shaping_factor: Optional[jnp.ndarray] = None,
    reward_shaping_params: Optional[Dict] = None,
) -> Tuple[OvercookedState, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Legacy transition order:
      1) resolve_interacts
      2) resolve_movement
      3) step_environment_effects

    Accepts `shaping_factor` as a JAX scalar (preferred — vmappable, no recompile
    when value changes) or legacy `reward_shaping_params` dict for backward compat.
    """
    # Resolve shaping factor
    if shaping_factor is not None:
        sf = shaping_factor
    elif reward_shaping_params is not None:
        # Legacy dict path: infer scalar factor from PLACEMENT_IN_POT_REW
        base = _BASE_PLACEMENT_REW if _BASE_PLACEMENT_REW > 0 else 1.0
        sf = jnp.array(
            float(reward_shaping_params.get("PLACEMENT_IN_POT_REW", _BASE_PLACEMENT_REW)) / base
        )
    else:
        sf = jnp.array(1.0)

    # Compute legacy dense reward in *unscaled* units first, then apply shaping
    # factor to the training reward path. This preserves TF logging contract:
    # - ep_sparse_r: unscaled sparse
    # - ep_shaped_r: unscaled dense
    # - episode["r"]: sparse + dense * shaping_factor
    st, sparse_reward, shaped_unscaled = resolve_interacts(
        terrain=terrain,
        state=state,
        joint_action=joint_action.astype(jnp.int32),
        placement_in_pot_rew=_BASE_PLACEMENT_REW,
        dish_pickup_rew=_BASE_DISH_REW,
        soup_pickup_rew=_BASE_SOUP_REW,
    )
    shaped_reward = shaped_unscaled * sf
    st = _resolve_movement(terrain, st, joint_action.astype(jnp.int32))
    st = _step_environment_effects(terrain, st)
    st = st.replace(timestep=st.timestep + jnp.array(1, dtype=jnp.int32))
    info = {
        "shaped_r_by_agent": jnp.array([shaped_reward, shaped_reward], dtype=jnp.float32),
        "shaped_r_unscaled": shaped_unscaled.astype(jnp.float32),
        "shaping_factor": sf.astype(jnp.float32),
    }
    return st, sparse_reward.astype(jnp.float32), shaped_reward.astype(jnp.float32), info
