"""Legacy-style handcrafted BC featurization."""

import jax.numpy as jnp

from human_aware_rl_jax_lift.env.state import (
    OBJ_DISH,
    OBJ_NONE,
    OBJ_ONION,
    OBJ_SOUP,
    OvercookedState,
    SOUP_ONION,
    TERRAIN_COUNTER,
    Terrain,
)


def _closest_delta_dist(src: jnp.ndarray, positions: jnp.ndarray, mask: jnp.ndarray):
    """Return closest delta + distance under a static-shape mask."""
    deltas = (positions - src[None, :]).astype(jnp.float32)
    manhattan = jnp.abs(deltas[:, 0]) + jnp.abs(deltas[:, 1])
    large = jnp.array(1e6, dtype=jnp.float32)
    masked_dist = jnp.where(mask, manhattan, large)
    idx = jnp.argmin(masked_dist)
    has_any = jnp.any(mask)
    best_delta = jnp.where(has_any, deltas[idx], jnp.zeros((2,), dtype=jnp.float32))
    best_dist = jnp.where(has_any, masked_dist[idx], large)
    return best_delta, best_dist, has_any


def _closest_delta(src: jnp.ndarray, positions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    delta, _, _ = _closest_delta_dist(src, positions, mask)
    return delta


def _held_obj_one_hot(obj_id: jnp.ndarray) -> jnp.ndarray:
    # Legacy uses onion/soup/dish axes.
    return jnp.array(
        [
            jnp.where(obj_id == OBJ_ONION, 1.0, 0.0),
            jnp.where(obj_id == OBJ_SOUP, 1.0, 0.0),
            jnp.where(obj_id == OBJ_DISH, 1.0, 0.0),
        ],
        dtype=jnp.float32,
    )


def _orientation_one_hot(o: jnp.ndarray) -> jnp.ndarray:
    return jnp.eye(4, dtype=jnp.float32)[o]


def _wall_features(terrain: Terrain, pos: jnp.ndarray) -> jnp.ndarray:
    dirs = jnp.array([[0, -1], [0, 1], [1, 0], [-1, 0]], dtype=jnp.int32)
    h, w = terrain.grid.shape
    out = []
    for d in dirs:
        p = pos + d
        in_bounds = (p[0] >= 0) & (p[1] >= 0) & (p[0] < w) & (p[1] < h)
        clipped = jnp.array(
            [jnp.clip(p[0], 0, w - 1), jnp.clip(p[1], 0, h - 1)],
            dtype=jnp.int32,
        )
        is_wall = ~in_bounds | (~terrain.walkable_mask[clipped[1], clipped[0]])
        out.append(jnp.where(is_wall, 1.0, 0.0))
    return jnp.array(out, dtype=jnp.float32)


def _facing_empty_counter(terrain: Terrain, state: OvercookedState, i: int) -> jnp.ndarray:
    dirs = jnp.array([[0, -1], [0, 1], [1, 0], [-1, 0]], dtype=jnp.int32)
    pos = state.player_pos[i]
    face_pos = pos + dirs[state.player_or[i]]
    is_counter = (
        (face_pos[0] >= 0)
        & (face_pos[1] >= 0)
        & (face_pos[0] < terrain.grid.shape[1])
        & (face_pos[1] < terrain.grid.shape[0])
        & (terrain.grid[face_pos[1], face_pos[0]] == TERRAIN_COUNTER)
    )
    match = (terrain.counter_positions == face_pos[None, :]).all(axis=1) & terrain.counter_mask
    idx = jnp.argmax(match.astype(jnp.int32))
    empty = terrain.counter_mask[idx] & (state.counter_obj[idx] == OBJ_NONE)
    return jnp.array([jnp.where(is_counter & empty, 1.0, 0.0)], dtype=jnp.float32)


def featurize_state_64(terrain: Terrain, state: OvercookedState):
    """Return `(features_p0, features_p1)` vectors following legacy feature layout."""

    def player_feats(i: int) -> jnp.ndarray:
        pos = state.player_pos[i]
        held = state.held_obj[i]
        pot_type = state.pot_state[:, 0]
        pot_items = state.pot_state[:, 1]
        pot_cook = state.pot_state[:, 2]
        pot_valid = terrain.pot_mask

        empty_pot_mask = pot_valid & (pot_type == 0)
        one_onion_pot_mask = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 1)
        two_onion_pot_mask = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 2)
        cooking_pot_mask = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 3) & (pot_cook < terrain.cook_time)
        ready_pot_mask = pot_valid & (pot_type == SOUP_ONION) & (pot_items == 3) & (pot_cook >= terrain.cook_time)

        counter_onion_mask = (state.counter_obj == OBJ_ONION) & terrain.counter_mask
        counter_dish_mask = (state.counter_obj == OBJ_DISH) & terrain.counter_mask
        counter_soup_mask = (state.counter_obj == OBJ_SOUP) & terrain.counter_mask

        onion_disp_delta, onion_disp_dist, onion_disp_valid = _closest_delta_dist(
            pos, terrain.onion_disp_positions, terrain.onion_disp_mask
        )
        onion_ctr_delta, onion_ctr_dist, onion_ctr_valid = _closest_delta_dist(
            pos, terrain.counter_positions, counter_onion_mask
        )
        use_onion_ctr = onion_ctr_valid & ((~onion_disp_valid) | (onion_ctr_dist < onion_disp_dist))
        onion_delta = jnp.where(use_onion_ctr, onion_ctr_delta, onion_disp_delta)

        dish_disp_delta, dish_disp_dist, dish_disp_valid = _closest_delta_dist(
            pos, terrain.dish_disp_positions, terrain.dish_disp_mask
        )
        dish_ctr_delta, dish_ctr_dist, dish_ctr_valid = _closest_delta_dist(
            pos, terrain.counter_positions, counter_dish_mask
        )
        use_dish_ctr = dish_ctr_valid & ((~dish_disp_valid) | (dish_ctr_dist < dish_disp_dist))
        dish_delta = jnp.where(use_dish_ctr, dish_ctr_delta, dish_disp_delta)

        soup_delta, _, _ = _closest_delta_dist(pos, terrain.counter_positions, counter_soup_mask)
        serve_delta, _, _ = _closest_delta_dist(pos, terrain.serve_positions, terrain.serve_mask)
        onion_delta = jnp.where((held == OBJ_ONION), jnp.zeros((2,), dtype=jnp.float32), onion_delta)
        dish_delta = jnp.where((held == OBJ_DISH), jnp.zeros((2,), dtype=jnp.float32), dish_delta)
        soup_delta = jnp.where((held == OBJ_SOUP), jnp.zeros((2,), dtype=jnp.float32), soup_delta)

        feats = [
            _orientation_one_hot(state.player_or[i]),
            _held_obj_one_hot(held),
            onion_delta,
            _closest_delta(pos, terrain.pot_positions, empty_pot_mask),
            _closest_delta(pos, terrain.pot_positions, one_onion_pot_mask),
            _closest_delta(pos, terrain.pot_positions, two_onion_pot_mask),
            _closest_delta(pos, terrain.pot_positions, cooking_pot_mask),
            _closest_delta(pos, terrain.pot_positions, ready_pot_mask),
            dish_delta,
            soup_delta,
            serve_delta,
            _facing_empty_counter(terrain, state, i),
            _wall_features(terrain, pos),
        ]
        return jnp.concatenate(feats, axis=0)

    p0 = player_feats(0)
    p1 = player_feats(1)
    rel10 = (state.player_pos[1] - state.player_pos[0]).astype(jnp.float32)
    rel01 = (state.player_pos[0] - state.player_pos[1]).astype(jnp.float32)
    abs0 = state.player_pos[0].astype(jnp.float32)
    abs1 = state.player_pos[1].astype(jnp.float32)
    return jnp.concatenate([p0, p1, rel10, abs0], axis=0), jnp.concatenate([p1, p0, rel01, abs1], axis=0)
