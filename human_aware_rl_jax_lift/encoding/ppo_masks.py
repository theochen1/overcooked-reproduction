"""Legacy 20-channel lossless encoding for PPO/CNN policies.

IMPORTANT: This mirrors the *legacy* human_aware_rl / overcooked_ai_py
`OvercookedGridworld.lossless_state_encoding` convention where the first two
axes are ordered as (width, height) and are indexed by (x, y) positions.
"""

import jax.numpy as jnp

from human_aware_rl_jax_lift.env.state import OBJ_DISH, OBJ_ONION, OBJ_SOUP, OvercookedState, Terrain


def _make_layer(shape, positions: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """Create a 2D layer with axis-0 = x, axis-1 = y (legacy convention)."""
    layer = jnp.zeros(shape, dtype=jnp.int32)
    xs = positions[:, 0]
    ys = positions[:, 1]
    # Use additive scatter to avoid padded (0,0) entries with value 0
    # overwriting a real non-zero feature that may also live at (0,0).
    return layer.at[xs, ys].add(values)


def _player_orientation_layer(
    shape,
    pos: jnp.ndarray,
    orientation_idx: jnp.ndarray,
    target_idx: int,
) -> jnp.ndarray:
    v = jnp.where(orientation_idx == target_idx, 1, 0)
    return _make_layer(shape, pos[None, :], jnp.array([v], dtype=jnp.int32))


def _object_layer_from_sources(
    shape,
    counter_positions: jnp.ndarray,
    counter_mask: jnp.ndarray,
    counter_obj: jnp.ndarray,
    obj_id: int,
    player_pos: jnp.ndarray,
    held_obj: jnp.ndarray,
):
    # Replicate TF legacy dict.update bug: when any player holds an object of
    # this type, all_objects_by_type overwrites unowned objects with player
    # objects, effectively dropping counter objects of the same type.
    any_player_holds = jnp.any(held_obj == obj_id)
    counter_vals = jnp.where(
        any_player_holds,
        jnp.zeros_like(counter_mask, dtype=jnp.int32),
        (counter_mask & (counter_obj == obj_id)).astype(jnp.int32),
    )
    counter_layer = _make_layer(shape, counter_positions, counter_vals)
    player_vals = (held_obj == obj_id).astype(jnp.int32)
    player_layer = _make_layer(shape, player_pos, player_vals)
    return counter_layer + player_layer


def lossless_state_encoding_20(terrain: Terrain, state: OvercookedState):
    """Return tuple(obs_for_player0, obs_for_player1), each WxHx20.

    Matches legacy `OvercookedGridworld.lossless_state_encoding` exactly.
    """
    h, w = terrain.grid.shape
    shape = (w, h)

    def process(primary: int):
        other = 1 - primary
        layers = []

        # 1-2 player loc
        layers.append(_make_layer(shape, state.player_pos[primary][None, :], jnp.array([1], dtype=jnp.int32)))
        layers.append(_make_layer(shape, state.player_pos[other][None, :], jnp.array([1], dtype=jnp.int32)))

        # 3-10 orientation channels
        for p in [primary, other]:
            for d in range(4):
                layers.append(_player_orientation_layer(shape, state.player_pos[p], state.player_or[p], d))

        # 11-15 base terrain masks
        pot_vals = terrain.pot_mask.astype(jnp.int32)
        ctr_vals = terrain.counter_mask.astype(jnp.int32)
        onion_vals = terrain.onion_disp_mask.astype(jnp.int32)
        dish_vals = terrain.dish_disp_mask.astype(jnp.int32)
        serve_vals = terrain.serve_mask.astype(jnp.int32)
        layers.append(_make_layer(shape, terrain.pot_positions, pot_vals))
        layers.append(_make_layer(shape, terrain.counter_positions, ctr_vals))
        layers.append(_make_layer(shape, terrain.onion_disp_positions, onion_vals))
        layers.append(_make_layer(shape, terrain.dish_disp_positions, dish_vals))
        layers.append(_make_layer(shape, terrain.serve_positions, serve_vals))

        # 16 onions_in_pot, 17 onions_cook_time
        # Replicate TF legacy dict.update bug: when any player holds a soup,
        # all_objects_by_type drops pot soup objects, zeroing these channels.
        any_holds_soup = jnp.any(state.held_obj == OBJ_SOUP)
        pot_mask_i = terrain.pot_mask.astype(jnp.int32)
        onion_count = jnp.where(
            any_holds_soup, jnp.zeros_like(state.pot_state[:, 1]),
            state.pot_state[:, 1],
        ) * pot_mask_i
        cook_time = jnp.where(
            any_holds_soup, jnp.zeros_like(state.pot_state[:, 2]),
            state.pot_state[:, 2],
        ) * pot_mask_i
        layers.append(_make_layer(shape, terrain.pot_positions, onion_count))
        layers.append(_make_layer(shape, terrain.pot_positions, cook_time))

        # 18 onion_soup_loc (soups not in pots)
        soup_layer = _object_layer_from_sources(
            shape=shape,
            counter_positions=terrain.counter_positions,
            counter_mask=terrain.counter_mask,
            counter_obj=state.counter_obj,
            obj_id=OBJ_SOUP,
            player_pos=state.player_pos,
            held_obj=state.held_obj,
        )
        layers.append(soup_layer)

        # 19 dishes, 20 onions
        dish_layer = _object_layer_from_sources(
            shape=shape,
            counter_positions=terrain.counter_positions,
            counter_mask=terrain.counter_mask,
            counter_obj=state.counter_obj,
            obj_id=OBJ_DISH,
            player_pos=state.player_pos,
            held_obj=state.held_obj,
        )
        onion_layer = _object_layer_from_sources(
            shape=shape,
            counter_positions=terrain.counter_positions,
            counter_mask=terrain.counter_mask,
            counter_obj=state.counter_obj,
            obj_id=OBJ_ONION,
            player_pos=state.player_pos,
            held_obj=state.held_obj,
        )
        layers.append(dish_layer)
        layers.append(onion_layer)

        stacked = jnp.stack(layers, axis=0)
        # Keep legacy axis order: (W, H, C)
        return jnp.transpose(stacked, (1, 2, 0))

    return process(0), process(1)
