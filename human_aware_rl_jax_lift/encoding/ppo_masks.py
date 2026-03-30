"""20-channel lossless state encoding for PPO.

The first two axes are ordered as (width, height), indexed by (x, y) positions.
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
    *,
    tf_compat: bool = True,
):
    if tf_compat:
        # When tf_compat=True, counter objects of this type are zeroed if any
        # player holds the same type.
        any_player_holds = jnp.any(held_obj == obj_id)
        counter_vals = jnp.where(
            any_player_holds,
            jnp.zeros_like(counter_mask, dtype=jnp.int32),
            (counter_mask & (counter_obj == obj_id)).astype(jnp.int32),
        )
    else:
        # Fixed: always show counter objects regardless of what players hold.
        counter_vals = (counter_mask & (counter_obj == obj_id)).astype(jnp.int32)
    counter_layer = _make_layer(shape, counter_positions, counter_vals)
    player_vals = (held_obj == obj_id).astype(jnp.int32)
    player_layer = _make_layer(shape, player_pos, player_vals)
    return counter_layer + player_layer


def lossless_state_encoding_20(terrain: Terrain, state: OvercookedState, *, tf_compat: bool = True):
    """Return tuple(obs_for_player0, obs_for_player1), each WxHx20.

    When tf_compat=True (default):
    - Pot channels are zeroed when any player holds a soup.
    - Counter objects are zeroed when any player holds the same type.

    When tf_compat=False, observations always reflect the true game state.
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
        pot_mask_i = terrain.pot_mask.astype(jnp.int32)
        if tf_compat:
            # When tf_compat=True, pot channels are zeroed if any player holds a soup.
            any_holds_soup = jnp.any(state.held_obj == OBJ_SOUP)
            onion_count = jnp.where(
                any_holds_soup, jnp.zeros_like(state.pot_state[:, 1]),
                state.pot_state[:, 1],
            ) * pot_mask_i
            cook_time = jnp.where(
                any_holds_soup, jnp.zeros_like(state.pot_state[:, 2]),
                state.pot_state[:, 2],
            ) * pot_mask_i
        else:
            # Fixed: always show actual pot state regardless of held objects.
            onion_count = state.pot_state[:, 1] * pot_mask_i
            cook_time = state.pot_state[:, 2] * pot_mask_i
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
            tf_compat=tf_compat,
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
            tf_compat=tf_compat,
        )
        onion_layer = _object_layer_from_sources(
            shape=shape,
            counter_positions=terrain.counter_positions,
            counter_mask=terrain.counter_mask,
            counter_obj=state.counter_obj,
            obj_id=OBJ_ONION,
            player_pos=state.player_pos,
            held_obj=state.held_obj,
            tf_compat=tf_compat,
        )
        layers.append(dish_layer)
        layers.append(onion_layer)

        stacked = jnp.stack(layers, axis=0)
        # Keep legacy axis order: (W, H, C)
        return jnp.transpose(stacked, (1, 2, 0))

    return process(0), process(1)
