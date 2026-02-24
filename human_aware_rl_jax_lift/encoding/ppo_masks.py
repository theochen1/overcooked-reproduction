"""Legacy 20-channel lossless encoding for PPO/CNN policies."""

import jax.numpy as jnp

from human_aware_rl_jax_lift.env.state import OBJ_DISH, OBJ_ONION, OBJ_SOUP, OvercookedState, Terrain, pos_to_yx


def _make_layer(shape, positions: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    layer = jnp.zeros(shape, dtype=jnp.int32)
    ys = positions[:, 1]
    xs = positions[:, 0]
    return layer.at[ys, xs].set(values)


def _player_orientation_layer(shape, pos: jnp.ndarray, orientation_idx: jnp.ndarray, target_idx: int) -> jnp.ndarray:
    v = jnp.where(orientation_idx == target_idx, 1, 0)
    return _make_layer(shape, pos[None, :], jnp.array([v], dtype=jnp.int32))


def lossless_state_encoding_20(terrain: Terrain, state: OvercookedState):
    """
    Return tuple(obs_for_player0, obs_for_player1), each HxWx20.
    Mirrors legacy `overcooked_mdp.lossless_state_encoding`.
    """
    h, w = terrain.grid.shape
    shape = (h, w)

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
        onion_count = state.pot_state[:, 1] * terrain.pot_mask.astype(jnp.int32)
        cook_time = state.pot_state[:, 2] * terrain.pot_mask.astype(jnp.int32)
        layers.append(_make_layer(shape, terrain.pot_positions, onion_count))
        layers.append(_make_layer(shape, terrain.pot_positions, cook_time))

        # 18 onion_soup_loc (soups not in pots)
        soup_positions = []
        soup_values = []
        for i in range(2):
            if state.held_obj[i] == OBJ_SOUP:
                soup_positions.append(state.player_pos[i])
                soup_values.append(jnp.array(1, dtype=jnp.int32))
        ctr_soup_mask = (state.counter_obj == OBJ_SOUP) & terrain.counter_mask
        ctr_soup_pos = terrain.counter_positions[ctr_soup_mask]
        if ctr_soup_pos.size > 0:
            for p in ctr_soup_pos:
                soup_positions.append(p)
                soup_values.append(jnp.array(1, dtype=jnp.int32))
        if soup_positions:
            layers.append(_make_layer(shape, jnp.stack(soup_positions), jnp.stack(soup_values)))
        else:
            layers.append(jnp.zeros(shape, dtype=jnp.int32))

        # 19 dishes, 20 onions
        dish_positions = []
        dish_values = []
        onion_positions = []
        onion_values = []
        ctr_dish_mask = (state.counter_obj == OBJ_DISH) & terrain.counter_mask
        ctr_onion_mask = (state.counter_obj == OBJ_ONION) & terrain.counter_mask
        if jnp.any(ctr_dish_mask):
            for p in terrain.counter_positions[ctr_dish_mask]:
                dish_positions.append(p)
                dish_values.append(jnp.array(1, dtype=jnp.int32))
        if jnp.any(ctr_onion_mask):
            for p in terrain.counter_positions[ctr_onion_mask]:
                onion_positions.append(p)
                onion_values.append(jnp.array(1, dtype=jnp.int32))
        for i in range(2):
            if state.held_obj[i] == OBJ_DISH:
                dish_positions.append(state.player_pos[i]); dish_values.append(jnp.array(1, dtype=jnp.int32))
            elif state.held_obj[i] == OBJ_ONION:
                onion_positions.append(state.player_pos[i]); onion_values.append(jnp.array(1, dtype=jnp.int32))
        layers.append(_make_layer(shape, jnp.stack(dish_positions), jnp.stack(dish_values)) if dish_positions else jnp.zeros(shape, dtype=jnp.int32))
        layers.append(_make_layer(shape, jnp.stack(onion_positions), jnp.stack(onion_values)) if onion_positions else jnp.zeros(shape, dtype=jnp.int32))

        stacked = jnp.stack(layers, axis=0)
        return jnp.transpose(stacked, (1, 2, 0))

    return process(0), process(1)
