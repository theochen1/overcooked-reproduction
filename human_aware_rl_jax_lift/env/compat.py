"""Compatibility helpers against legacy python env objects."""

from typing import Tuple

import jax.numpy as jnp

from .state import (
    OBJ_DISH,
    OBJ_ONION,
    OBJ_SOUP,
    OBJ_TOMATO,
    SOUP_ONION,
    SOUP_TOMATO,
    OvercookedState,
    Terrain,
)


def _obj_name_to_id(name: str) -> int:
    if name == "onion":
        return OBJ_ONION
    if name == "tomato":
        return OBJ_TOMATO
    if name == "dish":
        return OBJ_DISH
    if name == "soup":
        return OBJ_SOUP
    return 0


def _soup_type_to_id(name: str) -> int:
    return SOUP_ONION if name == "onion" else SOUP_TOMATO


def from_legacy_state(terrain: Terrain, legacy_state) -> OvercookedState:
    """Convert legacy `OvercookedState` object to static JAX state."""
    from .state import make_initial_state

    st = make_initial_state(terrain)
    player_pos = []
    player_or = []
    held_obj = []
    held_soup = []
    for p in legacy_state.players:
        player_pos.append(p.position)
        player_or.append({(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}[p.orientation])
        if p.held_object is None:
            held_obj.append(0)
            held_soup.append((0, 0, 0))
        elif p.held_object.name == "soup":
            soup_type, n, c = p.held_object.state
            held_obj.append(OBJ_SOUP)
            held_soup.append((_soup_type_to_id(soup_type), n, c))
        else:
            held_obj.append(_obj_name_to_id(p.held_object.name))
            held_soup.append((0, 0, 0))

    st = st.replace(
        player_pos=jnp.array(player_pos, dtype=jnp.int32),
        player_or=jnp.array(player_or, dtype=jnp.int32),
        held_obj=jnp.array(held_obj, dtype=jnp.int32),
        held_soup=jnp.array(held_soup, dtype=jnp.int32),
    )

    # Terrain index maps for fixed-size arrays.
    pot_index = {
        tuple(jnp.asarray(terrain.pot_positions[i]).tolist()): i
        for i, m in enumerate(jnp.asarray(terrain.pot_mask).tolist())
        if m
    }
    counter_index = {
        tuple(jnp.asarray(terrain.counter_positions[i]).tolist()): i
        for i, m in enumerate(jnp.asarray(terrain.counter_mask).tolist())
        if m
    }

    pot_state = st.pot_state
    counter_obj = st.counter_obj
    counter_soup = st.counter_soup
    for obj in legacy_state.objects.values():
        pos = tuple(obj.position)
        if obj.name == "soup":
            soup_type, n, c = obj.state
            payload = jnp.array([_soup_type_to_id(soup_type), int(n), int(c)], dtype=jnp.int32)
            if pos in pot_index:
                pot_state = pot_state.at[pot_index[pos]].set(payload)
            elif pos in counter_index:
                idx = counter_index[pos]
                counter_obj = counter_obj.at[idx].set(OBJ_SOUP)
                counter_soup = counter_soup.at[idx].set(payload)
        elif pos in counter_index:
            idx = counter_index[pos]
            counter_obj = counter_obj.at[idx].set(_obj_name_to_id(obj.name))

    st = st.replace(pot_state=pot_state, counter_obj=counter_obj, counter_soup=counter_soup)
    return st
