"""Collision logic for two-player Overcooked dynamics."""

import jax.numpy as jnp


def has_same_cell_collision(new_pos: jnp.ndarray) -> jnp.ndarray:
    """Return True when both players land on the same position."""
    return jnp.all(new_pos[0] == new_pos[1])


def has_swap_collision(old_pos: jnp.ndarray, new_pos: jnp.ndarray) -> jnp.ndarray:
    """Return True when players swap positions in one step."""
    return jnp.all(new_pos[0] == old_pos[1]) & jnp.all(new_pos[1] == old_pos[0])


def resolve_player_collisions(old_pos: jnp.ndarray, new_pos: jnp.ndarray) -> jnp.ndarray:
    """Legacy collision semantics: on collision, both players stay put."""
    collided = has_same_cell_collision(new_pos) | has_swap_collision(old_pos, new_pos)
    return jnp.where(collided, old_pos, new_pos)
