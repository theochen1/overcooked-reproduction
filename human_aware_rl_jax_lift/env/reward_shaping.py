"""Reward shaping utilities matching legacy parameters."""

import jax.numpy as jnp


DEFAULT_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3.0,
    "DISH_PICKUP_REWARD": 3.0,
    "SOUP_PICKUP_REWARD": 5.0,
    "DISH_DISP_DISTANCE_REW": 0.0,
    "POT_DISTANCE_REW": 0.0,
    "SOUP_DISTANCE_REW": 0.0,
}


def annealed_shaping_factor(initial: float, horizon: float, t: jnp.ndarray) -> jnp.ndarray:
    """Legacy-style linear annealing to zero by `horizon`."""
    if horizon <= 0:
        return jnp.array(initial, dtype=jnp.float32)
    frac = jnp.maximum(1.0 - (t.astype(jnp.float32) / float(horizon)), 0.0)
    return jnp.array(initial, dtype=jnp.float32) * frac
