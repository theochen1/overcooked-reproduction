"""Deterministic seeding helpers."""

import random

import jax
import numpy as np


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)
