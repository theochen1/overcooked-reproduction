"""Flax BC model: 2-layer MLP with hidden size 64."""

import flax.linen as nn
import jax.numpy as jnp


class BCPolicy(nn.Module):
    num_actions: int = 6
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        return nn.Dense(self.num_actions)(x)
