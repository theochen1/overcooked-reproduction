"""Flax BC model: 2-layer MLP with hidden size 64."""

import flax.linen as nn
import jax.numpy as jnp

# TF uses ortho_init(scale=sqrt(2)) for hidden layers (a2c/utils.py:57),
# and ortho_init(scale=0.01) for the output (logits) layer
# (distributions.py:168 → linear('pi', 6, init_scale=0.01)).
_ORTHO_INIT = nn.initializers.orthogonal(scale=jnp.sqrt(2.0))
_ORTHO_INIT_SMALL = nn.initializers.orthogonal(scale=0.01)
_ZEROS_INIT = nn.initializers.zeros


class BCPolicy(nn.Module):
    num_actions: int = 6
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTHO_INIT, bias_init=_ZEROS_INIT)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTHO_INIT, bias_init=_ZEROS_INIT)(x)
        x = nn.tanh(x)
        return nn.Dense(self.num_actions, kernel_init=_ORTHO_INIT_SMALL, bias_init=_ZEROS_INIT)(x)
