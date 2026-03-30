"""CNN actor-critic architecture matching legacy conv_and_mlp."""

import functools

import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, orthogonal


_glorot = glorot_uniform()
_ortho_pi = orthogonal(scale=0.01)
_ortho_vf = orthogonal(scale=1.0)
_lrelu = functools.partial(nn.leaky_relu, negative_slope=0.2)


class ActorCriticCNN(nn.Module):
    num_actions: int = 6
    num_filters: int = 25
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Conv(self.num_filters, (5, 5), padding="SAME",
                     kernel_init=_glorot)(x)
        x = _lrelu(x)
        x = nn.Conv(self.num_filters, (3, 3), padding="SAME",
                     kernel_init=_glorot)(x)
        x = _lrelu(x)
        x = nn.Conv(self.num_filters, (3, 3), padding="VALID",
                     kernel_init=_glorot)(x)
        x = _lrelu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim, kernel_init=_glorot)(x)
        x = _lrelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_glorot)(x)
        x = _lrelu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_glorot)(x)
        x = _lrelu(x)
        logits = nn.Dense(self.num_actions, kernel_init=_ortho_pi)(x)
        value = nn.Dense(1, kernel_init=_ortho_vf)(x).squeeze(-1)
        return logits, value
