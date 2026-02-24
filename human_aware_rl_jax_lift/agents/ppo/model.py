"""CNN actor-critic architecture matching legacy conv_and_mlp."""

import flax.linen as nn
import jax.numpy as jnp


class ActorCriticCNN(nn.Module):
    num_actions: int = 6
    num_filters: int = 25
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Conv(self.num_filters, (5, 5), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(self.num_filters, (3, 3), padding="SAME")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Conv(self.num_filters, (3, 3), padding="VALID")(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value
