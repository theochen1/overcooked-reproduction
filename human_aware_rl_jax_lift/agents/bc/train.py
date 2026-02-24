"""BC training loop with cross-entropy objective."""

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from .model import BCPolicy


@dataclass
class BCTrainConfig:
    num_epochs: int = 100
    learning_rate: float = 1e-3
    adam_eps: float = 1e-8
    batch_size: int = 64


def create_train_state(rng, input_dim: int, config: BCTrainConfig) -> TrainState:
    model = BCPolicy()
    params = model.init(rng, jnp.zeros((1, input_dim), dtype=jnp.float32))
    tx = optax.adam(learning_rate=config.learning_rate, eps=config.adam_eps)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        acc = (jnp.argmax(logits, axis=-1) == y).mean()
        return loss, acc

    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    next_state = state.apply_gradients(grads=grads)
    return next_state, {"loss": loss, "acc": acc}


def train_bc(features: jnp.ndarray, labels: jnp.ndarray, rng, config: BCTrainConfig) -> Dict[str, object]:
    """Train BC policy from feature/label tensors."""
    state = create_train_state(rng, features.shape[-1], config)
    n = features.shape[0]
    for _ in range(config.num_epochs):
        perm = np.random.permutation(n)
        features_epoch = features[perm]
        labels_epoch = labels[perm]
        for i in range(0, n, config.batch_size):
            xb = features_epoch[i : i + config.batch_size]
            yb = labels_epoch[i : i + config.batch_size]
            state, _ = train_step(state, xb, yb)
    return {"state": state}
