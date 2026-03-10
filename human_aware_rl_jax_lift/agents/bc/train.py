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
    train_fraction: float = 0.85


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
    """Train BC policy from feature/label tensors.

    Mirrors TF behavioural_cloning.py:
    - 85/15 train/val split (train_fraction=0.85)
    - Evaluates val loss after every epoch
    - Returns params from the epoch with lowest val loss (not final epoch)
    - Shuffle is driven by the provided JAX rng key for reproducibility
    """
    n = features.shape[0]
    n_train = int(n * config.train_fraction)

    # Shuffle once with JAX rng before splitting so val set is random
    rng, split_rng, init_rng = jax.random.split(rng, 3)
    perm = jax.random.permutation(split_rng, n)
    features = features[perm]
    labels = labels[perm]

    x_train, x_val = features[:n_train], features[n_train:]
    y_train, y_val = labels[:n_train], labels[n_train:]

    state = create_train_state(init_rng, features.shape[-1], config)

    # Close over BCPolicy().apply (a pure function) so jit can trace it.
    # Passing state.apply_fn (a bound method) as a traced arg fails.
    _apply = BCPolicy().apply

    @jax.jit
    def _val_loss(params, x, y):
        logits = _apply(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    best_val_loss = float("inf")
    best_params = state.params

    for epoch in range(config.num_epochs):
        # Per-epoch shuffle of training set using a derived key
        rng, epoch_rng = jax.random.split(rng)
        epoch_perm = jax.random.permutation(epoch_rng, n_train)
        x_epoch = x_train[epoch_perm]
        y_epoch = y_train[epoch_perm]

        for i in range(0, n_train, config.batch_size):
            xb = x_epoch[i : i + config.batch_size]
            yb = y_epoch[i : i + config.batch_size]
            state, _ = train_step(state, xb, yb)

        # Evaluate on validation set and checkpoint if best so far
        epoch_val_loss = float(_val_loss(state.params, x_val, y_val))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_params = state.params

    return {"params": best_params, "best_val_loss": best_val_loss}
