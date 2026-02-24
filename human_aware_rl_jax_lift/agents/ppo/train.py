"""PPO training utilities."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import PPOConfig
from .model import ActorCriticCNN


def create_train_state(rng, obs_shape, config: PPOConfig) -> TrainState:
    model = ActorCriticCNN(
        num_actions=config.num_actions,
        num_filters=config.num_filters,
        hidden_dim=config.hidden_dim,
    )
    params = model.init(rng, jnp.zeros((1,) + obs_shape, dtype=jnp.float32))
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    adv = jnp.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        next_value = jnp.where(t + 1 < values.shape[0], values[t + 1], 0.0)
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv = adv.at[t].set(gae)
    return adv, adv + values


def ppo_update_step(
    state: TrainState,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_logp: jnp.ndarray,
    old_values: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    config: PPOConfig,
):
    def loss_fn(params):
        logits, values = state.apply_fn(params, obs)
        logp_all = jax.nn.log_softmax(logits)
        logp = logp_all[jnp.arange(actions.shape[0]), actions]
        ratio = jnp.exp(logp - old_logp)

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        v_clip = old_values + jnp.clip(values - old_values, -config.clip_eps, config.clip_eps)
        vf_loss1 = jnp.square(values - returns)
        vf_loss2 = jnp.square(v_clip - returns)
        critic_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

        entropy = -(jax.nn.softmax(logits) * logp_all).sum(axis=-1).mean()
        loss = actor_loss + config.vf_coef * critic_loss - config.ent_coef * entropy
        return loss, (actor_loss, critic_loss, entropy)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "actor_loss": aux[0], "critic_loss": aux[1], "entropy": aux[2]}
