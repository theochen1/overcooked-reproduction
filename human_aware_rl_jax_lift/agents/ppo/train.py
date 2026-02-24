"""PPO training utilities."""

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import PPOConfig
from .model import ActorCriticCNN


def create_train_state(
    rng,
    obs_shape,
    config: PPOConfig,
    learning_rate: float | Callable[[int], float] | None = None,
) -> TrainState:
    model = ActorCriticCNN(
        num_actions=config.num_actions,
        num_filters=config.num_filters,
        hidden_dim=config.hidden_dim,
    )
    params = model.init(rng, jnp.zeros((1,) + obs_shape, dtype=jnp.float32))
    lr = config.learning_rate if learning_rate is None else learning_rate
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_gae(
    rewards,
    values,
    dones,
    gamma: float,
    lam: float,
    bootstrap_value: jnp.ndarray | None = None,
):
    """
    Compute generalized advantages with reverse-time scan.

    Inputs are time-major tensors [T, N] (or [T] for single-env).
    """
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=jnp.float32)

    if bootstrap_value is None:
        tail = jnp.zeros_like(values[-1:])
    else:
        tail = jnp.asarray(bootstrap_value, dtype=jnp.float32)[None, ...]
    next_values = jnp.concatenate([values[1:], tail], axis=0)
    not_done = 1.0 - dones
    deltas = rewards + gamma * next_values * not_done - values

    def scan_fn(carry, inputs):
        delta_t, not_done_t = inputs
        adv_t = delta_t + gamma * lam * not_done_t * carry
        return adv_t, adv_t

    _, adv_rev = jax.lax.scan(
        scan_fn,
        init=jnp.zeros_like(values[-1]),
        xs=(jnp.flip(deltas, axis=0), jnp.flip(not_done, axis=0)),
    )
    adv = jnp.flip(adv_rev, axis=0)
    returns = adv + values
    return adv, returns


@jax.jit
def _ppo_update_step_jit(
    state: TrainState,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_logp: jnp.ndarray,
    old_values: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    def loss_fn(params):
        logits, values = state.apply_fn(params, obs)
        logp_all = jax.nn.log_softmax(logits)
        logp = logp_all[jnp.arange(actions.shape[0]), actions]
        ratio = jnp.exp(logp - old_logp)

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        v_clip = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
        vf_loss1 = jnp.square(values - returns)
        vf_loss2 = jnp.square(v_clip - returns)
        critic_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

        entropy = -(jax.nn.softmax(logits) * logp_all).sum(axis=-1).mean()
        loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
        return loss, (actor_loss, critic_loss, entropy)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "actor_loss": aux[0], "critic_loss": aux[1], "entropy": aux[2]}


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
    return _ppo_update_step_jit(
        state=state,
        obs=obs,
        actions=actions,
        old_logp=old_logp,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        clip_eps=float(config.clip_eps),
        vf_coef=float(config.vf_coef),
        ent_coef=float(config.ent_coef),
    )
