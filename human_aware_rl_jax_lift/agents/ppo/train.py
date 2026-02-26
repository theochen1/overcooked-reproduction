"""PPO training utilities."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import PPOConfig
from .model import ActorCriticCNN


def _trunk_sq_norm(params_tree) -> jnp.ndarray:
    p = params_tree["params"]
    sq = jnp.array(0.0, dtype=jnp.float32)
    for i in range(3):
        sq = sq + jnp.sum(jnp.square(p[f"Conv_{i}"]["kernel"]))
        sq = sq + jnp.sum(jnp.square(p[f"Conv_{i}"]["bias"]))
        sq = sq + jnp.sum(jnp.square(p[f"Dense_{i}"]["kernel"]))
        sq = sq + jnp.sum(jnp.square(p[f"Dense_{i}"]["bias"]))
    return sq


def _trunk_dot(a_tree, b_tree) -> jnp.ndarray:
    a = a_tree["params"]
    b = b_tree["params"]
    dot = jnp.array(0.0, dtype=jnp.float32)
    for i in range(3):
        dot = dot + jnp.sum(a[f"Conv_{i}"]["kernel"] * b[f"Conv_{i}"]["kernel"])
        dot = dot + jnp.sum(a[f"Conv_{i}"]["bias"] * b[f"Conv_{i}"]["bias"])
        dot = dot + jnp.sum(a[f"Dense_{i}"]["kernel"] * b[f"Dense_{i}"]["kernel"])
        dot = dot + jnp.sum(a[f"Dense_{i}"]["bias"] * b[f"Dense_{i}"]["bias"])
    return dot


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
        optax.adam(lr, eps=1e-5),  # Match TF Baselines default
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
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
    JIT-compiled for fast execution on GPU.
    Inputs are time-major tensors [T, N] (or [T] for single-env).
    """
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values  = jnp.asarray(values,  dtype=jnp.float32)
    dones   = jnp.asarray(dones,   dtype=jnp.float32)

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


@partial(jax.jit, static_argnames=("normalize_advantages", "compute_trunk_grad_decomp"))
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
    max_grad_norm: float,
    normalize_advantages: bool,
    compute_trunk_grad_decomp: bool,
):
    def loss_fn(params):
        logits, values = state.apply_fn(params, obs)
        logp_all = jax.nn.log_softmax(logits)
        logp = logp_all[jnp.arange(actions.shape[0]), actions]
        ratio = jnp.exp(logp - old_logp)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        adv = jax.lax.cond(
            jnp.asarray(normalize_advantages),
            lambda a: (a - adv_mean) / adv_std,
            lambda a: a,
            advantages,
        )
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        v_clip = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
        vf_loss1 = jnp.square(values - returns)
        vf_loss2 = jnp.square(v_clip - returns)
        critic_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

        entropy = -(jax.nn.softmax(logits) * logp_all).sum(axis=-1).mean()
        loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

        # Diagnostics matching Baselines logging convention
        approxkl  = 0.5 * jnp.mean(jnp.square(old_logp - logp))
        clipfrac  = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

        return loss, (
            actor_loss,
            critic_loss,
            entropy,
            approxkl,
            clipfrac,
            adv_mean,
            adv_std,
            advantages.min(),
            advantages.max(),
            adv.mean(),
            adv.std(),
            adv.min(),
            adv.max(),
        )

    old_params = state.params
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(old_params)
    trunk_grad_norm_total = jnp.array(0.0, dtype=jnp.float32)
    trunk_grad_norm_actor = jnp.array(0.0, dtype=jnp.float32)
    trunk_grad_norm_critic = jnp.array(0.0, dtype=jnp.float32)
    trunk_grad_cos_actor_critic = jnp.array(0.0, dtype=jnp.float32)
    trunk_grad_actor_share_total = jnp.array(0.0, dtype=jnp.float32)
    trunk_grad_critic_share_total = jnp.array(0.0, dtype=jnp.float32)
    if compute_trunk_grad_decomp:
        def actor_component_loss_fn(params):
            logits, values = state.apply_fn(params, obs)
            logp_all = jax.nn.log_softmax(logits)
            logp = logp_all[jnp.arange(actions.shape[0]), actions]
            ratio = jnp.exp(logp - old_logp)
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            adv = jax.lax.cond(
                jnp.asarray(normalize_advantages),
                lambda a: (a - adv_mean) / adv_std,
                lambda a: a,
                advantages,
            )
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            entropy = -(jax.nn.softmax(logits) * logp_all).sum(axis=-1).mean()
            return actor_loss - ent_coef * entropy

        def critic_component_loss_fn(params):
            logits, values = state.apply_fn(params, obs)
            _ = logits
            v_clip = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
            vf_loss1 = jnp.square(values - returns)
            vf_loss2 = jnp.square(v_clip - returns)
            critic_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()
            return vf_coef * critic_loss

        actor_grads = jax.grad(actor_component_loss_fn)(old_params)
        critic_grads = jax.grad(critic_component_loss_fn)(old_params)

        trunk_sq_total = _trunk_sq_norm(grads)
        trunk_sq_actor = _trunk_sq_norm(actor_grads)
        trunk_sq_critic = _trunk_sq_norm(critic_grads)
        trunk_dot_actor_critic = _trunk_dot(actor_grads, critic_grads)
        trunk_grad_norm_total = jnp.sqrt(trunk_sq_total)
        trunk_grad_norm_actor = jnp.sqrt(trunk_sq_actor)
        trunk_grad_norm_critic = jnp.sqrt(trunk_sq_critic)
        trunk_grad_cos_actor_critic = trunk_dot_actor_critic / (
            trunk_grad_norm_actor * trunk_grad_norm_critic + 1e-12
        )
        trunk_grad_actor_share_total = trunk_grad_norm_actor / (trunk_grad_norm_total + 1e-12)
        trunk_grad_critic_share_total = trunk_grad_norm_critic / (trunk_grad_norm_total + 1e-12)

    state = state.apply_gradients(grads=grads)
    delta_params = jax.tree_util.tree_map(lambda new, old: new - old, state.params, old_params)
    (
        actor_loss,
        critic_loss,
        entropy,
        approxkl,
        clipfrac,
        adv_mean,
        adv_std,
        adv_min,
        adv_max,
        adv_norm_mean,
        adv_norm_std,
        adv_norm_min,
        adv_norm_max,
    ) = aux
    params_grads = grads["params"]
    conv_grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(params_grads[f"Conv_{i}"]["kernel"])) +
        jnp.sum(jnp.square(params_grads[f"Conv_{i}"]["bias"]))
        for i in range(3)
    ))
    dense_grad_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(params_grads[f"Dense_{i}"]["kernel"])) +
        jnp.sum(jnp.square(params_grads[f"Dense_{i}"]["bias"]))
        for i in range(3)
    ))
    policy_head_grad_norm = jnp.sqrt(
        jnp.sum(jnp.square(params_grads["Dense_3"]["kernel"])) +
        jnp.sum(jnp.square(params_grads["Dense_3"]["bias"]))
    )
    value_head_grad_norm = jnp.sqrt(
        jnp.sum(jnp.square(params_grads["Dense_4"]["kernel"])) +
        jnp.sum(jnp.square(params_grads["Dense_4"]["bias"]))
    )
    params_delta = delta_params["params"]
    delta_norm_global = optax.global_norm(delta_params)
    conv_delta_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(params_delta[f"Conv_{i}"]["kernel"])) +
        jnp.sum(jnp.square(params_delta[f"Conv_{i}"]["bias"]))
        for i in range(3)
    ))
    dense_delta_norm = jnp.sqrt(sum(
        jnp.sum(jnp.square(params_delta[f"Dense_{i}"]["kernel"])) +
        jnp.sum(jnp.square(params_delta[f"Dense_{i}"]["bias"]))
        for i in range(3)
    ))
    policy_head_delta_norm = jnp.sqrt(
        jnp.sum(jnp.square(params_delta["Dense_3"]["kernel"])) +
        jnp.sum(jnp.square(params_delta["Dense_3"]["bias"]))
    )
    value_head_delta_norm = jnp.sqrt(
        jnp.sum(jnp.square(params_delta["Dense_4"]["kernel"])) +
        jnp.sum(jnp.square(params_delta["Dense_4"]["bias"]))
    )
    vf_loss_scaled = vf_coef * critic_loss
    entropy_bonus_scaled = ent_coef * entropy
    grad_norm_global = optax.global_norm(grads)
    grad_clip_coef = jnp.minimum(1.0, max_grad_norm / (grad_norm_global + 1e-12))
    return state, {
        "loss":         loss,
        "policy_loss":  actor_loss,   # match Baselines key names
        "value_loss":   critic_loss,
        "entropy":      entropy,
        "approxkl":     approxkl,
        "clipfrac":     clipfrac,
        "grad_norm_global": grad_norm_global,
        "grad_clip_coef": grad_clip_coef,
        "grad_norm_conv": conv_grad_norm,
        "grad_norm_dense": dense_grad_norm,
        "grad_norm_policy_head": policy_head_grad_norm,
        "grad_norm_value_head": value_head_grad_norm,
        "delta_norm_global": delta_norm_global,
        "delta_norm_conv": conv_delta_norm,
        "delta_norm_dense": dense_delta_norm,
        "delta_norm_policy_head": policy_head_delta_norm,
        "delta_norm_value_head": value_head_delta_norm,
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "adv_min": adv_min,
        "adv_max": adv_max,
        "adv_norm_mean": adv_norm_mean,
        "adv_norm_std": adv_norm_std,
        "adv_norm_min": adv_norm_min,
        "adv_norm_max": adv_norm_max,
        "loss_component_actor": actor_loss,
        "loss_component_value_scaled": vf_loss_scaled,
        "loss_component_entropy_scaled": entropy_bonus_scaled,
        "trunk_grad_norm_total": trunk_grad_norm_total,
        "trunk_grad_norm_actor": trunk_grad_norm_actor,
        "trunk_grad_norm_critic": trunk_grad_norm_critic,
        "trunk_grad_cos_actor_critic": trunk_grad_cos_actor_critic,
        "trunk_grad_actor_share_total": trunk_grad_actor_share_total,
        "trunk_grad_critic_share_total": trunk_grad_critic_share_total,
    }


def ppo_update_step(
    state: TrainState,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_logp: jnp.ndarray,
    old_values: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    config: PPOConfig,
    *,
    normalize_advantages: bool = True,
    compute_trunk_grad_decomp: bool = False,
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
        max_grad_norm=float(config.max_grad_norm),
        normalize_advantages=bool(normalize_advantages),
        compute_trunk_grad_decomp=bool(compute_trunk_grad_decomp),
    )
