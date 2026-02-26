"""On-device rollout via jax.lax.scan — eliminates the Python step loop.

Previous design (runner.py)
---------------------------
  for step in range(horizon):            # 400 Python iterations
      actions = policy_step(obs)         # GPU->CPU: actions, values, logp
      other  = partner.act(obs1)         # GPU->CPU
      step_out = vec_env.step_all(...)   # 90 device-to-host syncs

This design
-----------
  rollout_data, final_state = jax.lax.scan(scan_step, init, None, horizon)

Everything — forward pass, env step, obs encoding — stays on device for the
entire horizon.  The only host transfer is the single ``jnp.asarray`` of the
numpy obs at the very start, and the final readout of the rollout arrays for
GAE / PPO update.

Self-play note
--------------
BCPartner uses a Python loop internally (stuck detection, featurize_state_64)
and cannot be scanned.  For now ``make_rollout_fn`` only supports self-play
(``other_agent_type='sp'``) inside the scan.  BC-partner rollouts fall back
to the old runner.py automatically via ``ppo_run.py``.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from .vec_env_jax import BatchedEnvState, batched_step, encode_obs
from human_aware_rl_jax_lift.env.state import Terrain


# ---------------------------------------------------------------------------
# Output container (mirrors RolloutBatch in runner.py for drop-in use)
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    obs: jnp.ndarray         # [T, N, H, W, C]
    actions: jnp.ndarray     # [T, N]
    rewards: jnp.ndarray     # [T, N]
    dones: jnp.ndarray       # [T, N]
    values: jnp.ndarray      # [T, N]
    log_probs: jnp.ndarray   # [T, N]
    next_value: jnp.ndarray  # [N]
    sparse_rewards: jnp.ndarray  # [T, N]  sparse component for logging
    infos: dict              # scalar summary stats (populated after scan)


def _value_to_1d(values: jnp.ndarray, *, name: str) -> jnp.ndarray:
    """Ensure value head output is [N] for PPO/GAE.

    Some apply_fn implementations return [N] while others return [N, 1].
    This normalizes both cases without using squeeze on a non-singleton axis.
    """
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[-1] == 1:
        return values[:, 0]
    # Static-shape check at trace time (helps catch accidental multi-head values)
    raise ValueError(f"{name} must have shape [N] or [N,1], got {values.shape}")


# ---------------------------------------------------------------------------
# Scan-based rollout builder
# ---------------------------------------------------------------------------

def make_rollout_fn(
    terrain: Terrain,
    horizon: int,
    num_envs: int,
    *,
    randomize_agent_idx: bool = False,
    bootstrap_with_zero_obs: bool = False,
):
    """
    Return a JIT-compiled function::

        rollout, final_bstate, final_obs0 = fn(
            train_state, bstate, obs0, shaping_factor, sp_factor, rng
        )

    Parameters
    ----------
    terrain        : static Terrain pytree (treated as static by XLA)
    horizon        : rollout length (static — determines scan unroll count)
    num_envs       : number of parallel envs (static)

    The returned function is suitable for both the first call (which triggers
    XLA compilation) and all subsequent calls (which reuse the compiled kernel).
    """

    def _rollout(
        train_state: TrainState,
        bstate: BatchedEnvState,
        obs0: jnp.ndarray,           # [N, H, W, C]
        shaping_factor: jnp.ndarray, # scalar
        sp_factor: jnp.ndarray,      # scalar — fraction of other-agent SP steps
        rng: jax.Array,
    ):
        def scan_step(carry, _):
            bstate, obs0, obs1, rng = carry
            rng, rng_train, rng_other, rng_sp_mix, rng_reset = jax.random.split(rng, 5)

            # ---- Training-agent forward pass --------------------------------
            logits, values = train_state.apply_fn(train_state.params, obs0)
            values = _value_to_1d(values, name="values")  # [N]
            actions = jax.random.categorical(rng_train, logits).astype(jnp.int32)
            logp_all = jax.nn.log_softmax(logits)
            logp = logp_all[jnp.arange(num_envs), actions]  # [N]

            # ---- Other-agent forward pass (self-play) -----------------------
            # sp_factor controls mix: 1.0 = pure SP, 0.0 = always use obs1
            # (for BC-partner support, fall back to old runner; see docstring)
            logits_other, _ = train_state.apply_fn(train_state.params, obs1)
            other_actions_sp = jax.random.categorical(rng_other, logits_other).astype(jnp.int32)
            # When sp_factor < 1 we could blend with a BC policy here;
            # for now, pure self-play: other_actions = sp actions.
            other_actions = other_actions_sp

            # ---- Environment step (vmapped, on-device) ----------------------
            reset_keys = jax.random.split(rng_reset, num_envs)
            new_bstate, new_obs0, new_obs1, rewards, dones, sparse_r = batched_step(
                terrain,
                bstate,
                actions,
                other_actions,
                reset_keys,
                shaping_factor,
                horizon,
                player_order_actions=False,
                randomize_agent_idx=randomize_agent_idx,
            )

            transition = (obs0, actions, rewards, dones, values, logp, sparse_r)
            return (new_bstate, new_obs0, new_obs1, rng), transition

        # Encode initial observations (obs1 needed for other-agent pass)
        obs0_f = obs0.astype(jnp.float32)
        obs1_init: jnp.ndarray
        obs0_init, obs1_init = encode_obs(terrain, bstate)  # [N, H, W, C]
        obs0_init = obs0_f  # use passed-in obs0 (already correct)

        init_carry = (bstate, obs0_init, obs1_init, rng)
        (final_bstate, final_obs0, _final_obs1, _rng), transitions = jax.lax.scan(
            scan_step,
            init=init_carry,
            xs=None,
            length=horizon,
        )

        obs_t, actions_t, rewards_t, dones_t, values_t, logp_t, sparse_t = transitions
        # obs_t: [T, N, H, W, C], etc.

        # Bootstrap value for GAE
        bootstrap_obs = jnp.zeros_like(final_obs0) if bootstrap_with_zero_obs else final_obs0
        _, next_values = train_state.apply_fn(train_state.params, bootstrap_obs)
        next_value = _value_to_1d(next_values, name="next_values")  # [N]

        return (
            obs_t, actions_t, rewards_t, dones_t, values_t, logp_t, sparse_t,
            next_value, final_bstate, final_obs0,
        )

    compiled = jax.jit(_rollout)

    def rollout_fn(
        train_state: TrainState,
        bstate: BatchedEnvState,
        obs0: jnp.ndarray,
        shaping_factor: float,
        sp_factor: float,
        rng: jax.Array,
    ) -> tuple[RolloutBatch, BatchedEnvState, jnp.ndarray]:
        """Public wrapper: calls compiled scan, packages into RolloutBatch."""
        sf = jnp.array(shaping_factor, dtype=jnp.float32)
        spf = jnp.array(sp_factor, dtype=jnp.float32)

        obs_t, actions_t, rewards_t, dones_t, values_t, logp_t, sparse_t, \
            next_value, final_bstate, final_obs0 = compiled(
                train_state, bstate, obs0, sf, spf, rng
            )

        # -------------------------------------------------------------------
        # Episode stats (match legacy runner.py semantics)
        # - Legacy logs mean over completed episodes of total return "episode['r']"
        # - Here we reconstruct episode returns by accumulating reward until done.
        # -------------------------------------------------------------------
        dones_f = dones_t.astype(jnp.float32)

        def _acc_step(carry, x):
            acc_r, acc_sr = carry
            r, sr, d = x
            acc_r = acc_r + r
            acc_sr = acc_sr + sr
            ep_r = jnp.where(d > 0.0, acc_r, 0.0)
            ep_sr = jnp.where(d > 0.0, acc_sr, 0.0)
            acc_r = jnp.where(d > 0.0, 0.0, acc_r)
            acc_sr = jnp.where(d > 0.0, 0.0, acc_sr)
            return (acc_r, acc_sr), (ep_r, ep_sr)

        (_, _), (ep_r_t, ep_sr_t) = jax.lax.scan(
            _acc_step,
            init=(
                jnp.zeros((num_envs,), dtype=rewards_t.dtype),
                jnp.zeros((num_envs,), dtype=sparse_t.dtype),
            ),
            xs=(rewards_t, sparse_t, dones_f),
        )

        episodes = jnp.sum(dones_f)
        eprewmean = jnp.where(episodes > 0.0, jnp.sum(ep_r_t) / episodes, 0.0)
        ep_sparse_mean = jnp.where(episodes > 0.0, jnp.sum(ep_sr_t) / episodes, 0.0)
        dones_np = np.asarray(dones_f)
        ep_r_np = np.asarray(ep_r_t)
        ep_sr_np = np.asarray(ep_sr_t)
        completed_eprew = ep_r_np[dones_np > 0.0]
        completed_sparse = ep_sr_np[dones_np > 0.0]

        batch = RolloutBatch(
            obs=obs_t,
            actions=actions_t,
            rewards=rewards_t,
            dones=dones_t,
            values=values_t,
            log_probs=logp_t,
            next_value=next_value,
            sparse_rewards=sparse_t,
            infos={
                "eprewmean": float(eprewmean),
                "ep_sparse_rew_mean": float(ep_sparse_mean),
                "episodes_this_rollout": int(np.asarray(episodes)),
                "completed_eprew": completed_eprew,
                "completed_ep_sparse_rew": completed_sparse,
            },
        )
        return batch, final_bstate, final_obs0

    return rollout_fn
