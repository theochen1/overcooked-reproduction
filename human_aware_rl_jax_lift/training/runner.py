"""On-device rollout via jax.lax.scan — eliminates the Python step loop.

Supports self-play (other_agent = same policy) and BC partner (other_agent =
batched BC policy from featurize_state_64 + BCPolicy).

BC/self-play mixing semantics
----------------------------
The legacy TF/py runner treats mixing as an environment-level behavior controlled
by:
- self_play_randomization: probability of using self-play instead of BC
- trajectory_sp: if True, sample the choice once per episode trajectory (per env)
  and keep it fixed until reset; if False, sample independently per env per step.

This file implements the same semantics inside the scan path, and also ports the
legacy BC "unstuck" rule (stuck_time=3) to a pure-JAX partner state.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from human_aware_rl_jax_lift.agents.bc.model import BCPolicy
from human_aware_rl_jax_lift.encoding.bc_features import featurize_state_64
from human_aware_rl_jax_lift.env.state import Terrain

from .vec_env_jax import BatchedEnvState, batched_step, encode_obs


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
    """Ensure value head output is [N] for PPO/GAE."""
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[-1] == 1:
        return values[:, 0]
    raise ValueError(f"{name} must have shape [N] or [N,1], got {values.shape}")


def _init_bc_partner_state(num_envs: int):
    """Init JAX-carried state for BC unstuck rule."""
    pos_hist = jnp.zeros((num_envs, 4, 2), dtype=jnp.int32)
    act_hist = jnp.zeros((num_envs, 4), dtype=jnp.int32)
    hist_len = jnp.zeros((num_envs,), dtype=jnp.int32)
    use_sp_mask = jnp.zeros((num_envs,), dtype=jnp.bool_)
    return pos_hist, act_hist, hist_len, use_sp_mask


def _unstuck_adjust_probs(
    probs: jnp.ndarray,
    pos_hist: jnp.ndarray,
    act_hist: jnp.ndarray,
    hist_len: jnp.ndarray,
    *,
    stuck_time: int,
):
    """Vectorized port of BCAgent._unstuck_adjust (legacy)."""
    if stuck_time <= 0:
        return probs

    # Only apply once we have >= stuck_time+1 samples. Legacy stuck_time=3 => need 4.
    need = jnp.array(stuck_time + 1, dtype=jnp.int32)
    ready = hist_len >= need

    # same_pos over the whole ring buffer (len 4). Only meaningful when ready.
    p0 = pos_hist[:, 0, :]
    same_pos = jnp.all(pos_hist == p0[:, None, :], axis=(1, 2))
    apply = ready & same_pos

    # Block any actions that appeared in the recent history.
    # mask[a]=0 if action a appears in act_hist, else 1.
    one_hot = jax.nn.one_hot(act_hist, probs.shape[-1], dtype=jnp.float32)  # [N,4,A]
    blocked = jnp.clip(jnp.sum(one_hot, axis=1), 0.0, 1.0)                 # [N,A]
    mask = 1.0 - blocked

    out = probs * mask
    norm = jnp.sum(out, axis=-1, keepdims=True)
    renormed = jnp.where(norm > 0.0, out / norm, probs)
    return jnp.where(apply[:, None], renormed, probs)


def _push_history(
    pos_hist: jnp.ndarray,
    act_hist: jnp.ndarray,
    hist_len: jnp.ndarray,
    *,
    pos_xy: jnp.ndarray,
    act: jnp.ndarray,
    update_mask: jnp.ndarray,
):
    """Append (pos,act) into fixed-size ring buffers when update_mask is True."""
    # shift-left and append
    pos_shifted = jnp.concatenate([pos_hist[:, 1:, :], pos_xy[:, None, :]], axis=1)
    act_shifted = jnp.concatenate([act_hist[:, 1:], act[:, None]], axis=1)

    pos_hist = jnp.where(update_mask[:, None, None], pos_shifted, pos_hist)
    act_hist = jnp.where(update_mask[:, None], act_shifted, act_hist)

    new_len = jnp.minimum(hist_len + 1, jnp.array(4, dtype=jnp.int32))
    hist_len = jnp.where(update_mask, new_len, hist_len)
    return pos_hist, act_hist, hist_len


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
    bc_params: Optional[dict] = None,
    trajectory_sp: bool = True,
    bc_stuck_time: int = 3,
):
    """Build a rollout fn that is JIT-compiled and scan-based.

    Parameters
    ----------
    bc_params:
      If None, other agent is self-play.
      If provided, other agent is BC mixed with self-play via sp_factor.
    trajectory_sp:
      If True, sample SP-vs-BC choice once per env per episode trajectory.
      If False, sample independently per env per step.
    bc_stuck_time:
      Legacy unstuck rule parameter (default 3).
    """

    if bc_params is None:

        def _rollout(
            train_state: TrainState,
            bstate: BatchedEnvState,
            obs0: jnp.ndarray,           # [N, H, W, C]
            shaping_factor: jnp.ndarray, # scalar
            sp_factor: jnp.ndarray,      # scalar
            rng: jax.Array,
        ):
            def scan_step(carry, _):
                bstate, obs0, obs1, rng = carry
                rng, rng_train, rng_other, rng_reset = jax.random.split(rng, 4)

                logits, values = train_state.apply_fn(train_state.params, obs0)
                values = _value_to_1d(values, name="values")
                actions = jax.random.categorical(rng_train, logits).astype(jnp.int32)
                logp_all = jax.nn.log_softmax(logits)
                logp = logp_all[jnp.arange(num_envs), actions]

                logits_other, _ = train_state.apply_fn(train_state.params, obs1)
                other_actions = jax.random.categorical(rng_other, logits_other).astype(jnp.int32)

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

            obs0_f = obs0.astype(jnp.float32)
            obs0_init, obs1_init = encode_obs(terrain, bstate)
            obs0_init = obs0_f

            init_carry = (bstate, obs0_init, obs1_init, rng)
            (final_bstate, final_obs0, _final_obs1, _rng), transitions = jax.lax.scan(
                scan_step,
                init=init_carry,
                xs=None,
                length=horizon,
            )

            obs_t, actions_t, rewards_t, dones_t, values_t, logp_t, sparse_t = transitions

            bootstrap_obs = jnp.zeros_like(final_obs0) if bootstrap_with_zero_obs else final_obs0
            _, next_values = train_state.apply_fn(train_state.params, bootstrap_obs)
            next_value = _value_to_1d(next_values, name="next_values")

            return (
                obs_t, actions_t, rewards_t, dones_t, values_t, logp_t, sparse_t,
                next_value, final_bstate, final_obs0,
            )

    else:

        def _rollout(
            train_state: TrainState,
            bstate: BatchedEnvState,
            obs0: jnp.ndarray,
            shaping_factor: jnp.ndarray,
            sp_factor: jnp.ndarray,
            rng: jax.Array,
        ):
            pos_hist0, act_hist0, hist_len0, use_sp0 = _init_bc_partner_state(num_envs)

            def scan_step(carry, _):
                bstate, obs0, obs1, rng, pos_hist, act_hist, hist_len, use_sp_mask = carry
                rng, rng_train, rng_other_sp, rng_other_bc, rng_sp_mix, rng_reset = jax.random.split(rng, 6)

                # Episode reset indicator (env step resets state to timestep==0)
                reset_ep = (bstate.states.timestep == 0)

                # Reset BC partner history at episode boundaries (legacy behavior)
                zeros_pos = jnp.zeros_like(pos_hist)
                zeros_act = jnp.zeros_like(act_hist)
                pos_hist = jnp.where(reset_ep[:, None, None], zeros_pos, pos_hist)
                act_hist = jnp.where(reset_ep[:, None], zeros_act, act_hist)
                hist_len = jnp.where(reset_ep, jnp.zeros_like(hist_len), hist_len)

                # ---- Training-agent forward pass ----------------------------
                logits, values = train_state.apply_fn(train_state.params, obs0)
                values = _value_to_1d(values, name="values")
                actions = jax.random.categorical(rng_train, logits).astype(jnp.int32)
                logp_all = jax.nn.log_softmax(logits)
                logp = logp_all[jnp.arange(num_envs), actions]

                # ---- Other-agent SP -----------------------------------------
                logits_other, _ = train_state.apply_fn(train_state.params, obs1)
                other_actions_sp = jax.random.categorical(rng_other_sp, logits_other).astype(jnp.int32)

                # ---- Other-agent BC (batched features + legacy unstuck) ------
                f0, f1 = jax.vmap(featurize_state_64, in_axes=(None, 0))(terrain, bstate.states)
                other_feats = jnp.where(bstate.agent_idx[:, None] == 0, f1, f0).astype(jnp.float32)
                logits_bc = BCPolicy().apply(bc_params, other_feats)
                probs_bc = jax.nn.softmax(logits_bc, axis=-1)

                # Mixing semantics: environment-level (per env) and trajectory-level if requested.
                mix_u = jax.random.uniform(rng_sp_mix, (num_envs,))
                if trajectory_sp:
                    # sample at resets only; keep fixed within the episode trajectory
                    use_sp_mask = jnp.where(reset_ep, mix_u < sp_factor, use_sp_mask)
                    choose_sp = use_sp_mask
                else:
                    choose_sp = mix_u < sp_factor

                # Apply stuck detection only when we actually use BC for this env.
                use_bc = ~choose_sp
                probs_bc = _unstuck_adjust_probs(
                    probs_bc, pos_hist, act_hist, hist_len, stuck_time=int(bc_stuck_time)
                )

                logp_bc = jnp.log(probs_bc + 1e-20)
                other_actions_bc = jax.random.categorical(rng_other_bc, logp_bc).astype(jnp.int32)

                other_actions = jnp.where(choose_sp, other_actions_sp, other_actions_bc)

                # Update BC history only for envs using BC this step.
                other_idx = 1 - bstate.agent_idx  # [N]
                pos_xy = bstate.states.player_pos[jnp.arange(num_envs), other_idx]  # [N,2]
                pos_hist, act_hist, hist_len = _push_history(
                    pos_hist,
                    act_hist,
                    hist_len,
                    pos_xy=pos_xy,
                    act=other_actions_bc,
                    update_mask=use_bc,
                )

                # ---- Environment step --------------------------------------
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
                new_carry = (new_bstate, new_obs0, new_obs1, rng, pos_hist, act_hist, hist_len, use_sp_mask)
                return new_carry, transition

            obs0_f = obs0.astype(jnp.float32)
            obs0_init, obs1_init = encode_obs(terrain, bstate)
            obs0_init = obs0_f

            init_carry = (bstate, obs0_init, obs1_init, rng, pos_hist0, act_hist0, hist_len0, use_sp0)
            (final_bstate, final_obs0, _final_obs1, _rng, _ph, _ah, _hl, _us), transitions = jax.lax.scan(
                scan_step,
                init=init_carry,
                xs=None,
                length=horizon,
            )

            obs_t, actions_t, rewards_t, dones_t, values_t, logp_t, sparse_t = transitions

            bootstrap_obs = jnp.zeros_like(final_obs0) if bootstrap_with_zero_obs else final_obs0
            _, next_values = train_state.apply_fn(train_state.params, bootstrap_obs)
            next_value = _value_to_1d(next_values, name="next_values")

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
        sf = jnp.array(shaping_factor, dtype=jnp.float32)
        spf = jnp.array(sp_factor, dtype=jnp.float32)

        obs_t, actions_t, rewards_t, dones_t, values_t, logp_t, sparse_t, \
            next_value, final_bstate, final_obs0 = compiled(
                train_state, bstate, obs0, sf, spf, rng
            )

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
