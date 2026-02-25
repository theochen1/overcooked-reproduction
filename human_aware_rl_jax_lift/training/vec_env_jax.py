"""Fully-JAX vectorized env: batched OvercookedState pytree + vmap(env_step).

Replaces the serial Python for-loop in vec_env.py.  Because OvercookedState
is a @flax.struct.dataclass (a JAX pytree) every field already has a well-
defined batch dimension after vmap — no state refactoring required.

Key design choices
------------------
* ``make_batched_state`` builds a batched pytree with a leading [N] dim on
  every state field by vmapping ``make_initial_state`` over dummy indices.
* ``batched_step`` is ``@jax.jit`` and calls ``jax.vmap(_single_step)``.
  Each vmap'd call is a pure function: no Python conditionals on JAX values.
* Episode resets are handled *inside* vmap with ``jax.lax.cond`` /
  ``jnp.where`` so the XLA program stays static-shape.
* The ``shaping_factor`` scalar is passed as a JAX array so it can be
  updated between updates without recompilation (not a static arg).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

from human_aware_rl_jax_lift.encoding.ppo_masks import lossless_state_encoding_20
from human_aware_rl_jax_lift.env.overcooked_mdp import step as env_step
from human_aware_rl_jax_lift.env.state import OvercookedState, Terrain, make_initial_state


# ---------------------------------------------------------------------------
# Batched state container
# ---------------------------------------------------------------------------

@struct.dataclass
class BatchedEnvState:
    """Holds N OvercookedStates stacked along a leading batch dimension."""
    states: OvercookedState        # every field: [N, ...]
    agent_idx: jnp.ndarray         # [N] int32 — which player is the training agent
    ep_sparse_accum: jnp.ndarray   # [N] float32
    ep_shaped_accum: jnp.ndarray   # [N] float32


def make_batched_state(terrain: Terrain, num_envs: int, rng: jax.Array) -> BatchedEnvState:
    """Construct a BatchedEnvState with N independent initial states."""
    # vmap over a dummy index array — all envs start with the same initial
    # geometry but different RNG keys for agent_idx assignment.
    init_states = jax.vmap(lambda _: make_initial_state(terrain))(jnp.arange(num_envs))
    agent_idx = jax.random.randint(rng, (num_envs,), 0, 2, dtype=jnp.int32)
    return BatchedEnvState(
        states=init_states,
        agent_idx=agent_idx,
        ep_sparse_accum=jnp.zeros(num_envs, dtype=jnp.float32),
        ep_shaped_accum=jnp.zeros(num_envs, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Observation encoding (vmap-able, pure JAX)
# ---------------------------------------------------------------------------

def _encode_single(terrain: Terrain, state: OvercookedState, agent_idx: jnp.ndarray):
    """Encode one state; returns (obs_training_agent, obs_other_agent)."""
    p0_obs, p1_obs = lossless_state_encoding_20(terrain, state)
    # No Python if/else — use jnp.where on every element so XLA can lower it.
    obs0 = jnp.where(agent_idx == 0, p0_obs, p1_obs)
    obs1 = jnp.where(agent_idx == 0, p1_obs, p0_obs)
    return obs0.astype(jnp.float32), obs1.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Single-env step (will be vmapped)
# ---------------------------------------------------------------------------

def _single_step(
    terrain: Terrain,
    state: OvercookedState,
    agent_idx: jnp.ndarray,
    training_action: jnp.ndarray,
    other_action: jnp.ndarray,
    ep_sparse: jnp.ndarray,
    ep_shaped: jnp.ndarray,
    shaping_factor: jnp.ndarray,
    horizon: int,
) -> Tuple[OvercookedState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure transition for one environment; safe to vmap."""
    # Build joint action respecting which player is the training agent.
    # Use jnp.stack + indexing to avoid Python branching.
    actions_if_0 = jnp.stack([training_action, other_action])   # [ta, oa]
    actions_if_1 = jnp.stack([other_action, training_action])   # [oa, ta]
    joint = jnp.where(agent_idx == 0, actions_if_0, actions_if_1).astype(jnp.int32)

    next_state, sparse, shaped, _ = env_step(
        terrain, state, joint, shaping_factor=shaping_factor
    )
    reward = sparse + shaped
    done = (next_state.timestep >= jnp.array(horizon, dtype=jnp.int32)).astype(jnp.float32)

    # Accumulate episode rewards; reset on done.
    new_ep_sparse = jnp.where(done, 0.0, ep_sparse + sparse)
    new_ep_shaped = jnp.where(done, 0.0, ep_shaped + shaped)

    # Flip agent_idx on episode end.
    new_agent_idx = jnp.where(done.astype(jnp.bool_), 1 - agent_idx, agent_idx)

    return next_state, reward, done, new_agent_idx, new_ep_sparse, new_ep_shaped, sparse


# ---------------------------------------------------------------------------
# Batched step — the main public API
# ---------------------------------------------------------------------------

@jax.jit
def batched_step(
    terrain: Terrain,
    bstate: BatchedEnvState,
    training_actions: jnp.ndarray,   # [N] int32
    other_actions: jnp.ndarray,       # [N] int32
    shaping_factor: jnp.ndarray,      # scalar float32
    horizon: int,
) -> Tuple[BatchedEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Step all N environments in parallel via vmap.

    Returns
    -------
    new_bstate : BatchedEnvState
    obs0       : [N, H, W, C]  training-agent observations
    obs1       : [N, H, W, C]  other-agent observations
    rewards    : [N]  float32  total (sparse + shaped) reward
    dones      : [N]  float32  1.0 on episode end
    sparse_r   : [N]  float32  sparse component only (for logging)
    """
    next_states, rewards, dones, new_agent_idx, new_ep_sparse, new_ep_shaped, sparse_r = jax.vmap(
        lambda s, idx, ta, oa, ep_sp, ep_sh: _single_step(
            terrain, s, idx, ta, oa, ep_sp, ep_sh, shaping_factor, horizon
        )
    )(
        bstate.states,
        bstate.agent_idx,
        training_actions,
        other_actions,
        bstate.ep_sparse_accum,
        bstate.ep_shaped_accum,
    )

    new_bstate = BatchedEnvState(
        states=next_states,
        agent_idx=new_agent_idx,
        ep_sparse_accum=new_ep_sparse,
        ep_shaped_accum=new_ep_shaped,
    )

    # Encode observations for next step — also vmapped, stays on device.
    obs0, obs1 = jax.vmap(
        lambda s, idx: _encode_single(terrain, s, idx)
    )(next_states, new_agent_idx)

    return new_bstate, obs0, obs1, rewards, dones, sparse_r


# ---------------------------------------------------------------------------
# Convenience: encode initial observations from a fresh BatchedEnvState
# ---------------------------------------------------------------------------

def encode_obs(terrain: Terrain, bstate: BatchedEnvState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (obs0, obs1) for current bstate without stepping."""
    return jax.vmap(
        lambda s, idx: _encode_single(terrain, s, idx)
    )(bstate.states, bstate.agent_idx)
