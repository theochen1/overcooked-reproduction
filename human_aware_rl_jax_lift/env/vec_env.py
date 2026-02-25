"""Vectorized environment stack — fully batched with jax.vmap.

Replaces the old per-env Python loop + 90 GPU->CPU syncs per timestep with:
  - jax.vmap(env_step) over all envs in one JIT'd call
  - A single bulk GPU->CPU transfer per timestep
  - No float()/int() inside the hot path
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.encoding.ppo_masks import lossless_state_encoding_20
from human_aware_rl_jax_lift.env.overcooked_mdp import step as env_step
from human_aware_rl_jax_lift.env.state import OvercookedState, Terrain, make_initial_state


@dataclass
class VecStepOut:
    states: OvercookedState  # batched pytree
    obs0: np.ndarray
    obs1: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict]
    other_agent_env_idx: np.ndarray


class VectorizedEnv:
    """
    Fully batched Overcooked environment.

    Internal state is a single batched OvercookedState pytree (leading dim = num_envs).
    Each call to step_all() issues ONE vmapped JAX dispatch and ONE bulk transfer.
    """

    def __init__(
        self,
        terrain: Terrain,
        num_envs: int,
        horizon: int = 400,
        reward_shaping_params: Optional[Dict] = None,
        randomize_agent_idx: bool = True,
    ):
        self.terrain = terrain
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.randomize_agent_idx = bool(randomize_agent_idx)

        # Store shaping factor as a JAX scalar so it's a traced value —
        # changing it does NOT trigger XLA recompilation.
        self._shaping_factor = jnp.array(1.0)
        if reward_shaping_params is not None:
            self._shaping_factor = self._factor_from_dict(reward_shaping_params)

        # Build JIT + vmap functions once at construction.
        # terrain is captured in the closure (static per layout).
        # in_axes=(0, 0, None): state and joint vary per env; shaping_factor is shared.
        self._step_jit = jax.jit(
            jax.vmap(
                lambda s, j, sf: env_step(terrain, s, j, shaping_factor=sf),
                in_axes=(0, 0, None),
            )
        )
        self._encode_jit = jax.jit(
            jax.vmap(lambda s: lossless_state_encoding_20(terrain, s))
        )

        self._batched_state: OvercookedState = self._make_batched_initial()
        self.agent_idx: np.ndarray = np.zeros((self.num_envs,), dtype=np.int32)
        self.ep_sparse_accum: np.ndarray = np.zeros((self.num_envs,), dtype=np.float32)
        self.ep_shaped_accum: np.ndarray = np.zeros((self.num_envs,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _factor_from_dict(params: Dict) -> jnp.ndarray:
        base = 3.0  # DEFAULT PLACEMENT_IN_POT_REW
        val = float(params.get("PLACEMENT_IN_POT_REW", base))
        return jnp.array(val / base if base > 0 else 1.0)

    def _make_batched_initial(self) -> OvercookedState:
        initial = make_initial_state(self.terrain)
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (self.num_envs,) + x.shape),
            initial,
        )

    def _encode_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Encode all envs in one vmapped call, then swap by agent_idx on CPU."""
        p0_batch, p1_batch = self._encode_jit(self._batched_state)
        p0_np = np.asarray(p0_batch, dtype=np.float32)
        p1_np = np.asarray(p1_batch, dtype=np.float32)
        ndim_extra = p0_np.ndim - 1
        idx = self.agent_idx.reshape((-1,) + (1,) * ndim_extra)
        obs0 = np.where(idx == 0, p0_np, p1_np)
        obs1 = np.where(idx == 0, p1_np, p0_np)
        return obs0, obs1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def states(self) -> OvercookedState:
        """Batched state pytree (for partner/runner compatibility)."""
        return self._batched_state

    @property
    def reward_shaping_params(self) -> Dict:
        f = float(self._shaping_factor)
        return {
            "PLACEMENT_IN_POT_REW":    3.0 * f,
            "DISH_PICKUP_REWARD":      3.0 * f,
            "SOUP_PICKUP_REWARD":      5.0 * f,
            "DISH_DISP_DISTANCE_REW":  0.0,
            "POT_DISTANCE_REW":        0.0,
            "SOUP_DISTANCE_REW":       0.0,
        }

    @reward_shaping_params.setter
    def reward_shaping_params(self, params: Optional[Dict]) -> None:
        if params is None:
            self._shaping_factor = jnp.array(1.0)
        else:
            self._shaping_factor = self._factor_from_dict(params)

    def _sample_agent_idx(self) -> np.ndarray:
        if not self.randomize_agent_idx:
            return np.zeros((self.num_envs,), dtype=np.int32)
        return np.random.randint(0, 2, size=(self.num_envs,), dtype=np.int32)

    def reset_all(self) -> Tuple[OvercookedState, np.ndarray, np.ndarray, np.ndarray]:
        """Reset all envs and return (batched_state, obs0, obs1, agent_idx)."""
        self._batched_state = self._make_batched_initial()
        self.agent_idx = self._sample_agent_idx()
        self.ep_sparse_accum[:] = 0.0
        self.ep_shaped_accum[:] = 0.0
        obs0, obs1 = self._encode_obs()
        return self._batched_state, obs0, obs1, self.agent_idx.copy()

    def step_all(self, training_actions: np.ndarray, other_actions: np.ndarray) -> VecStepOut:
        """
        Step all environments in one vmapped JAX call.

        GPU->CPU transfer: ONE bulk transfer for (sparse_rew, shaped_rew, timesteps).
        No Python per-env loop, no float()/int() inside the hot path.
        """
        ta = jnp.asarray(training_actions, dtype=jnp.int32)
        oa = jnp.asarray(other_actions, dtype=jnp.int32)
        idx_jnp = jnp.asarray(self.agent_idx, dtype=jnp.int32)

        # Build joint actions: when agent_idx=0, player0=training, player1=other
        p0 = jnp.where(idx_jnp == 0, ta, oa)
        p1 = jnp.where(idx_jnp == 0, oa, ta)
        joints = jnp.stack([p0, p1], axis=1)  # (num_envs, 2)

        # ----------------------------------------------------------------
        # ONE vmapped JAX dispatch — replaces 30-env Python loop
        # ----------------------------------------------------------------
        next_states, sparse_rew, shaped_rew, step_infos = self._step_jit(
            self._batched_state, joints, self._shaping_factor
        )
        self._batched_state = next_states

        # ----------------------------------------------------------------
        # ONE bulk GPU->CPU transfer (was 90 separate float()/int() calls)
        # ----------------------------------------------------------------
        sparse_np   = np.asarray(sparse_rew,          dtype=np.float32)  # (num_envs,)
        shaped_np   = np.asarray(shaped_rew,          dtype=np.float32)  # (num_envs,)
        timestep_np = np.asarray(next_states.timestep, dtype=np.int32)   # (num_envs,)
        sr_by_agent = np.asarray(
            step_infos["shaped_r_by_agent"], dtype=np.float32
        )  # (num_envs, 2)

        dones   = timestep_np >= self.horizon  # (num_envs,) bool
        rewards = sparse_np + shaped_np

        self.ep_sparse_accum += sparse_np
        self.ep_shaped_accum += shaped_np

        # ----------------------------------------------------------------
        # Episode bookkeeping (CPU only — no GPU involved)
        # ----------------------------------------------------------------
        info_list: List[Dict] = []
        for i in range(self.num_envs):
            info: Dict = {
                "shaped_r": float(shaped_np[i]),
                "sparse_r": float(sparse_np[i]),
                "shaped_r_by_agent": sr_by_agent[i],
            }
            if dones[i]:
                info["episode"] = {
                    "r":          float(self.ep_sparse_accum[i] + self.ep_shaped_accum[i]),
                    "ep_sparse_r": float(self.ep_sparse_accum[i]),
                    "ep_shaped_r": float(self.ep_shaped_accum[i]),
                }
                self.ep_sparse_accum[i] = 0.0
                self.ep_shaped_accum[i] = 0.0
                if self.randomize_agent_idx:
                    self.agent_idx[i] = int(np.random.randint(0, 2))
            info_list.append(info)

        # ----------------------------------------------------------------
        # Reset done envs in batched JAX state
        # ----------------------------------------------------------------
        if np.any(dones):
            reset_state = make_initial_state(self.terrain)
            done_mask = jnp.asarray(dones)  # (num_envs,) bool

            def reset_leaf(
                r_leaf: jnp.ndarray, b_leaf: jnp.ndarray
            ) -> jnp.ndarray:
                # r_leaf: shape (), (k,), (k, m), ...
                # b_leaf: shape (num_envs,), (num_envs, k), ...
                extra_dims = (1,) * r_leaf.ndim
                mask = done_mask.reshape((-1,) + extra_dims)
                tiled = jnp.broadcast_to(r_leaf[None], b_leaf.shape)
                return jnp.where(mask, tiled, b_leaf)

            self._batched_state = jax.tree_util.tree_map(
                reset_leaf, reset_state, self._batched_state
            )

        obs0, obs1 = self._encode_obs()

        return VecStepOut(
            states=self._batched_state,
            obs0=obs0,
            obs1=obs1,
            rewards=rewards,
            dones=dones,
            infos=info_list,
            other_agent_env_idx=1 - self.agent_idx.copy(),
        )
