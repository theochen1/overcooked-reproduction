"""Vectorized environment stack for legacy-faithful PPO rollouts."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.encoding.ppo_masks import lossless_state_encoding_20
from human_aware_rl_jax_lift.env.overcooked_mdp import step as env_step
from human_aware_rl_jax_lift.env.state import OvercookedState, Terrain, make_initial_state


@dataclass
class VecStepOut:
    """Container for vectorized step outputs."""

    states: List[OvercookedState]
    obs0: np.ndarray
    obs1: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict[str, object]]
    other_agent_env_idx: np.ndarray


class VectorizedEnv:
    """
    Vectorized multi-env wrapper for JAX-lifted Overcooked dynamics.

    Mirrors the legacy runner assumptions:
    - `obs0` is always for the training agent
    - `obs1` is always for the other agent
    - environment randomizes `agent_idx` on reset
    """

    def __init__(
        self,
        terrain: Terrain,
        num_envs: int,
        horizon: int = 400,
        reward_shaping_params: Dict[str, float] | None = None,
        randomize_agent_idx: bool = True,
    ):
        self.terrain = terrain
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.reward_shaping_params = reward_shaping_params
        self.randomize_agent_idx = bool(randomize_agent_idx)

        self.states: List[OvercookedState] = [make_initial_state(terrain) for _ in range(self.num_envs)]
        self.agent_idx: np.ndarray = np.zeros((self.num_envs,), dtype=np.int32)
        self.last_sparse: np.ndarray = np.zeros((self.num_envs,), dtype=np.float32)
        self.last_shaped: np.ndarray = np.zeros((self.num_envs,), dtype=np.float32)
        self.ep_sparse_accum: np.ndarray = np.zeros((self.num_envs,), dtype=np.float32)
        self.ep_shaped_accum: np.ndarray = np.zeros((self.num_envs,), dtype=np.float32)

    def _sample_agent_idx(self) -> np.ndarray:
        if not self.randomize_agent_idx:
            return np.zeros((self.num_envs,), dtype=np.int32)
        return np.random.randint(0, 2, size=(self.num_envs,), dtype=np.int32)

    def _current_shaping_factor(self) -> float:
        """Infer current scalar shaping factor from PLACEMENT_IN_POT_REW."""
        if not self.reward_shaping_params:
            return 1.0
        base = 3.0
        return float(self.reward_shaping_params.get("PLACEMENT_IN_POT_REW", base)) / base

    def _encode_for_agent_idx(self, state: OvercookedState, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        p0_obs, p1_obs = lossless_state_encoding_20(self.terrain, state)
        if int(idx) == 0:
            return np.asarray(p0_obs, dtype=np.float32), np.asarray(p1_obs, dtype=np.float32)
        return np.asarray(p1_obs, dtype=np.float32), np.asarray(p0_obs, dtype=np.float32)

    def _reset_single(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        self.states[i] = make_initial_state(self.terrain)
        self.ep_sparse_accum[i] = 0.0
        self.ep_shaped_accum[i] = 0.0
        return self._encode_for_agent_idx(self.states[i], int(self.agent_idx[i]))

    def reset_all(self) -> Tuple[List[OvercookedState], np.ndarray, np.ndarray, np.ndarray]:
        """Reset all envs and return states + (obs0, obs1, agent_idx)."""
        self.agent_idx = self._sample_agent_idx()
        obs0_list, obs1_list = [], []
        for i in range(self.num_envs):
            o0, o1 = self._reset_single(i)
            obs0_list.append(o0)
            obs1_list.append(o1)
        return self.states, np.stack(obs0_list), np.stack(obs1_list), self.agent_idx.copy()

    def step_all(self, training_actions: np.ndarray, other_actions: np.ndarray) -> VecStepOut:
        """
        Step all environments by one timestep.

        `training_actions` and `other_actions` are action indices in [0, 5].
        """
        obs0_list, obs1_list = [], []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: List[Dict[str, object]] = []

        for i in range(self.num_envs):
            ta = int(training_actions[i])
            oa = int(other_actions[i])
            if int(self.agent_idx[i]) == 0:
                joint = jnp.array([ta, oa], dtype=jnp.int32)
            else:
                joint = jnp.array([oa, ta], dtype=jnp.int32)

            next_state, sparse, shaped, info = env_step(
                self.terrain,
                self.states[i],
                joint,
                reward_shaping_params=self.reward_shaping_params,
            )
            self.states[i] = next_state

            sparse_f = float(sparse)
            shaped_scaled_f = float(shaped)
            shaped_unscaled_f = float(info.get("shaped_r_unscaled", shaped_scaled_f))
            sf = float(info.get("shaping_factor", self._current_shaping_factor()))
            rew_f = sparse_f + shaped_scaled_f
            self.ep_sparse_accum[i] += sparse_f
            self.ep_shaped_accum[i] += shaped_unscaled_f

            done = int(next_state.timestep) >= self.horizon
            if done:
                ep_info = {
                    "r": float(self.ep_sparse_accum[i] + self.ep_shaped_accum[i] * sf),
                    "ep_sparse_r": float(self.ep_sparse_accum[i]),
                    "ep_shaped_r": float(self.ep_shaped_accum[i]),
                }
                self.agent_idx[i] = int(np.random.randint(0, 2)) if self.randomize_agent_idx else 0
                o0, o1 = self._reset_single(i)
                info_out = {
                    "episode": ep_info,
                    "shaped_r": shaped_unscaled_f,
                    "shaped_r_scaled": shaped_scaled_f,
                    "sparse_r": sparse_f,
                    "shaped_r_by_agent": np.asarray(info["shaped_r_by_agent"]),
                }
            else:
                o0, o1 = self._encode_for_agent_idx(self.states[i], int(self.agent_idx[i]))
                info_out = {
                    "shaped_r": shaped_unscaled_f,
                    "shaped_r_scaled": shaped_scaled_f,
                    "sparse_r": sparse_f,
                    "shaped_r_by_agent": np.asarray(info["shaped_r_by_agent"]),
                }

            obs0_list.append(o0)
            obs1_list.append(o1)
            rewards[i] = rew_f
            dones[i] = done
            infos.append(info_out)

        return VecStepOut(
            states=self.states,
            obs0=np.stack(obs0_list),
            obs1=np.stack(obs1_list),
            rewards=rewards,
            dones=dones,
            infos=infos,
            other_agent_env_idx=1 - self.agent_idx.copy(),
        )
