"""Rollout collection mirroring legacy two-agent runner behavior."""

from dataclasses import dataclass
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from .partners import Partner
from .vec_env import VectorizedEnv


def _policy_step_impl(apply_fn, params, obs: jnp.ndarray, rng: jax.Array):
    logits, values = apply_fn(params, obs)
    actions = jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)
    logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), actions]
    return actions, values, logp


@dataclass
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    next_value: np.ndarray
    infos: Dict[str, object]


class RolloutRunner:
    """Collects fixed-horizon rollouts for PPO updates."""

    def __init__(
        self,
        vec_env: VectorizedEnv,
        train_state: TrainState,
        other_agent: Partner,
        horizon: int,
        trajectory_self_play: bool = True,
    ):
        self.vec_env = vec_env
        self.train_state = train_state
        self.other_agent = other_agent
        self.horizon = int(horizon)
        self.trajectory_self_play = bool(trajectory_self_play)
        self._policy_step_jit = jax.jit(
            lambda params, obs, rng: _policy_step_impl(self.train_state.apply_fn, params, obs, rng)
        )
        _, self.obs0, self.obs1, self.agent_idx = self.vec_env.reset_all()

    def _policy_step(
        self, obs: np.ndarray, rng: jax.Array
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions, values, logp = self._policy_step_jit(
            self.train_state.params,
            jnp.asarray(obs, dtype=jnp.float32),
            rng,
        )
        return (
            np.asarray(actions, dtype=np.int32),
            np.asarray(values, dtype=np.float32),
            np.asarray(logp, dtype=np.float32),
        )

    def collect_rollout(self, rng: jax.Array, self_play_randomization: float = 0.0) -> RolloutBatch:
        """Collect one PPO rollout of length `horizon` across all envs."""
        obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
        val_buf, logp_buf = [], []
        ep_returns, ep_sparse_returns = [], []

        for _ in range(self.horizon):
            rng, rng_policy, rng_partner = jax.random.split(rng, 3)
            actions, values, logp = self._policy_step(self.obs0, rng_policy)
            other_actions = self.other_agent.act(
                self.obs1,
                rng_partner,
                train_state=self.train_state,
                self_play_randomization=self_play_randomization,
                trajectory_sp=self.trajectory_self_play,
                states=self.vec_env.states,
                agent_idx=self.vec_env.agent_idx,
            )
            step_out = self.vec_env.step_all(actions, other_actions)

            obs_buf.append(self.obs0.copy())
            act_buf.append(actions)
            rew_buf.append(step_out.rewards.copy())
            done_buf.append(step_out.dones.astype(np.float32))
            val_buf.append(values)
            logp_buf.append(logp)

            for info in step_out.infos:
                if "episode" in info:
                    ep_returns.append(float(info["episode"]["r"]))
                    ep_sparse_returns.append(float(info["episode"]["ep_sparse_r"]))

            self.obs0 = step_out.obs0
            self.obs1 = step_out.obs1
            self.agent_idx = step_out.other_agent_env_idx

        _, next_values = self.train_state.apply_fn(self.train_state.params, jnp.asarray(self.obs0, dtype=jnp.float32))
        infos = {
            "eprewmean": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "ep_sparse_rew_mean": float(np.mean(ep_sparse_returns)) if ep_sparse_returns else 0.0,
            "episodes_this_rollout": len(ep_returns),
        }
        return RolloutBatch(
            obs=np.asarray(obs_buf, dtype=np.float32),
            actions=np.asarray(act_buf, dtype=np.int32),
            rewards=np.asarray(rew_buf, dtype=np.float32),
            dones=np.asarray(done_buf, dtype=np.float32),
            values=np.asarray(val_buf, dtype=np.float32),
            log_probs=np.asarray(logp_buf, dtype=np.float32),
            next_value=np.asarray(next_values, dtype=np.float32),
            infos=infos,
        )
