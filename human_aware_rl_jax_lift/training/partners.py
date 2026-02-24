"""Partner-agent abstractions for PPO rollouts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.agents.bc.agent import BCAgent
from human_aware_rl_jax_lift.encoding.bc_features import featurize_state_64
from human_aware_rl_jax_lift.env.state import Terrain


def _sample_actions_from_logits(logits: jnp.ndarray, rng: jax.Array, stochastic: bool) -> jnp.ndarray:
    if stochastic:
        return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


class Partner(Protocol):
    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        """Return batch of discrete actions for other agent."""


@dataclass
class SelfPlayPartner:
    stochastic: bool = True

    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        train_state = kwargs["train_state"]
        logits, _ = train_state.apply_fn(train_state.params, jnp.asarray(obs1_batch, dtype=jnp.float32))
        acts = _sample_actions_from_logits(logits, rng, self.stochastic)
        return np.asarray(acts, dtype=np.int32)


@dataclass
class BCPartner:
    params: dict
    terrain: Terrain
    stochastic: bool = True
    stuck_time: int = 3
    _agents: list[BCAgent] | None = None

    @classmethod
    def from_path(
        cls,
        bc_model_path: str | Path,
        terrain: Terrain,
        stochastic: bool = True,
        stuck_time: int = 3,
    ) -> "BCPartner":
        import pickle

        path = Path(bc_model_path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        # Supports saving plain params or wrapped payload.
        params = payload.get("params", payload) if isinstance(payload, dict) else payload
        return cls(params=params, terrain=terrain, stochastic=stochastic, stuck_time=stuck_time)

    def _ensure_agents(self, num_envs: int) -> None:
        if self._agents is not None and len(self._agents) == num_envs:
            return
        self._agents = [
            BCAgent(params=self.params, stochastic=self.stochastic, stuck_time=self.stuck_time)
            for _ in range(num_envs)
        ]

    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        states = kwargs["states"]
        agent_idx = kwargs["agent_idx"]
        num_envs = len(states)
        self._ensure_agents(num_envs)
        assert self._agents is not None

        rngs = jax.random.split(rng, num_envs)
        actions = np.zeros((num_envs,), dtype=np.int32)
        for i, st in enumerate(states):
            # Auto-reset unstuck history when env has just reset.
            if int(st.timestep) == 0 and self._agents[i].pos_history:
                self._agents[i] = BCAgent(params=self.params, stochastic=self.stochastic, stuck_time=self.stuck_time)

            f0, f1 = featurize_state_64(self.terrain, st)
            feat = f1 if int(agent_idx[i]) == 0 else f0
            act = self._agents[i].sample_action(jnp.asarray(feat, dtype=jnp.float32), rngs[i])
            # Track position for the controlled "other" agent in this env.
            other_idx = 1 - int(agent_idx[i])
            pos_xy = tuple(np.asarray(st.player_pos[other_idx]).tolist())
            self._agents[i].update_history(pos_xy, int(act))
            actions[i] = int(act)
        return actions


@dataclass
class MixedPartner:
    bc_partner: BCPartner
    trajectory_sp: bool = True
    stochastic: bool = True

    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        train_state = kwargs["train_state"]
        self_play_randomization = float(kwargs.get("self_play_randomization", 0.0))

        rng_sp, rng_bc, rng_mix = jax.random.split(rng, 3)
        sp_logits, _ = train_state.apply_fn(train_state.params, jnp.asarray(obs1_batch, dtype=jnp.float32))
        sp_actions = _sample_actions_from_logits(sp_logits, rng_sp, self.stochastic)
        bc_actions = jnp.asarray(self.bc_partner.act(obs1_batch, rng_bc, **kwargs), dtype=jnp.int32)

        if self.trajectory_sp:
            use_sp = jax.random.uniform(rng_mix, ()) < jnp.array(self_play_randomization, dtype=jnp.float32)
            actions = jnp.where(use_sp, sp_actions, bc_actions)
        else:
            choose = jax.random.uniform(rng_mix, (obs1_batch.shape[0],)) < jnp.array(
                self_play_randomization, dtype=jnp.float32
            )
            actions = jnp.where(choose, sp_actions, bc_actions)
        return np.asarray(actions, dtype=np.int32)
