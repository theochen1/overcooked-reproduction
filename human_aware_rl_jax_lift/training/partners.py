"""Partner-agent abstractions for PPO rollouts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.agents.bc.model import BCPolicy
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

    @classmethod
    def from_path(cls, bc_model_path: str | Path, terrain: Terrain, stochastic: bool = True) -> "BCPartner":
        import pickle

        path = Path(bc_model_path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        # Supports saving plain params or wrapped payload.
        params = payload.get("params", payload) if isinstance(payload, dict) else payload
        return cls(params=params, terrain=terrain, stochastic=stochastic)

    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        states = kwargs["states"]
        agent_idx = kwargs["agent_idx"]
        bc_feats = []
        for i, st in enumerate(states):
            f0, f1 = featurize_state_64(self.terrain, st)
            feat = f1 if int(agent_idx[i]) == 0 else f0
            bc_feats.append(np.asarray(feat, dtype=np.float32))
        bc_feats_batch = jnp.asarray(np.stack(bc_feats), dtype=jnp.float32)
        logits = BCPolicy().apply(self.params, bc_feats_batch)
        acts = _sample_actions_from_logits(logits, rng, self.stochastic)
        return np.asarray(acts, dtype=np.int32)


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
