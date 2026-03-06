"""Partner-agent abstractions for PPO rollouts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.agents.bc.agent import BCAgent
from human_aware_rl_jax_lift.encoding.bc_features import featurize_state_64
from human_aware_rl_jax_lift.env.state import Terrain


def _sample_actions_from_logits(
    logits: jnp.ndarray, rng: jax.Array, stochastic: bool
) -> jnp.ndarray:
    if stochastic:
        return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


class Partner(Protocol):
    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        """Return batch of discrete actions for other agent."""


@dataclass
class SelfPlayPartner:
    stochastic: bool = True
    # Lazily initialised JIT-compiled forward pass.
    # Cached after first call; re-used for all subsequent steps without
    # recompilation (params are JAX arrays = traced values, not static).
    _jit_apply: object = field(default=None, init=False, repr=False, compare=False)

    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        train_state = kwargs["train_state"]
        # Initialise the JIT wrapper once (apply_fn never changes between updates)
        if self._jit_apply is None:
            self._jit_apply = jax.jit(train_state.apply_fn)
        logits, _ = self._jit_apply(
            train_state.params,
            jnp.asarray(obs1_batch, dtype=jnp.float32),
        )
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
        # states may be a batched OvercookedState pytree (new vec_env) or list
        # In either case we index per-env below.
        if hasattr(states, "timestep"):
            # Batched pytree: extract per-env arrays via numpy
            timesteps = np.asarray(states.timestep,    dtype=np.int32)
            player_pos = np.asarray(states.player_pos, dtype=np.int32)
            player_or  = np.asarray(states.player_or,  dtype=np.int32)
            num_envs = timesteps.shape[0]
        else:
            num_envs = len(states)
            # List of states: runner passes materialized (numpy) states
            timesteps  = np.array([int(st.timestep) for st in states], dtype=np.int32)
            player_pos = None
            player_or  = None

        self._ensure_agents(num_envs)
        assert self._agents is not None

        rngs = jax.random.split(rng, num_envs)
        actions = np.zeros((num_envs,), dtype=np.int32)
        for i in range(num_envs):
            if timesteps[i] == 0 and self._agents[i].pos_history:
                self._agents[i] = BCAgent(
                    params=self.params, stochastic=self.stochastic, stuck_time=self.stuck_time
                )
            if hasattr(states, "timestep"):
                # Batched pytree: slice env i
                st_i = jax.tree_util.tree_map(lambda x: x[i], states)
            else:
                st_i = states[i]
            f0, f1 = featurize_state_64(self.terrain, st_i)
            feat = f1 if int(agent_idx[i]) == 0 else f0
            act = self._agents[i].sample_action(jnp.asarray(feat, dtype=jnp.float32), rngs[i])
            other_idx = 1 - int(agent_idx[i])
            if player_pos is not None:
                pos_xy = tuple(player_pos[i, other_idx].tolist())
                or_val  = int(player_or[i, other_idx])
            else:
                pos_xy = tuple(np.asarray(st_i.player_pos[other_idx]).tolist())
                or_val  = int(np.asarray(st_i.player_or[other_idx]))
            self._agents[i].update_history(pos_xy, or_val, int(act))
            actions[i] = int(act)
        return actions


@dataclass
class MixedPartner:
    bc_partner: BCPartner
    trajectory_sp: bool = True
    stochastic: bool = True
    _jit_apply: object = field(default=None, init=False, repr=False, compare=False)

    def act(self, obs1_batch: np.ndarray, rng: jax.Array, **kwargs) -> np.ndarray:
        train_state = kwargs["train_state"]
        self_play_randomization = float(kwargs.get("self_play_randomization", 0.0))

        if self._jit_apply is None:
            self._jit_apply = jax.jit(train_state.apply_fn)

        rng_sp, rng_bc, rng_mix = jax.random.split(rng, 3)
        sp_logits, _ = self._jit_apply(
            train_state.params,
            jnp.asarray(obs1_batch, dtype=jnp.float32),
        )
        sp_actions = _sample_actions_from_logits(sp_logits, rng_sp, self.stochastic)
        bc_actions = jnp.asarray(
            self.bc_partner.act(obs1_batch, rng_bc, **kwargs), dtype=jnp.int32
        )

        if self.trajectory_sp:
            use_sp = (
                jax.random.uniform(rng_mix, ()) < jnp.array(self_play_randomization, dtype=jnp.float32)
            )
            actions = jnp.where(use_sp, sp_actions, bc_actions)
        else:
            choose = jax.random.uniform(rng_mix, (obs1_batch.shape[0],)) < jnp.array(
                self_play_randomization, dtype=jnp.float32
            )
            actions = jnp.where(choose, sp_actions, bc_actions)
        return np.asarray(actions, dtype=np.int32)
