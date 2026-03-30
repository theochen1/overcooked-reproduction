"""BC policy wrapper with unstuck heuristic."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

import jax
import jax.numpy as jnp

from .model import BCPolicy

# Action indices matching paper_config.yaml order:
# 0=north, 1=south, 2=east, 3=west, 4=stay, 5=interact
_MOVEMENT_ACTIONS = frozenset({0, 1, 2, 3})
_ACTION_STAY = 4


@dataclass
class BCAgent:
    params: dict
    stochastic: bool = True
    stuck_time: int = 3
    no_waits: bool = False
    pos_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=4))
    or_history: Deque[int] = field(default_factory=lambda: deque(maxlen=4))
    act_history: Deque[int] = field(default_factory=lambda: deque(maxlen=4))
    # Counts how many update_obs() calls have been made; used to gate stuck detection.
    # Separate from len(pos_history) because deques saturate at maxlen.
    _obs_count: int = field(default=0, init=False, repr=False)

    def action_probs(self, features: jnp.ndarray) -> jnp.ndarray:
        logits = BCPolicy().apply(self.params, features[None, ...])[0]
        probs = jax.nn.softmax(logits)
        if self.no_waits:
            probs = probs.at[_ACTION_STAY].set(0.0)
            norm = probs.sum()
            probs = jnp.where(norm > 0, probs / norm, probs)
        return self._unstuck_adjust(probs)

    def sample_action(self, features: jnp.ndarray, rng) -> int:
        probs = self.action_probs(features)
        if self.stochastic:
            return int(jax.random.choice(rng, jnp.arange(probs.shape[0]), p=probs))
        return int(jnp.argmax(probs))

    def update_obs(self, position: Tuple[int, int], orientation: int) -> None:
        """Push the CURRENT step's position and orientation into history.

        Must be called BEFORE action_probs()/sample_action() so that the stuck
        check includes the current observation.
        After calling this, call update_action() with the chosen action.
        """
        self.pos_history.append(position)
        self.or_history.append(orientation)
        self._obs_count += 1

    def update_action(self, action: int) -> None:
        """Push the chosen action into history (call after sample_action)."""
        self.act_history.append(action)

    def _unstuck_adjust(self, probs: jnp.ndarray) -> jnp.ndarray:
        # Require stuck_time+2 observations before firing.
        if self.stuck_time <= 0 or self._obs_count < self.stuck_time + 2:
            return probs
        same_pos = all(p == self.pos_history[0] for p in self.pos_history)
        same_or  = all(o == self.or_history[0]  for o in self.or_history)
        if not (same_pos and same_or):
            return probs

        # Block only the last stuck_time actions, not all in history.
        blocked_actions = set(list(self.act_history)[-self.stuck_time:])

        # Skip adjustment if all movement directions would be blocked.
        if _MOVEMENT_ACTIONS.issubset(blocked_actions):
            return probs

        mask = jnp.ones_like(probs)
        for a in blocked_actions:
            mask = mask.at[a].set(0.0)
        out = probs * mask
        norm = out.sum()
        return jnp.where(norm > 0, out / norm, probs)
