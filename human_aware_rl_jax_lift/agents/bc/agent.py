"""BC policy wrapper with legacy unstuck rule."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

import jax
import jax.numpy as jnp

from .model import BCPolicy

# Action indices matching paper_config.yaml order:
# 0=north, 1=south, 2=east, 3=west, 4=stay, 5=interact
_MOVEMENT_ACTIONS = frozenset({0, 1, 2, 3})


@dataclass
class BCAgent:
    params: dict
    stochastic: bool = True
    stuck_time: int = 3
    pos_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=4))
    or_history: Deque[int] = field(default_factory=lambda: deque(maxlen=4))
    act_history: Deque[int] = field(default_factory=lambda: deque(maxlen=4))

    def action_probs(self, features: jnp.ndarray) -> jnp.ndarray:
        logits = BCPolicy().apply(self.params, features[None, ...])[0]
        probs = jax.nn.softmax(logits)
        return self._unstuck_adjust(probs)

    def sample_action(self, features: jnp.ndarray, rng) -> int:
        probs = self.action_probs(features)
        if self.stochastic:
            return int(jax.random.choice(rng, jnp.arange(probs.shape[0]), p=probs))
        return int(jnp.argmax(probs))

    def update_history(self, position: Tuple[int, int], orientation: int, action: int) -> None:
        """Record position, orientation, and action for the stuck heuristic.

        Orientation must be passed alongside position to match TF
        ImitationAgentFromPolicy.is_stuck which checks pos_and_or. Tracking
        position alone incorrectly flags agents that rotate in place (e.g.
        turning to face a counter before interacting).
        """
        self.pos_history.append(position)
        self.or_history.append(orientation)
        self.act_history.append(action)

    def _unstuck_adjust(self, probs: jnp.ndarray) -> jnp.ndarray:
        if self.stuck_time <= 0 or len(self.pos_history) < self.stuck_time + 1:
            return probs
        same_pos = all(p == self.pos_history[0] for p in self.pos_history)
        same_or  = all(o == self.or_history[0]  for o in self.or_history)
        if not (same_pos and same_or):
            return probs

        blocked_actions = set(self.act_history)

        # Mirror TF's assertion: skip adjustment entirely if no movement
        # direction would remain unblocked after masking. This matches:
        #   assert any([a not in last_actions for a in Direction.ALL_DIRECTIONS])
        # in ImitationAgentFromPolicy.unblock_if_stuck.
        if _MOVEMENT_ACTIONS.issubset(blocked_actions):
            return probs

        mask = jnp.ones_like(probs)
        for a in blocked_actions:
            mask = mask.at[a].set(0.0)
        out = probs * mask
        norm = out.sum()
        return jnp.where(norm > 0, out / norm, probs)
