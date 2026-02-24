"""BC policy wrapper with legacy unstuck rule."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import jax
import jax.numpy as jnp

from .model import BCPolicy


@dataclass
class BCAgent:
    params: dict
    stochastic: bool = True
    stuck_time: int = 3
    pos_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=4))
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

    def update_history(self, position: Tuple[int, int], action: int) -> None:
        self.pos_history.append(position)
        self.act_history.append(action)

    def _unstuck_adjust(self, probs: jnp.ndarray) -> jnp.ndarray:
        if self.stuck_time <= 0 or len(self.pos_history) < self.stuck_time + 1:
            return probs
        same_pos = all(p == self.pos_history[0] for p in self.pos_history)
        if not same_pos:
            return probs
        blocked_actions = list(self.act_history)
        mask = jnp.ones_like(probs)
        for a in blocked_actions:
            mask = mask.at[a].set(0.0)
        out = probs * mask
        norm = out.sum()
        return jnp.where(norm > 0, out / norm, probs)
