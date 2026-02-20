"""
Bridge module for converting trained policies to Overcooked Agents.

This module provides utilities for loading trained policies (from JAX/PyTorch)
and wrapping them as Overcooked-compatible Agent objects for evaluation.
"""

from human_aware_rl.bridge.jax_agent import (
    JaxPolicyAgent,
    load_jax_agent,
)
from human_aware_rl.bridge.evaluate import (
    evaluate_agent_pair,
    evaluate_self_play,
    load_and_evaluate,
)

__all__ = [
    "JaxPolicyAgent",
    "load_jax_agent",
    "evaluate_agent_pair",
    "evaluate_self_play",
    "load_and_evaluate",
]

