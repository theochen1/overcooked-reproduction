"""Backwards-compatible alias for the default JAX PPO loop.

This repo is JAX-first, so the canonical entrypoint is:

    from human_aware_rl_jax_lift.training.ppo_run import ppo_run

This module remains for older scripts that import ppo_run_jax directly.
"""

from .ppo_run import ppo_run as ppo_run_jax

__all__ = ["ppo_run_jax"]
