"""Training library for faithful legacy-style workflows."""

from .ppo_run import ppo_run

# pbt_run imports RolloutRunner from training.runner, which does not exist in this
# JAX-lift repo (only runner_jax exists). Omit from package init so that
# train_ppo_bc and other scripts that only need ppo_run/checkpoints can run.
# To use PBT: from human_aware_rl_jax_lift.training.pbt_run import pbt_run
# (will fail until a legacy-style runner or PBT port to runner_jax exists).

__all__ = ["ppo_run"]
