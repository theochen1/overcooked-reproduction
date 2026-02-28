"""Training library for faithful legacy-style workflows."""

from .ppo_run import ppo_run

# pbt_run imports RolloutRunner from training.runner (legacy class). Omit from package init so that
# train_ppo_bc and other scripts that only need ppo_run/checkpoints can run.
# To use PBT: from human_aware_rl_jax_lift.training.pbt_run import pbt_run
# (PBT uses runner.make_rollout_fn; RolloutRunner is legacy and not used here).

__all__ = ["ppo_run"]
