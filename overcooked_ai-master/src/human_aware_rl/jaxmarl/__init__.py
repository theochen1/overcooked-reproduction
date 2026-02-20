"""
JAX-based Multi-Agent RL for Overcooked AI.

This module provides a JAX/Flax implementation of PPO for training
agents in the Overcooked cooperative cooking environment. It faithfully
reproduces the original 2019 paper results using modern JAX infrastructure.

Key Features:
- PPO training with self-play support
- Vectorized environment wrapper for efficient parallel rollouts
- Compatible with the original paper's 20-channel observation encoding
- Achieves 140+ training reward on random0_legacy (forced_coordination)
- Achieves 120+ training reward on random3_legacy

Usage:
    from human_aware_rl.jaxmarl import PPOConfig, PPOTrainer
    
    config = PPOConfig(
        layout_name="random0_legacy",
        total_timesteps=5_000_000,
        ent_coef=0.01,
    )
    trainer = PPOTrainer(config)
    results = trainer.train()

For paper reproduction with original hyperparameters:
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout random0_legacy
"""

from human_aware_rl.jaxmarl.overcooked_env import (
    OvercookedJaxEnv,
    OvercookedJaxEnvConfig,
    VectorizedOvercookedEnv,
)
from human_aware_rl.jaxmarl.ppo import (
    PPOConfig,
    PPOTrainer,
    train_ppo,
)

__all__ = [
    "OvercookedJaxEnv",
    "OvercookedJaxEnvConfig",
    "VectorizedOvercookedEnv",
    "PPOConfig",
    "PPOTrainer",
    "train_ppo",
]
