#!/usr/bin/env python
"""
Paper Reproduction Training Script (PPO Self-Play)

Trains PPO self-play agents using per-layout hyperparameters from the paper's Table 2.
Pulls configs from paper_configs.py to avoid hardcoded values drifting out of sync.

Key settings matching the paper:
1. Uses legacy 20-channel observation encoding (not 26-channel)
2. Uses per-minibatch advantage normalization (not per-batch)
3. Entropy coefficient = 0.01 (constant, no annealing)
4. Per-layout learning rate (1e-3 / 6e-4 / 8e-4 depending on layout)
5. VF_COEF = 0.5
6. num_envs = 30 (30 envs * 400 steps = 12,000 batch)
7. num_minibatches = 6, minibatch_size = 2000
8. Per-layout reward shaping horizon (2.5e6 or 3.5e6)
9. No LR annealing for SP (constant LR)
10. Shared shaped reward for both agents (not per-agent)
11. Agent index randomization on reset (learns from both starting positions)
12. Glorot uniform weight init for conv/dense layers (not orthogonal)
13. Leaky ReLU with negative_slope=0.2 (matching TensorFlow default)
14. Stochastic action sampling in evaluation (matches training behavior)

Usage:
    cd overcooked_ai-master/src
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout cramped_room
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout forced_coordination
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --all_layouts --seeds 0,10,20,30,40
"""

import argparse
import os
import sys

# Ensure we can import from the correct location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from human_aware_rl.jaxmarl.ppo import PPOConfig, PPOTrainer
from human_aware_rl.ppo.configs.paper_configs import (
    PAPER_LAYOUTS,
    LAYOUT_TO_ENV,
    get_ppo_sp_config,
)


def get_paper_reproduction_config(
    layout: str,
    seed: int = 0,
    total_timesteps: int = None,
    results_dir: str = "paper_reproduction_results"
) -> PPOConfig:
    """
    Get PPO SP config for a layout using paper_configs.py (Paper Table 2).

    Args:
        layout: Paper layout name (e.g., 'cramped_room', 'forced_coordination')
        seed: Random seed
        total_timesteps: Override total timesteps (None = 10M default)
        results_dir: Directory to save results

    Returns:
        PPOConfig with paper-matching hyperparameters
    """
    # Get config dict from paper_configs (has per-layout LR, rew horizon, etc.)
    overrides = {"results_dir": results_dir}
    if total_timesteps is not None:
        overrides["total_timesteps"] = total_timesteps

    config_dict = get_ppo_sp_config(layout=layout, seed=seed, **overrides)

    # Map to legacy layout name for the environment
    env_layout = LAYOUT_TO_ENV.get(layout, layout)

    return PPOConfig(
        # Environment
        layout_name=env_layout,
        horizon=config_dict.get("horizon", 400),
        num_envs=config_dict.get("num_workers", 30),
        old_dynamics=config_dict.get("old_dynamics", True),

        # Training
        total_timesteps=config_dict["total_timesteps"],
        learning_rate=config_dict["learning_rate"],
        num_steps=config_dict.get("rollout_fragment_length", 400),
        num_minibatches=config_dict.get("num_minibatches", 6),
        num_epochs=config_dict.get("num_sgd_iter", 8),

        # PPO hyperparameters
        gamma=config_dict["gamma"],
        gae_lambda=config_dict["gae_lambda"],
        clip_eps=config_dict["clip_eps"],
        ent_coef=config_dict.get("entropy_coeff_start", 0.01),
        vf_coef=config_dict["vf_coef"],
        max_grad_norm=config_dict["max_grad_norm"],
        kl_coeff=config_dict.get("kl_coeff", 0.2),

        # No LR/entropy annealing for SP
        use_lr_annealing=False,
        use_entropy_annealing=False,
        entropy_coeff_start=config_dict.get("entropy_coeff_start", 0.01),
        entropy_coeff_end=config_dict.get("entropy_coeff_end", 0.01),

        # Value function clipping (original baselines uses this)
        clip_vf=True,

        # Reward shaping -- per-layout from paper Table 2
        reward_shaping_factor=config_dict.get("reward_shaping_factor", 1.0),
        reward_shaping_horizon=config_dict["reward_shaping_horizon"],
        use_phi=config_dict.get("use_phi", False),

        # Observation encoding
        use_legacy_encoding=config_dict.get("use_legacy_encoding", True),

        # Network architecture
        num_hidden_layers=config_dict.get("num_hidden_layers", 3),
        hidden_dim=config_dict.get("hidden_dim", 64),
        num_filters=config_dict.get("num_filters", 25),
        num_conv_layers=config_dict.get("num_conv_layers", 3),
        use_lstm=config_dict.get("use_lstm", False),

        # Logging
        log_interval=1,
        save_interval=50,
        eval_interval=25,
        eval_num_games=5,
        verbose=True,

        # No early stopping for paper reproduction
        use_early_stopping=False,

        # No BC partner for self-play
        bc_schedule=[(0, 0.0), (float('inf'), 0.0)],

        # Output
        results_dir=results_dir,
        experiment_name=config_dict["experiment_name"],
        seed=seed,
    )


def train_layout(layout: str, seed: int, total_timesteps: int = None,
                 results_dir: str = "paper_reproduction_results"):
    """Train a single layout/seed combination."""
    config = get_paper_reproduction_config(
        layout=layout,
        seed=seed,
        total_timesteps=total_timesteps,
        results_dir=results_dir,
    )

    env_layout = LAYOUT_TO_ENV.get(layout, layout)

    # Print config summary
    print("=" * 60)
    print("PAPER REPRODUCTION TRAINING (PPO SP, Table 2)")
    print("=" * 60)
    print(f"Layout: {layout} -> {env_layout}")
    print(f"Seed: {config.seed}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Num envs: {config.num_envs}")
    print(f"Steps per env: {config.num_steps}")
    print(f"Batch size: {config.num_envs * config.num_steps:,}")
    print(f"Num minibatches: {config.num_minibatches} (minibatch_size={config.num_envs * config.num_steps // config.num_minibatches})")
    print()
    print("Key hyperparameters (Paper Table 2):")
    print(f"  Learning rate: {config.learning_rate} (constant)")
    print(f"  Entropy coef: {config.ent_coef}")
    print(f"  VF coef: {config.vf_coef}")
    print(f"  Clip epsilon: {config.clip_eps}")
    print(f"  Max grad norm: {config.max_grad_norm}")
    print(f"  GAE lambda: {config.gae_lambda}")
    print(f"  Gamma: {config.gamma}")
    print()
    print("Reward shaping:")
    print(f"  Reward shaping factor: {config.reward_shaping_factor}")
    print(f"  Reward shaping horizon: {config.reward_shaping_horizon:,.0f} (anneals to 0 by this step)")
    print("=" * 60)
    print()

    # Create trainer and train
    trainer = PPOTrainer(config)

    print(f"Observation shape: {trainer.obs_shape}")
    print()

    results = trainer.train()

    # Print final results
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final mean reward: {results.get('final_mean_reward', 'N/A'):.2f}")
    print(f"Best mean reward: {results.get('best_mean_reward', 'N/A'):.2f}")
    print(f"Final eval reward: {results.get('final_eval_reward', 'N/A'):.2f}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train PPO SP agents for paper reproduction (Table 2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--layout', type=str, default=None,
                        choices=PAPER_LAYOUTS + ['random0_legacy', 'random3_legacy'],
                        help='Layout to train on')
    parser.add_argument('--all_layouts', action='store_true',
                        help='Train all 5 paper layouts')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (for single layout)')
    parser.add_argument('--seeds', type=str, default='0,10,20,30,40',
                        help='Comma-separated seeds (for --all_layouts)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total timesteps (default: 10M)')
    parser.add_argument('--results-dir', type=str, default='paper_reproduction_results',
                        help='Directory to save results')
    args = parser.parse_args()

    # Map legacy names to paper names for convenience
    layout_aliases = {
        'random0_legacy': 'forced_coordination',
        'random3_legacy': 'counter_circuit',
    }

    if args.all_layouts:
        seeds = [int(s) for s in args.seeds.split(',')]
        for layout in PAPER_LAYOUTS:
            for seed in seeds:
                print(f"\n{'#'*60}")
                print(f"# {layout} seed={seed}")
                print(f"{'#'*60}\n")
                try:
                    train_layout(layout, seed, args.timesteps, args.results_dir)
                except Exception as e:
                    print(f"ERROR training {layout} seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
    elif args.layout:
        layout = layout_aliases.get(args.layout, args.layout)
        train_layout(layout, args.seed, args.timesteps, args.results_dir)
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()
