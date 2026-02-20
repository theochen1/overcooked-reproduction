"""
PPO Training Client for Overcooked AI.

This script provides a command-line interface for training PPO agents
in the Overcooked environment using the JAX-based training infrastructure.

Usage:
    python ppo_client.py [options]
    
    # Self-play training
    python ppo_client.py --layout cramped_room --total_timesteps 1000000
    
    # Training with BC agent partner
    python ppo_client.py --layout cramped_room --bc_model_dir path/to/bc_model
"""

import argparse
import os
import sys
import json
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Check if JAX is available
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Sacred for experiment tracking (optional)
try:
    from sacred import Experiment, SETTINGS
    from sacred.observers import FileStorageObserver
    SACRED_AVAILABLE = True
    
    # Setup Sacred
    SETTINGS.CONFIG.READ_ONLY_CONFIG = False
    ex = Experiment("PPO_Overcooked")
except ImportError:
    SACRED_AVAILABLE = False
    ex = None

# WandB for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Default configuration
DEFAULT_CONFIG = {
    # Environment
    "layout_name": "cramped_room",
    "horizon": 400,
    "num_envs": 8,
    
    # Training
    "total_timesteps": 1_000_000,
    "learning_rate": 3e-4,
    "num_steps": 128,
    "num_minibatches": 4,
    "num_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    
    # Network
    "num_hidden_layers": 3,
    "hidden_dim": 64,
    "num_filters": 25,
    "num_conv_layers": 3,
    "use_lstm": False,
    "cell_size": 256,
    
    # Reward shaping
    "reward_shaping_factor": 1.0,
    "reward_shaping_horizon": 0,
    "use_phi": True,
    
    # BC schedule
    "bc_schedule": [(0, 0.0), (float('inf'), 0.0)],
    "bc_model_dir": None,
    
    # Logging
    "log_interval": 10,
    "save_interval": 100,
    "eval_interval": 50,
    "eval_num_games": 5,
    "verbose": True,
    "use_wandb": False,
    "wandb_project": "overcooked-ai",
    
    # Output
    "results_dir": "results",
    "experiment_name": None,
    "seed": 0,
}

# Local (testing) configuration overrides
LOCAL_CONFIG = {
    "num_envs": 2,
    "total_timesteps": 10000,
    "log_interval": 5,
    "save_interval": 50,
    "verbose": True,
}


def get_config(local: bool = False, **overrides) -> Dict[str, Any]:
    """
    Get training configuration.
    
    Args:
        local: Whether to use local (testing) configuration
        **overrides: Configuration overrides
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if local:
        config.update(LOCAL_CONFIG)
    
    config.update(overrides)
    
    # Generate experiment name if not provided
    if config["experiment_name"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config["experiment_name"] = f"ppo_{config['layout_name']}_{timestamp}"
    
    return config


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run PPO training with the given configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for PPO training. "
            "Install with: pip install jax jaxlib flax optax"
        )
    
    from human_aware_rl.jaxmarl.ppo import PPOConfig, PPOTrainer
    
    # Set random seed
    np.random.seed(config["seed"])
    
    # Initialize WandB if enabled
    if config["use_wandb"] and WANDB_AVAILABLE:
        wandb.init(
            project=config["wandb_project"],
            name=config["experiment_name"],
            config=config,
        )
    
    # Create PPO config
    ppo_config = PPOConfig(
        layout_name=config["layout_name"],
        horizon=config["horizon"],
        num_envs=config["num_envs"],
        total_timesteps=config["total_timesteps"],
        learning_rate=config["learning_rate"],
        num_steps=config["num_steps"],
        num_minibatches=config["num_minibatches"],
        num_epochs=config["num_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_eps=config["clip_eps"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        num_hidden_layers=config["num_hidden_layers"],
        hidden_dim=config["hidden_dim"],
        num_filters=config["num_filters"],
        num_conv_layers=config["num_conv_layers"],
        use_lstm=config["use_lstm"],
        cell_size=config["cell_size"],
        reward_shaping_factor=config["reward_shaping_factor"],
        reward_shaping_horizon=config["reward_shaping_horizon"],
        use_phi=config["use_phi"],
        bc_schedule=config["bc_schedule"],
        bc_model_dir=config["bc_model_dir"],
        log_interval=config["log_interval"],
        save_interval=config["save_interval"],
        eval_interval=config["eval_interval"],
        eval_num_games=config["eval_num_games"],
        verbose=config["verbose"],
        results_dir=config["results_dir"],
        experiment_name=config["experiment_name"],
    )
    
    # Create trainer and run
    trainer = PPOTrainer(ppo_config)
    results = trainer.train()
    
    # Log final results to WandB
    if config["use_wandb"] and WANDB_AVAILABLE:
        wandb.log({
            "total_timesteps": results["total_timesteps"],
        })
        wandb.finish()
    
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agents for Overcooked AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment
    parser.add_argument(
        "--layout", "--layout_name",
        type=str,
        default=DEFAULT_CONFIG["layout_name"],
        help="Overcooked layout name"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_CONFIG["horizon"],
        help="Episode horizon"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=DEFAULT_CONFIG["num_envs"],
        help="Number of parallel environments"
    )
    
    # Training
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=DEFAULT_CONFIG["total_timesteps"],
        help="Total training timesteps"
    )
    parser.add_argument(
        "--learning_rate", "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed"
    )
    
    # BC schedule
    parser.add_argument(
        "--bc_model_dir",
        type=str,
        default=None,
        help="Path to BC model directory for BC-schedule training"
    )
    
    # Reward shaping
    parser.add_argument(
        "--use_phi",
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_CONFIG["use_phi"],
        help="Use potential-based reward shaping"
    )
    parser.add_argument(
        "--reward_shaping_factor",
        type=float,
        default=DEFAULT_CONFIG["reward_shaping_factor"],
        help="Reward shaping factor"
    )
    
    # Network
    parser.add_argument(
        "--use_lstm",
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_CONFIG["use_lstm"],
        help="Use LSTM network"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
        help="Hidden layer dimension"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_CONFIG["verbose"],
        help="Verbose output"
    )
    parser.add_argument(
        "--use_wandb",
        type=lambda x: x.lower() == 'true',
        default=DEFAULT_CONFIG["use_wandb"],
        help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=DEFAULT_CONFIG["wandb_project"],
        help="WandB project name"
    )
    
    # Output
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_CONFIG["results_dir"],
        help="Results directory"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name"
    )
    
    # Local mode
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local (testing) configuration"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build configuration
    overrides = {
        "layout_name": args.layout,
        "horizon": args.horizon,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "bc_model_dir": args.bc_model_dir,
        "use_phi": args.use_phi,
        "reward_shaping_factor": args.reward_shaping_factor,
        "use_lstm": args.use_lstm,
        "hidden_dim": args.hidden_dim,
        "verbose": args.verbose,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "results_dir": args.results_dir,
        "experiment_name": args.experiment_name,
    }
    
    config = get_config(local=args.local, **overrides)
    
    if config["verbose"]:
        print("=" * 60)
        print("PPO Training Configuration")
        print("=" * 60)
        for key, value in sorted(config.items()):
            if key != "bc_schedule":
                print(f"  {key}: {value}")
        print("=" * 60)
    
    # Run training
    try:
        results = run_training(config)
        
        if config["verbose"]:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print(f"Total timesteps: {results['total_timesteps']}")
            print("=" * 60)
        
        return results
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None


# Sacred experiment configuration (if available)
if SACRED_AVAILABLE and ex is not None:
    @ex.config
    def sacred_config():
        """Sacred configuration."""
        layout_name = DEFAULT_CONFIG["layout_name"]
        horizon = DEFAULT_CONFIG["horizon"]
        num_envs = DEFAULT_CONFIG["num_envs"]
        total_timesteps = DEFAULT_CONFIG["total_timesteps"]
        learning_rate = DEFAULT_CONFIG["learning_rate"]
        seed = DEFAULT_CONFIG["seed"]
        bc_model_dir = DEFAULT_CONFIG["bc_model_dir"]
        use_phi = DEFAULT_CONFIG["use_phi"]
        reward_shaping_factor = DEFAULT_CONFIG["reward_shaping_factor"]
        use_lstm = DEFAULT_CONFIG["use_lstm"]
        hidden_dim = DEFAULT_CONFIG["hidden_dim"]
        verbose = DEFAULT_CONFIG["verbose"]
        use_wandb = DEFAULT_CONFIG["use_wandb"]
        results_dir = DEFAULT_CONFIG["results_dir"]
        experiment_name = None

    @ex.main
    def sacred_main(
        layout_name, horizon, num_envs, total_timesteps, learning_rate,
        seed, bc_model_dir, use_phi, reward_shaping_factor, use_lstm,
        hidden_dim, verbose, use_wandb, results_dir, experiment_name
    ):
        """Sacred main function."""
        config = get_config(
            layout_name=layout_name,
            horizon=horizon,
            num_envs=num_envs,
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            seed=seed,
            bc_model_dir=bc_model_dir,
            use_phi=use_phi,
            reward_shaping_factor=reward_shaping_factor,
            use_lstm=use_lstm,
            hidden_dim=hidden_dim,
            verbose=verbose,
            use_wandb=use_wandb,
            results_dir=results_dir,
            experiment_name=experiment_name,
        )
        return run_training(config)


if __name__ == "__main__":
    if SACRED_AVAILABLE and len(sys.argv) > 1 and sys.argv[1] == "with":
        # Run with Sacred
        ex.run_commandline()
    else:
        # Run normally
        main()

