"""
PPO with Human Proxy Partner Training Script for Overcooked AI.

This script trains PPO agents with a Human Proxy (HP) model as partner.
The HP model is trained on held-out human test data, representing the
actual evaluation partner. This is the "gold standard" - an agent that
has direct access to the evaluation partner during training.

This represents an upper bound on what PPO_BC could achieve if the BC
training model perfectly matched the evaluation HP model.

Usage:
    # Train all layouts with all seeds
    python -m human_aware_rl.ppo.train_ppo_hp --all_layouts --seeds 0,10,20,30,40
    
    # Train single layout
    python -m human_aware_rl.ppo.train_ppo_hp --layout cramped_room --seed 0
    
    # Fast training for testing
    python -m human_aware_rl.ppo.train_ppo_hp --layout cramped_room --fast
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Check if JAX is available
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from human_aware_rl.ppo.configs.paper_configs import (
    PAPER_LAYOUTS,
    LAYOUT_TO_ENV,
    get_ppo_bc_config,
    PAPER_PPO_BC_CONFIGS,
)
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR


# Default HP model paths (trained on human TEST data - the evaluation partner)
DEFAULT_HP_MODEL_PATHS = {
    layout: os.path.join(BC_SAVE_DIR, "test", layout)
    for layout in PAPER_LAYOUTS
}


def train_ppo_hp(
    layout: str,
    seed: int = 0,
    hp_model_dir: Optional[str] = None,
    results_dir: str = "results",
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    **overrides
) -> Dict[str, Any]:
    """
    Train a PPO agent with Human Proxy partner for a specific layout.
    
    This is the "gold standard" experiment - training PPO directly with
    the evaluation partner (Human Proxy trained on test data).
    
    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        seed: Random seed
        hp_model_dir: Path to HP model directory (default: bc_runs/test/{layout})
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        **overrides: Additional config overrides
        
    Returns:
        Training results dictionary
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for PPO training. "
            "Install with: pip install jax jaxlib flax optax"
        )
    
    from human_aware_rl.jaxmarl.ppo import PPOConfig, PPOTrainer
    
    # Use default HP model path if not specified
    if hp_model_dir is None:
        hp_model_dir = DEFAULT_HP_MODEL_PATHS.get(layout)
        if hp_model_dir is None:
            raise ValueError(f"No default HP model path for layout: {layout}")
    
    # Check if HP model exists
    if not os.path.exists(hp_model_dir):
        raise FileNotFoundError(
            f"Human Proxy model not found at {hp_model_dir}. "
            f"Please train BC models first using: "
            f"python -m human_aware_rl.imitation.train_bc_models --all_layouts"
        )
    
    # Get paper configuration for this layout (same as PPO_BC)
    config_dict = get_ppo_bc_config(
        layout=layout,
        seed=seed,
        bc_model_dir=hp_model_dir,  # Use HP model instead of BC
        results_dir=results_dir,
        verbose=verbose,
        **overrides
    )
    
    # Update experiment name to reflect HP training
    config_dict["experiment_name"] = f"ppo_hp_{layout}_seed{seed}"
    
    if verbose:
        print("\n" + "="*60)
        print(f"Training PPO with Human Proxy Partner (Gold Standard)")
        print("="*60)
        print(f"Layout: {layout} -> {config_dict['layout_name']}")
        print(f"Seed: {seed}")
        print(f"HP model: {hp_model_dir}")
        print(f"Total timesteps: {config_dict['total_timesteps']:,}")
        print(f"BC schedule: {config_dict['bc_schedule']}")
        print(f"Results dir: {results_dir}")
        print("="*60 + "\n")
    
    # Initialize WandB if enabled
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=config_dict["experiment_name"],
                config=config_dict,
            )
        except ImportError:
            print("Warning: wandb not available, skipping logging")
            use_wandb = False
    
    # Set random seeds
    np.random.seed(seed)
    
    # Convert bc_schedule to proper format
    bc_schedule = config_dict.get("bc_schedule", [(0, 1.0), (8e6, 0.0), (float('inf'), 0.0)])
    if isinstance(bc_schedule[0], tuple):
        bc_schedule_tuples = bc_schedule
    else:
        bc_schedule_tuples = [(int(bc_schedule[i]), float(bc_schedule[i+1])) 
                             for i in range(0, len(bc_schedule), 2)]
    
    # Create PPO config
    num_envs = min(config_dict.get("num_workers", 32), 32)
    
    ppo_config = PPOConfig(
        layout_name=config_dict["layout_name"],
        horizon=config_dict.get("horizon", 400),
        num_envs=num_envs,
        num_steps=config_dict.get("rollout_fragment_length", 400),
        total_timesteps=config_dict["total_timesteps"],
        learning_rate=config_dict["learning_rate"],
        gamma=config_dict["gamma"],
        gae_lambda=config_dict["gae_lambda"],
        clip_eps=config_dict["clip_eps"],
        vf_coef=config_dict["vf_coef"],
        max_grad_norm=config_dict["max_grad_norm"],
        num_minibatches=config_dict.get("num_minibatches", 10),
        num_hidden_layers=config_dict.get("num_hidden_layers", 3),
        hidden_dim=config_dict.get("hidden_dim", 64),
        num_filters=config_dict.get("num_filters", 25),
        num_conv_layers=config_dict.get("num_conv_layers", 3),
        use_lstm=config_dict.get("use_lstm", False),
        cell_size=config_dict.get("cell_size", 256),
        reward_shaping_factor=config_dict.get("reward_shaping_factor", 1.0),
        reward_shaping_horizon=config_dict.get("reward_shaping_horizon", float('inf')),
        use_phi=config_dict.get("use_phi", False),
        entropy_coeff_start=config_dict.get("entropy_coeff_start", 0.2),
        entropy_coeff_end=config_dict.get("entropy_coeff_end", 0.1),
        entropy_coeff_horizon=config_dict.get("entropy_coeff_horizon", 3e5),
        use_entropy_annealing=True,
        num_epochs=config_dict.get("num_sgd_iter", 8),
        log_interval=config_dict.get("log_interval", 1),
        save_interval=config_dict.get("save_interval", 50),
        eval_interval=config_dict.get("eval_interval", 25),
        early_stop_patience=config_dict.get("early_stop_patience", 100),
        bc_schedule=bc_schedule_tuples,
        bc_model_dir=hp_model_dir,  # Use HP model
        verbose=verbose,
        results_dir=results_dir,
        experiment_name=config_dict["experiment_name"],
        seed=seed,
    )
    
    # Create trainer and train
    trainer = PPOTrainer(ppo_config)
    results = trainer.train()
    
    # Save final config
    config_path = os.path.join(
        results_dir,
        config_dict["experiment_name"],
        "config.json"
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json_config = {k: v for k, v in config_dict.items() 
                       if not callable(v) and k != "bc_schedule"}
        json_config["bc_schedule"] = str(bc_schedule_tuples)
        json_config["hp_model_dir"] = hp_model_dir
        json_config["is_gold_standard"] = True
        json.dump(json_config, f, indent=2)
    
    # Finish WandB
    if use_wandb:
        wandb.finish()
    
    return results


def train_all_layouts(
    seeds: List[int],
    layouts: Optional[List[str]] = None,
    results_dir: str = "results",
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    hp_model_base_dir: Optional[str] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Train PPO_HP agents for all layouts with multiple seeds.
    
    Args:
        seeds: List of random seeds
        layouts: List of layouts to train (default: all paper layouts)
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        hp_model_base_dir: Base directory for HP models (default: BC_SAVE_DIR/test)
        
    Returns:
        Nested dict: {layout: {seed: results}}
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    
    all_results = {}
    total_runs = len(layouts) * len(seeds)
    current_run = 0
    
    for layout in layouts:
        all_results[layout] = {}
        
        # Get HP model path for this layout
        if hp_model_base_dir:
            hp_model_dir = os.path.join(hp_model_base_dir, layout)
        else:
            hp_model_dir = DEFAULT_HP_MODEL_PATHS.get(layout)
        
        for seed in seeds:
            current_run += 1
            
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# Run {current_run}/{total_runs}: {layout} (seed={seed})")
                print(f"# Gold Standard: PPO trained with Human Proxy")
                print(f"{'#'*60}")
            
            try:
                results = train_ppo_hp(
                    layout=layout,
                    seed=seed,
                    hp_model_dir=hp_model_dir,
                    results_dir=results_dir,
                    verbose=verbose,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project,
                )
                all_results[layout][seed] = results
                
            except Exception as e:
                print(f"Error training {layout} with seed {seed}: {e}")
                all_results[layout][seed] = {"error": str(e)}
    
    return all_results


def print_summary(results: Dict[str, Dict[int, Dict[str, Any]]]):
    """Print a summary of training results."""
    print("\n" + "="*60)
    print("PPO_HP (GOLD STANDARD) TRAINING SUMMARY")
    print("="*60)
    
    for layout, seed_results in results.items():
        print(f"\n{layout}:")
        print("-"*40)
        
        for seed, result in seed_results.items():
            if "error" in result:
                print(f"  Seed {seed}: ERROR - {result['error']}")
            else:
                timesteps = result.get("total_timesteps", "N/A")
                best_reward = result.get("best_mean_reward", "N/A")
                if isinstance(best_reward, float):
                    print(f"  Seed {seed}: {timesteps:,} timesteps, best reward: {best_reward:.1f}")
                else:
                    print(f"  Seed {seed}: {timesteps:,} timesteps completed")


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agents with Human Proxy partners (Gold Standard) for Overcooked AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        choices=PAPER_LAYOUTS,
        help="Train a single layout"
    )
    
    parser.add_argument(
        "--all_layouts",
        action="store_true",
        help="Train all 5 paper layouts"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (for single layout)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,10,20,30,40",
        help="Comma-separated list of seeds (for all layouts)"
    )
    
    parser.add_argument(
        "--hp_model_dir",
        type=str,
        default=None,
        help="Path to HP model directory (for single layout)"
    )
    
    parser.add_argument(
        "--hp_model_base_dir",
        type=str,
        default=None,
        help="Base directory for HP models (for all layouts)"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/ppo_hp",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging"
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="overcooked-ai",
        help="WandB project name"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use reduced settings for local testing (10k steps)"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings (1M steps, early stopping)"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps (e.g., 500000 for 500k)"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Override settings based on flags
    local_overrides = {}
    if args.local:
        local_overrides = {
            "total_timesteps": 10000,
            "num_workers": 4,
            "early_stop_patience": 10,
        }
    elif args.fast:
        local_overrides = {
            "total_timesteps": 1000000,
            "num_workers": 32,
            "early_stop_patience": 100,
            "save_interval": 25,
            "log_interval": 1,
        }
    
    if args.timesteps:
        local_overrides["total_timesteps"] = args.timesteps
    
    if args.layout:
        # Train single layout
        hp_model_dir = args.hp_model_dir
        if hp_model_dir is None:
            hp_model_dir = DEFAULT_HP_MODEL_PATHS.get(args.layout)
        
        results = train_ppo_hp(
            layout=args.layout,
            seed=args.seed,
            hp_model_dir=hp_model_dir,
            results_dir=args.results_dir,
            verbose=verbose,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            **local_overrides
        )
        print(f"\nTraining complete. Results: {results}")
        
    elif args.all_layouts:
        # Train all layouts
        results = train_all_layouts(
            seeds=seeds,
            layouts=PAPER_LAYOUTS,
            results_dir=args.results_dir,
            verbose=verbose,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            hp_model_base_dir=args.hp_model_base_dir,
        )
        print_summary(results)
        
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()

