"""
Population-Based Training Script for Overcooked AI.

This script trains PPO agents using Population-Based Training (PBT)
with the paper's hyperparameters.

Usage:
    # Train all layouts
    python -m human_aware_rl.ppo.train_pbt --all_layouts
    
    # Train single layout
    python -m human_aware_rl.ppo.train_pbt --layout cramped_room
    
    # Train with custom settings
    python -m human_aware_rl.ppo.train_pbt --layout cramped_room --population_size 4
"""

import argparse
import os
import sys
import json
from typing import Dict, List, Optional, Any

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
    get_pbt_config,
    PAPER_PBT_CONFIGS,
    PBT_COMMON_PARAMS,
)


def train_pbt(
    layout: str,
    results_dir: str = "results",
    verbose: bool = True,
    population_size: int = 8,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    **overrides
) -> Dict[str, Any]:
    """
    Train agents using PBT for a specific layout.
    
    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        results_dir: Directory to save results
        verbose: Whether to print progress
        population_size: Number of agents in population
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        **overrides: Additional config overrides
        
    Returns:
        Training results dictionary
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for PBT training. "
            "Install with: pip install jax jaxlib flax optax"
        )
    
    from human_aware_rl.jaxmarl.pbt import PBTConfig, PBTTrainer
    
    # Get paper configuration for this layout
    config_dict = get_pbt_config(
        layout=layout,
        results_dir=results_dir,
        verbose=verbose,
        population_size=population_size,
        **overrides
    )
    
    if verbose:
        print("\n" + "="*60)
        print(f"Training PBT Agents")
        print("="*60)
        print(f"Layout: {layout} -> {config_dict['layout_name']}")
        print(f"Population size: {config_dict['population_size']}")
        print(f"Total env steps: {config_dict['total_env_steps']:,.0f}")
        print(f"Initial learning rate: {config_dict['learning_rate']}")
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
    
    # Create PBT config
    pbt_config = PBTConfig(
        layout_name=config_dict["layout_name"],
        horizon=config_dict.get("horizon", 400),
        num_envs=min(config_dict.get("num_workers", 30), 8),
        old_dynamics=config_dict.get("old_dynamics", True),
        population_size=config_dict["population_size"],
        total_env_steps=int(config_dict["total_env_steps"]),
        ppo_iteration_timesteps=config_dict.get("ppo_iteration_timesteps", 40000),
        num_minibatches=config_dict.get("num_minibatches", 10),
        minibatch_size=config_dict.get("minibatch_size", 2000),
        num_epochs=config_dict.get("num_sgd_iter", 8),
        initial_learning_rate=config_dict["learning_rate"],
        initial_entropy_coeff=config_dict.get("initial_entropy_coeff", 0.5),
        initial_vf_coef=config_dict.get("initial_vf_coef", 0.1),
        num_hidden_layers=config_dict.get("num_hidden_layers", 3),
        hidden_dim=config_dict.get("hidden_dim", 64),
        num_filters=config_dict.get("num_filters", 25),
        num_conv_layers=config_dict.get("num_conv_layers", 3),
        reward_shaping_factor=config_dict.get("reward_shaping_factor", 1.0),
        reward_shaping_horizon=config_dict.get("reward_shaping_horizon", float('inf')),
        use_phi=config_dict.get("use_phi", False),
        mutation_prob=config_dict.get("mutation_prob", 0.33),
        mutation_factor_low=config_dict.get("mutation_factor_low", 0.75),
        mutation_factor_high=config_dict.get("mutation_factor_high", 1.25),
        verbose=verbose,
        results_dir=results_dir,
        experiment_name=config_dict["experiment_name"],
    )
    
    # Create trainer and train
    trainer = PBTTrainer(pbt_config)
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
                       if not callable(v) and not isinstance(v, list)}
        json.dump(json_config, f, indent=2)
    
    # Finish WandB
    if use_wandb:
        wandb.finish()
    
    return results


def train_all_layouts(
    layouts: Optional[List[str]] = None,
    results_dir: str = "results",
    verbose: bool = True,
    population_size: int = 8,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
) -> Dict[str, Dict[str, Any]]:
    """
    Train PBT agents for all layouts.
    
    Args:
        layouts: List of layouts to train (default: all paper layouts)
        results_dir: Directory to save results
        verbose: Whether to print progress
        population_size: Number of agents in population
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        
    Returns:
        Dictionary mapping layout names to results
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    
    all_results = {}
    
    for i, layout in enumerate(layouts):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# Layout {i+1}/{len(layouts)}: {layout}")
            print(f"{'#'*60}")
        
        try:
            results = train_pbt(
                layout=layout,
                results_dir=results_dir,
                verbose=verbose,
                population_size=population_size,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
            )
            all_results[layout] = results
            
        except Exception as e:
            print(f"Error training {layout}: {e}")
            all_results[layout] = {"error": str(e)}
    
    return all_results


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print a summary of training results."""
    print("\n" + "="*60)
    print("PBT TRAINING SUMMARY")
    print("="*60)
    
    for layout, result in results.items():
        if "error" in result:
            print(f"\n{layout}: ERROR - {result['error']}")
        else:
            timesteps = result.get("total_timesteps", "N/A")
            best_agent = result.get("best_agent")
            if best_agent:
                print(f"\n{layout}:")
                print(f"  Total timesteps: {timesteps:,}")
                print(f"  Best agent fitness: {best_agent.fitness:.1f}")
                print(f"  Best agent hyperparams:")
                for k, v in best_agent.hyperparams.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.2e}")
                    else:
                        print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Train PBT agents for Overcooked AI",
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
        "--population_size",
        type=int,
        default=8,
        help="Number of agents in population"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/pbt",
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
        help="Use reduced settings for local testing"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    population_size = args.population_size
    
    # Local testing overrides
    if args.local:
        population_size = 2
    
    if args.layout:
        # Train single layout
        results = train_pbt(
            layout=args.layout,
            results_dir=args.results_dir,
            verbose=verbose,
            population_size=population_size,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
        )
        print(f"\nTraining complete.")
        if "best_agent" in results:
            print(f"Best agent fitness: {results['best_agent'].fitness:.1f}")
        
    elif args.all_layouts:
        # Train all layouts
        results = train_all_layouts(
            layouts=PAPER_LAYOUTS,
            results_dir=args.results_dir,
            verbose=verbose,
            population_size=population_size,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
        )
        print_summary(results)
        
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()

