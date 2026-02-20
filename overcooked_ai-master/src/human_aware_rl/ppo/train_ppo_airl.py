"""
PPO with AIRL Partner Training Script for Overcooked AI.

This script trains PPO agents with an AIRL-trained human model as partner.
The AIRL model is trained on human demonstration data (replacing BC),
and the PPO agent learns to coordinate with it through a similar annealing
schedule as PPO_BC.

Usage:
    # Train all layouts with all seeds
    python -m human_aware_rl.ppo.train_ppo_airl --all_layouts --seeds 0,10,20,30,40
    
    # Train single layout with specific AIRL model
    python -m human_aware_rl.ppo.train_ppo_airl --layout cramped_room --airl_model_dir path/to/airl_model
    
    # Use default AIRL model paths
    python -m human_aware_rl.ppo.train_ppo_airl --all_layouts --use_default_airl_models
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
    get_ppo_bc_config,  # Reuse BC config as base
    PAPER_PPO_BC_CONFIGS,
)
from human_aware_rl.imitation.airl import AIRL_SAVE_DIR


# Default AIRL model paths (trained on human training data)
DEFAULT_AIRL_MODEL_PATHS = {
    layout: os.path.join(AIRL_SAVE_DIR, "train", layout, f"airl_{layout}_train", "final")
    for layout in PAPER_LAYOUTS
}


class AIRLPartnerWrapper:
    """
    Wrapper to use AIRL agent as a partner in PPO training.
    Matches the interface expected by the PPO trainer.
    """
    
    def __init__(self, airl_agent):
        self.airl_agent = airl_agent
    
    def action(self, state):
        """Get action from AIRL agent."""
        return self.airl_agent.action(state)


def train_ppo_airl(
    layout: str,
    seed: int = 0,
    airl_model_dir: Optional[str] = None,
    results_dir: str = "results",
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    fast: bool = False,
    **overrides
) -> Dict[str, Any]:
    """
    Train a PPO agent with AIRL partner for a specific layout.
    
    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        seed: Random seed
        airl_model_dir: Path to AIRL model directory (default: use default path)
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        fast: Use faster training settings
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
    from human_aware_rl.imitation.airl import load_airl_model
    from human_aware_rl.imitation.airl_agent import AIRLAgent
    from overcooked_ai_py.mdp.actions import Action
    
    # Use default AIRL model path if not specified
    if airl_model_dir is None:
        airl_model_dir = DEFAULT_AIRL_MODEL_PATHS.get(layout)
        if airl_model_dir is None:
            raise ValueError(f"No default AIRL model path for layout: {layout}")
    
    # Check if AIRL model exists
    if not os.path.exists(airl_model_dir):
        raise FileNotFoundError(
            f"AIRL model not found at {airl_model_dir}. "
            f"Please train AIRL models first using: "
            f"python -m human_aware_rl.imitation.train_airl --all_layouts"
        )
    
    # Get paper configuration for this layout (reuse BC config as base)
    config_dict = get_ppo_bc_config(
        layout=layout,
        seed=seed,
        bc_model_dir=airl_model_dir,  # Reusing bc_model_dir field
        results_dir=results_dir,
        verbose=verbose,
        **overrides
    )
    
    # Override experiment name
    config_dict["experiment_name"] = f"ppo_airl_{layout}_seed{seed}"
    
    if verbose:
        print("\n" + "="*60)
        print(f"Training PPO with AIRL Partner")
        print("="*60)
        print(f"Layout: {layout} -> {config_dict['layout_name']}")
        print(f"Seed: {seed}")
        print(f"AIRL model: {airl_model_dir}")
        print(f"Total timesteps: {config_dict['total_timesteps']:,}")
        print(f"Fast mode: {fast}")
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
    
    # Create custom PPO trainer that uses AIRL agent
    # We need to modify the PPO trainer's partner loading
    
    # First, create the PPO config
    bc_schedule = config_dict.get("bc_schedule", [(0, 1.0), (8e6, 0.0), (float('inf'), 0.0)])
    if isinstance(bc_schedule[0], tuple):
        bc_schedule_tuples = bc_schedule
    else:
        bc_schedule_tuples = [(int(bc_schedule[i]), float(bc_schedule[i+1])) 
                             for i in range(0, len(bc_schedule), 2)]
    
    # Apply fast settings if requested
    if fast:
        config_dict["total_timesteps"] = 1_000_000
        config_dict["early_stop_patience"] = 100
    
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
        bc_schedule=bc_schedule_tuples,  # Reuse BC schedule for AIRL
        bc_model_dir=None,  # We'll set this manually
        verbose=verbose,
        results_dir=results_dir,
        experiment_name=config_dict["experiment_name"],
        seed=seed,
    )
    
    # Create trainer
    trainer = PPOTrainer(ppo_config)
    
    # Load AIRL agent and set it as the partner (instead of BC agent)
    policy, _, airl_config = load_airl_model(airl_model_dir)
    
    def featurize_fn(state):
        return trainer.envs.envs[0].base_env.featurize_state_mdp(state)
    
    trainer.bc_agent = AIRLAgent(
        policy=policy,
        config=airl_config,
        featurize_fn=featurize_fn,
        agent_index=1,  # AIRL agent plays as agent 1
        stochastic=True,
    )
    
    # Update schedule to use AIRL partner
    trainer.config = ppo_config._replace(bc_schedule=bc_schedule_tuples) if hasattr(ppo_config, '_replace') else ppo_config
    
    # Train
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
        json_config["partner_type"] = "airl"
        json_config["airl_model_dir"] = airl_model_dir
        json_config["bc_schedule"] = str(bc_schedule_tuples)
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
    airl_model_base_dir: Optional[str] = None,
    fast: bool = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Train PPO_AIRL agents for all layouts with multiple seeds.
    
    Args:
        seeds: List of random seeds
        layouts: List of layouts to train (default: all paper layouts)
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        airl_model_base_dir: Base directory for AIRL models
        fast: Use faster training settings
        
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
        
        # Get AIRL model path for this layout
        if airl_model_base_dir:
            airl_model_dir = os.path.join(airl_model_base_dir, layout, f"airl_{layout}_train", "final")
        else:
            airl_model_dir = DEFAULT_AIRL_MODEL_PATHS.get(layout)
        
        for seed in seeds:
            current_run += 1
            
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# Run {current_run}/{total_runs}: {layout} (seed={seed})")
                print(f"{'#'*60}")
            
            try:
                results = train_ppo_airl(
                    layout=layout,
                    seed=seed,
                    airl_model_dir=airl_model_dir,
                    results_dir=results_dir,
                    verbose=verbose,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project,
                    fast=fast,
                )
                all_results[layout][seed] = results
                
            except Exception as e:
                import traceback
                print(f"Error training {layout} with seed {seed}: {e}")
                traceback.print_exc()
                all_results[layout][seed] = {"error": str(e)}
    
    return all_results


def print_summary(results: Dict[str, Dict[int, Dict[str, Any]]]):
    """Print a summary of training results."""
    print("\n" + "="*60)
    print("PPO_AIRL TRAINING SUMMARY")
    print("="*60)
    
    for layout, seed_results in results.items():
        print(f"\n{layout}:")
        print("-"*40)
        
        for seed, result in seed_results.items():
            if "error" in result:
                print(f"  Seed {seed}: ERROR - {result['error']}")
            else:
                timesteps = result.get("total_timesteps", "N/A")
                reward = result.get("best_mean_reward", "N/A")
                if isinstance(reward, float):
                    print(f"  Seed {seed}: {timesteps:,} timesteps, best_reward={reward:.2f}")
                else:
                    print(f"  Seed {seed}: {timesteps:,} timesteps completed")


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agents with AIRL partners for Overcooked AI",
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
        "--airl_model_dir",
        type=str,
        default=None,
        help="Path to AIRL model directory (for single layout)"
    )
    
    parser.add_argument(
        "--airl_model_base_dir",
        type=str,
        default=None,
        help="Base directory for AIRL models (for all layouts)"
    )
    
    parser.add_argument(
        "--use_default_airl_models",
        action="store_true",
        help="Use default AIRL model paths"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/ppo_airl",
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
        "--fast",
        action="store_true",
        help="Use faster settings (1M steps, early stopping)"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Override settings
    overrides = {}
    if args.timesteps:
        overrides["total_timesteps"] = args.timesteps
    
    if args.layout:
        # Train single layout
        airl_model_dir = args.airl_model_dir
        if airl_model_dir is None and args.use_default_airl_models:
            airl_model_dir = DEFAULT_AIRL_MODEL_PATHS.get(args.layout)
        
        results = train_ppo_airl(
            layout=args.layout,
            seed=args.seed,
            airl_model_dir=airl_model_dir,
            results_dir=args.results_dir,
            verbose=verbose,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            fast=args.fast,
            **overrides
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
            airl_model_base_dir=args.airl_model_base_dir,
            fast=args.fast,
        )
        print_summary(results)
        
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()

