"""
Self-Play PPO Training Script for Overcooked AI.

This script trains PPO agents via self-play using the paper's hyperparameters.
It supports training all 5 layouts with multiple random seeds for reproducibility.

Usage:
    # Train all layouts with all seeds
    python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40
    
    # Train single layout
    python -m human_aware_rl.ppo.train_ppo_sp --layout cramped_room --seed 0
    
    # Train with custom output directory
    python -m human_aware_rl.ppo.train_ppo_sp --layout cramped_room --results_dir my_results/
"""

import argparse
import os
import sys
import json
from datetime import datetime
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
    get_ppo_sp_config,
    PAPER_PPO_SP_CONFIGS,
)


def train_ppo_sp(
    layout: str,
    seed: int = 0,
    results_dir: str = "results",
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    **overrides
) -> Dict[str, Any]:
    """
    Train a PPO agent via self-play for a specific layout.
    
    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        seed: Random seed
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
    
    # Get paper configuration for this layout
    config_dict = get_ppo_sp_config(
        layout=layout,
        seed=seed,
        results_dir=results_dir,
        verbose=verbose,
        **overrides
    )
    
    if verbose:
        print("\n" + "="*60)
        print(f"Training PPO Self-Play Agent")
        print("="*60)
        print(f"Layout: {layout} -> {config_dict['layout_name']}")
        print(f"Seed: {seed}")
        print(f"Total timesteps: {config_dict['total_timesteps']:,}")
        print(f"Num envs: {config_dict.get('num_workers', 30)}")
        print(f"Learning rate: {config_dict['learning_rate']}")
        print(f"VF coef: {config_dict['vf_coef']}")
        print(f"Ent coef: {config_dict.get('entropy_coeff_start', 0.1)}")
        print(f"Gamma: {config_dict['gamma']}")
        print(f"GAE lambda: {config_dict['gae_lambda']}")
        print(f"Clip epsilon: {config_dict['clip_eps']}")
        print(f"Clip epsilon end: {config_dict.get('clip_eps_end', 0.0)}")
        print(f"Clip end fraction: {config_dict.get('clip_end_fraction', 1.0)}")
        print(f"Cliprange schedule: {config_dict.get('cliprange_schedule', 'constant')}")
        print(f"Reward shaping horizon: {config_dict.get('reward_shaping_horizon', 2.5e6):.0e}")
        print(f"Legacy encoding: {config_dict.get('use_legacy_encoding', True)}")
        print(f"Old dynamics: {config_dict.get('old_dynamics', True)}")
        print(f"Results dir: {results_dir}")
        print("="*60 + "\n")
    
    # #region agent log
    # Debug instrumentation: Log config from train_ppo_sp (H1-H4)
    try:
        import json
        import time as time_module
        log_entry = json.dumps({
            "location": "train_ppo_sp.py:train_start",
            "message": "train_ppo_sp config",
            "data": {
                "paper_layout": layout,
                "env_layout": config_dict['layout_name'],
                "total_timesteps": config_dict['total_timesteps'],
                "clip_eps": config_dict['clip_eps'],
                "entropy_coeff_start": config_dict.get('entropy_coeff_start', 0.1),
                "entropy_coeff_end": config_dict.get('entropy_coeff_end', 0.1),
                "entropy_coeff_horizon": config_dict.get('entropy_coeff_horizon', 0),
                "reward_shaping_horizon": config_dict.get('reward_shaping_horizon', float('inf')),
            },
            "timestamp": int(time_module.time() * 1000),
            "sessionId": "debug-session",
            "hypothesisId": "H1,H2,H3,H4"
        })
        with open("/Users/theochen/Desktop/overcooked-reproduction/.cursor/debug.log", "a") as f:
            f.write(log_entry + "\n")
    except Exception:
        pass
    # #endregion
    
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
    
    # Create PPO config
    # Paper Table 2: 30 envs * 400 steps = 12,000 batch, minibatch_size = 2000
    num_envs = config_dict.get("num_workers", 30)
    
    ppo_config = PPOConfig(
        layout_name=config_dict["layout_name"],
        horizon=config_dict.get("horizon", 400),
        num_envs=num_envs,
        num_steps=config_dict.get("rollout_fragment_length", 400),  # Full episode
        total_timesteps=config_dict["total_timesteps"],
        learning_rate=config_dict["learning_rate"],
        gamma=config_dict["gamma"],
        gae_lambda=config_dict["gae_lambda"],
        clip_eps=config_dict["clip_eps"],
        clip_eps_end=config_dict.get("clip_eps_end", 0.0),
        clip_end_fraction=config_dict.get("clip_end_fraction", 1.0),
        cliprange_schedule=config_dict.get("cliprange_schedule", "constant"),
        vf_coef=config_dict["vf_coef"],
        ent_coef=config_dict.get("entropy_coeff_start", 0.1),  # Original: ENTROPY=0.1
        max_grad_norm=config_dict["max_grad_norm"],
        num_minibatches=config_dict.get("num_minibatches", 6),  # CORRECTED: Was 10
        num_hidden_layers=config_dict.get("num_hidden_layers", 3),
        hidden_dim=config_dict.get("hidden_dim", 64),
        num_filters=config_dict.get("num_filters", 25),
        num_conv_layers=config_dict.get("num_conv_layers", 3),
        use_lstm=config_dict.get("use_lstm", False),
        cell_size=config_dict.get("cell_size", 256),
        reward_shaping_factor=config_dict.get("reward_shaping_factor", 1.0),
        reward_shaping_horizon=config_dict.get("reward_shaping_horizon", 2.5e6),  # CORRECTED default
        use_phi=config_dict.get("use_phi", False),
        use_legacy_encoding=config_dict.get("use_legacy_encoding", True),  # ADDED: Use legacy encoding
        old_dynamics=config_dict.get("old_dynamics", True),  # ADDED: Use old dynamics
        entropy_coeff_start=config_dict.get("entropy_coeff_start", 0.1),  # Original: ENTROPY=0.1
        entropy_coeff_end=config_dict.get("entropy_coeff_end", 0.1),      # No annealing
        entropy_coeff_horizon=config_dict.get("entropy_coeff_horizon", 0), # No annealing
        use_entropy_annealing=config_dict.get("use_entropy_annealing", False),
        num_epochs=config_dict.get("num_sgd_iter", 8),
        log_interval=config_dict.get("log_interval", 1),
        save_interval=config_dict.get("save_interval", 50),
        eval_interval=config_dict.get("eval_interval", 25),
        eval_num_games=config_dict.get("evaluation_num_games", 5),  # ADDED
        early_stop_patience=config_dict.get("early_stop_patience", 100),
        use_early_stopping=config_dict.get("use_early_stopping", False),  # ADDED: Default off for paper repro
        verbose_debug=config_dict.get("verbose_debug", False),
        grad_diagnostics=config_dict.get("grad_diagnostics", False),
        verbose=verbose,
        results_dir=results_dir,
        experiment_name=config_dict["experiment_name"],
        seed=seed,
    )
    
    # Create trainer and train
    trainer = PPOTrainer(ppo_config)
    results = trainer.train()
    
    # Save final config
    run_dir = os.path.join(results_dir, config_dict["experiment_name"])
    os.makedirs(run_dir, exist_ok=True)
    
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        # Convert config to JSON-serializable format
        json_config = {k: v for k, v in config_dict.items() 
                       if not callable(v) and k != "bc_schedule"}
        json_config["bc_schedule"] = str(config_dict.get("bc_schedule", []))
        json.dump(json_config, f, indent=2)
    
    # Save metrics (including periodic evaluation results)
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        metrics = {}
        for k, v in results.items():
            if k == "train_info":
                continue  # Skip complex training info
            elif isinstance(v, (np.floating, np.integer)):
                metrics[k] = float(v)
            elif isinstance(v, np.ndarray):
                metrics[k] = v.tolist()
            elif isinstance(v, list):
                metrics[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                metrics[k] = v
        json.dump(metrics, f, indent=2)
    
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
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Train PPO self-play agents for all layouts with multiple seeds.
    
    Args:
        seeds: List of random seeds
        layouts: List of layouts to train (default: all paper layouts)
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        
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
        
        for seed in seeds:
            current_run += 1
            
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# Run {current_run}/{total_runs}: {layout} (seed={seed})")
                print(f"{'#'*60}")
            
            try:
                results = train_ppo_sp(
                    layout=layout,
                    seed=seed,
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
    print("TRAINING SUMMARY")
    print("="*60)
    
    for layout, seed_results in results.items():
        print(f"\n{layout}:")
        print("-"*40)
        
        for seed, result in seed_results.items():
            if "error" in result:
                print(f"  Seed {seed}: ERROR - {result['error']}")
            else:
                timesteps = result.get("total_timesteps", "N/A")
                print(f"  Seed {seed}: {timesteps:,} timesteps completed")


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO Self-Play agents for Overcooked AI",
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
        "--results_dir",
        type=str,
        default="results/ppo_sp",
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
    
    parser.add_argument(
        "--num_training_iters",
        type=int,
        default=None,
        help="Number of training iterations (overrides layout default)"
    )
    
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Enable early stopping (disabled by default for paper reproduction)"
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=None,
        help="Override value function loss coefficient (e.g., 1.0, 0.5, 0.25)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Override global gradient clipping threshold (e.g., 0.1, 0.3)"
    )
    parser.add_argument(
        "--verbose_debug",
        action="store_true",
        help="Enable detailed PPO debug diagnostics"
    )
    parser.add_argument(
        "--grad_diagnostics",
        action="store_true",
        help="Enable expensive per-loss-term gradient diagnostics"
    )
    parser.add_argument(
        "--cliprange_schedule",
        type=str,
        choices=["constant", "linear", "linear_to_end"],
        default=None,
        help="PPO cliprange schedule (Baselines-style)."
    )
    parser.add_argument(
        "--clip_eps",
        type=float,
        default=None,
        help="Override PPO clip epsilon start value."
    )
    parser.add_argument(
        "--clip_eps_end",
        type=float,
        default=None,
        help="Override PPO clip epsilon end value (used by linear_to_end)."
    )
    parser.add_argument(
        "--clip_end_fraction",
        type=float,
        default=None,
        help="Fraction of training to reach clip_eps_end for linear_to_end, then hold at end."
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
            "use_early_stopping": True,
            "early_stop_patience": 10,
        }
    elif args.fast:
        # Fast mode: use 10M timesteps (833 iters * 12000 batch = ~10M)
        local_overrides = {
            "total_timesteps": 10000000,  # 10M timesteps
            "use_early_stopping": False,  # Disable for paper reproduction
            "save_interval": 50,
            "log_interval": 1,
        }
    
    if args.timesteps:
        local_overrides["total_timesteps"] = args.timesteps
    
    if args.num_training_iters:
        # Convert iterations to timesteps (each iter = 12000 timesteps: 30 envs * 400 steps)
        local_overrides["total_timesteps"] = args.num_training_iters * 12000
    
    if args.use_early_stopping:
        local_overrides["use_early_stopping"] = True
    if args.vf_coef is not None:
        local_overrides["vf_coef"] = args.vf_coef
    if args.max_grad_norm is not None:
        local_overrides["max_grad_norm"] = args.max_grad_norm
    if args.verbose_debug:
        local_overrides["verbose_debug"] = True
    if args.grad_diagnostics:
        local_overrides["grad_diagnostics"] = True
    if args.cliprange_schedule is not None:
        local_overrides["cliprange_schedule"] = args.cliprange_schedule
    if args.clip_eps is not None:
        local_overrides["clip_eps"] = args.clip_eps
    if args.clip_eps_end is not None:
        local_overrides["clip_eps_end"] = args.clip_eps_end
    if args.clip_end_fraction is not None:
        local_overrides["clip_end_fraction"] = args.clip_end_fraction
    
    if args.layout:
        # Train single layout
        results = train_ppo_sp(
            layout=args.layout,
            seed=args.seed,
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
        )
        print_summary(results)
        
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()

