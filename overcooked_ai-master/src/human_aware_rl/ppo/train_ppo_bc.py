"""
PPO with BC Partner Training Script for Overcooked AI.

This script trains PPO agents with a BC (Behavior Cloning) human model as partner.
The BC model is trained on human demonstration data, and the PPO agent learns
to coordinate with it through a self-play schedule that starts at 100% self-play
and anneals to 100% BC partner (matching the original paper's SELF_PLAY_HORIZON).

Usage:
    # Train all layouts with all seeds
    python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0,10,20,30,40
    
    # Train single layout with specific BC model
    python -m human_aware_rl.ppo.train_ppo_bc --layout cramped_room --bc_model_dir path/to/bc_model
    
    # Use default BC model paths
    python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --use_default_bc_models
"""

import argparse
import os
import sys
import json
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
from human_aware_rl.ppo.run_paths import (
    build_run_name,
    build_training_output_paths,
    default_ppo_data_dir,
)
from human_aware_rl.ppo.run_registry_defaults import (
    get_default_agent_dir,
    get_default_run_name,
)


PARTNER_TYPE_TO_SPLIT = {
    "bc_train": "train",
    "bc_test": "test",
}

DEFAULT_AGENT_NAME = get_default_agent_dir("ppo_bc")


def train_ppo_bc(
    layout: str,
    seed: int = 0,
    bc_model_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    use_legacy_results_layout: bool = False,
    ex_name: Optional[str] = None,
    timestamp_dir: bool = False,
    ppo_data_dir: Optional[str] = None,
    partner_type: str = "bc_train",
    agent_name: str = DEFAULT_AGENT_NAME,
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    canonical_paper_entrypoint: bool = True,
    **overrides
) -> Dict[str, Any]:
    """
    Train a PPO agent with BC partner for a specific layout.
    
    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        seed: Random seed
        bc_model_dir: Path to BC model directory (default: use default path)
        results_dir: Legacy output directory. Ignored unless use_legacy_results_layout=True.
        use_legacy_results_layout: If True, use legacy results_dir/experiment_name layout.
        ex_name: Experiment/run name (used in DATA_DIR/ppo_runs layout)
        timestamp_dir: Prefix run directory with timestamp
        ppo_data_dir: Base directory for canonical ppo_runs storage
        partner_type: Which BC split to use (bc_train or bc_test)
        agent_name: Agent subdirectory name under each seed dir
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
    
    if partner_type not in PARTNER_TYPE_TO_SPLIT:
        raise ValueError(
            f"Unknown partner_type '{partner_type}'. "
            f"Expected one of: {list(PARTNER_TYPE_TO_SPLIT.keys())}"
        )

    # Use default BC model path if not specified
    if bc_model_dir is None:
        split = PARTNER_TYPE_TO_SPLIT[partner_type]
        bc_model_dir = os.path.join(BC_SAVE_DIR, split, layout)
        if bc_model_dir is None:
            raise ValueError(f"No default BC model path for layout: {layout}")
    
    # Check if BC model exists
    if not os.path.exists(bc_model_dir):
        raise FileNotFoundError(
            f"BC model not found at {bc_model_dir}. "
            f"Please train BC models first using: "
            f"python -m human_aware_rl.imitation.train_bc_models --all_layouts"
        )
    
    # Get paper configuration for this layout
    config_dict = get_ppo_bc_config(
        layout=layout,
        seed=seed,
        bc_model_dir=bc_model_dir,
        verbose=verbose,
        **overrides
    )
    
    # Use paper-parity legacy layout mapping from paper_configs.py.
    config_dict["layout_name"] = LAYOUT_TO_ENV.get(layout, layout)
    
    # Canonical run naming/layout mirrors deprecated repo:
    #   DATA_DIR/ppo_runs/<run_name>/seed<seed>/<agent_name>/checkpoint_*
    if ex_name is None:
        ex_name = get_default_run_name("ppo_bc", layout=layout, partner_type=partner_type)
    resolved_run_name = build_run_name(ex_name, timestamp_dir=timestamp_dir)
    ppo_data_dir = ppo_data_dir or default_ppo_data_dir()
    output_paths = build_training_output_paths(
        ppo_data_dir=ppo_data_dir,
        run_name=resolved_run_name,
        seed=seed,
        agent_name=agent_name,
    )

    trainer_results_dir = output_paths["trainer_results_dir"]
    trainer_experiment_name = output_paths["trainer_experiment_name"]

    if use_legacy_results_layout and results_dir and verbose:
        print(
            "[WARN] --use_legacy_results_layout is ignored. "
            "Checkpoints are always written under DATA_DIR/ppo_runs/<run>/seed<seed>/<agent>/"
        )

    if verbose:
        print("\n" + "="*60)
        print(f"Training PPO with BC Partner (Paper Table 3)")
        print("="*60)
        print(f"Layout: {layout} -> {config_dict['layout_name']}")
        print(f"Seed: {seed}")
        print(f"Partner type: {partner_type}")
        print(f"BC model: {bc_model_dir}")
        print(f"Total timesteps: {config_dict['total_timesteps']:,}")
        print(f"Learning rate: {config_dict['learning_rate']}")
        lr_factor = config_dict.get('lr_annealing_factor', 1.0)
        if lr_factor > 1.0:
            final_lr = config_dict['learning_rate'] / lr_factor
            print(f"LR annealing: factor={lr_factor} ({config_dict['learning_rate']:.2e} -> {final_lr:.2e})")
        else:
            print(f"LR annealing: disabled (constant)")
        print(f"VF coef: {config_dict['vf_coef']}")
        print(f"Reward shaping horizon: {config_dict.get('reward_shaping_horizon', 'inf'):,.0f}")
        print(f"Num minibatches: {config_dict.get('num_minibatches', 6)}")
        print(f"Num envs: {config_dict.get('num_workers', 30)} (batch={config_dict.get('num_workers', 30)*400:,})")
        print(f"Clip epsilon (start): {config_dict['clip_eps']}")
        print(f"Cliprange schedule: {config_dict.get('cliprange_schedule', 'constant')}")
        print(f"Clip epsilon end: {config_dict.get('clip_eps_end', 0.0)}")
        print(f"Clip end fraction: {config_dict.get('clip_end_fraction', 1.0)}")
        print(f"BC schedule: {config_dict['bc_schedule']}")
        print(f"Run name: {resolved_run_name}")
        print(f"PPO data dir: {ppo_data_dir}")
        print(f"Seed dir: {output_paths['seed_dir']}")
        print(f"Agent dir: {output_paths['agent_dir']}")
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
    # Use num_workers from config (Paper Table 3: 30 envs for 12,000 batch)
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
        # Entropy: fixed at 0.1 (original: ENTROPY=0.1, never overridden)
        entropy_coeff_start=config_dict.get("entropy_coeff_start", 0.1),
        entropy_coeff_end=config_dict.get("entropy_coeff_end", 0.1),
        entropy_coeff_horizon=config_dict.get("entropy_coeff_horizon", 0),
        use_entropy_annealing=config_dict.get("use_entropy_annealing", False),
        # LR annealing: Paper Table 3 uses factor-based annealing for PPO_BC
        use_lr_annealing=config_dict.get("use_lr_annealing", False),
        lr_annealing_factor=config_dict.get("lr_annealing_factor", 1.0),
        num_epochs=config_dict.get("num_sgd_iter", 8),
        log_interval=config_dict.get("log_interval", 1),
        save_interval=config_dict.get("save_interval", 50),
        eval_interval=config_dict.get("eval_interval", 25),
        early_stop_patience=config_dict.get("early_stop_patience", 100),
        bc_schedule=bc_schedule_tuples,
        bc_model_dir=bc_model_dir,
        verbose=verbose,
        results_dir=trainer_results_dir,
        experiment_name=trainer_experiment_name,
        seed=seed,
        canonical_paper_entrypoint=canonical_paper_entrypoint,
    )
    
    # Create trainer and train
    trainer = PPOTrainer(ppo_config)
    results = trainer.train()
    
    # Save final config
    config_path = os.path.join(
        trainer_results_dir,
        trainer_experiment_name,
        "config.json"
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json_config = {k: v for k, v in config_dict.items() 
                       if not callable(v) and k != "bc_schedule"}
        json_config["bc_schedule"] = str(bc_schedule_tuples)
        json_config["bc_model_dir"] = bc_model_dir
        json_config["partner_type"] = partner_type
        json_config["agent_name"] = trainer_experiment_name
        json_config["seed_dir"] = output_paths["seed_dir"]
        json_config["run_name"] = resolved_run_name
        json_config["ppo_data_dir"] = ppo_data_dir
        json_config["legacy_results_layout"] = use_legacy_results_layout
        json.dump(json_config, f, indent=2)
    
    # Finish WandB
    if use_wandb:
        wandb.finish()
    
    return results


def train_all_layouts(
    seeds: List[int],
    layouts: Optional[List[str]] = None,
    results_dir: Optional[str] = None,
    use_legacy_results_layout: bool = False,
    ex_name_prefix: Optional[str] = None,
    timestamp_dir: bool = False,
    ppo_data_dir: Optional[str] = None,
    partner_type: str = "bc_train",
    agent_name: str = DEFAULT_AGENT_NAME,
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    bc_model_base_dir: Optional[str] = None,
    canonical_paper_entrypoint: bool = True,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Train PPO_BC agents for all layouts with multiple seeds.
    
    Args:
        seeds: List of random seeds
        layouts: List of layouts to train (default: all paper layouts)
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        bc_model_base_dir: Base directory for BC models (default: BC_SAVE_DIR/train)
        
    Returns:
        Nested dict: {layout: {seed: results}}
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    
    if partner_type not in PARTNER_TYPE_TO_SPLIT:
        raise ValueError(
            f"Unknown partner_type '{partner_type}'. "
            f"Expected one of: {list(PARTNER_TYPE_TO_SPLIT.keys())}"
        )

    split = PARTNER_TYPE_TO_SPLIT[partner_type]
    default_base = os.path.join(BC_SAVE_DIR, split)
    bc_model_base = bc_model_base_dir or default_base
    if bc_model_base_dir:
        normalized_base = os.path.normpath(bc_model_base_dir)
        base_tail = os.path.basename(normalized_base)
        if base_tail in {"train", "test"} and base_tail != split and verbose:
            print(
                f"[WARN] bc_model_base_dir='{bc_model_base_dir}' conflicts with "
                f"partner_type='{partner_type}' (split='{split}'). "
                "Using explicit bc_model_base_dir override."
            )

    all_results = {}
    total_runs = len(layouts) * len(seeds)
    current_run = 0
    
    for layout in layouts:
        all_results[layout] = {}
        
        # Resolve split-aware BC model path for this layout.
        bc_model_dir = os.path.join(bc_model_base, layout)
        
        for seed in seeds:
            current_run += 1
            
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# Run {current_run}/{total_runs}: {layout} (seed={seed})")
                print(f"{'#'*60}")
            
            try:
                results = train_ppo_bc(
                    layout=layout,
                    seed=seed,
                    bc_model_dir=bc_model_dir,
                    results_dir=results_dir,
                    use_legacy_results_layout=use_legacy_results_layout,
                    ex_name=(
                        f"{ex_name_prefix}__layout-{layout}"
                        if ex_name_prefix
                        else get_default_run_name(
                            "ppo_bc", layout=layout, partner_type=partner_type
                        )
                    ),
                    timestamp_dir=timestamp_dir,
                    ppo_data_dir=ppo_data_dir,
                    partner_type=partner_type,
                    agent_name=agent_name,
                    verbose=verbose,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project,
                    canonical_paper_entrypoint=canonical_paper_entrypoint,
                )
                all_results[layout][seed] = results
                
            except Exception as e:
                print(f"Error training {layout} with seed {seed}: {e}")
                all_results[layout][seed] = {"error": str(e)}
    
    return all_results


def print_summary(results: Dict[str, Dict[int, Dict[str, Any]]]):
    """Print a summary of training results."""
    print("\n" + "="*60)
    print("PPO_BC TRAINING SUMMARY")
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
        description="Train PPO agents with BC partners for Overcooked AI",
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
        "--bc_model_dir",
        type=str,
        default=None,
        help="Path to BC model directory (for single layout)"
    )
    
    parser.add_argument(
        "--bc_model_base_dir",
        type=str,
        default=None,
        help="Base directory for BC models (for all layouts)"
    )
    
    parser.add_argument(
        "--use_default_bc_models",
        action="store_true",
        help="Use default BC model paths"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Legacy output directory (ignored unless --use_legacy_results_layout)"
    )
    parser.add_argument(
        "--use_legacy_results_layout",
        action="store_true",
        help="Use legacy results_dir/experiment_name layout instead of DATA_DIR/ppo_runs"
    )
    parser.add_argument(
        "--ex_name",
        type=str,
        default=None,
        help="Deprecated-style experiment name (run directory name without timestamp)"
    )
    parser.add_argument(
        "--timestamp_dir",
        action="store_true",
        help="Prefix run directory with timestamp"
    )
    parser.add_argument(
        "--ppo_data_dir",
        type=str,
        default=None,
        help="Base PPO run directory (default: DATA_DIR/ppo_runs)"
    )
    parser.add_argument(
        "--partner_type",
        type=str,
        choices=["bc_train", "bc_test"],
        default="bc_train",
        help="Partner split: bc_train (train demos) or bc_test (test demos / HP)"
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default=DEFAULT_AGENT_NAME,
        help="Agent subdirectory name under each seed directory"
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
        "--paper",
        dest="paper",
        action="store_true",
        default=True,
        help="Enable strict canonical paper mode"
    )
    parser.add_argument(
        "--not_paper",
        dest="paper",
        action="store_false",
        help="Disable canonical paper mode (ablation/experimental runs)"
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
    
    args = parser.parse_args()
    ablation_requested = any([
        args.local,
        args.fast,
        args.timesteps is not None,
        args.num_training_iters is not None,
        args.use_early_stopping,
    ])
    if args.paper and ablation_requested:
        parser.error(
            "--paper mode cannot be combined with ablation flags "
            "(e.g., --fast/--local/--timesteps). Use --not_paper."
        )
    canonical_paper_entrypoint = args.paper and not ablation_requested

    def _ablation_name(name: str) -> str:
        return name if name.startswith("ablation__") else f"ablation__{name}"

    
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
        local_overrides = {
            "total_timesteps": 1000000,  # 1M instead of paper's 10M
            "use_early_stopping": True,
            "early_stop_patience": 100,
            "save_interval": 25,
            "log_interval": 1,
        }
    
    if args.timesteps:
        local_overrides["total_timesteps"] = args.timesteps
    
    if args.num_training_iters:
        # Convert iterations to timesteps (each iter = 12000 timesteps: 30 envs * 400 steps)
        local_overrides["total_timesteps"] = args.num_training_iters * 12000
    
    if args.use_early_stopping:
        local_overrides["use_early_stopping"] = True
    
    if args.layout:
        # Train single layout
        bc_model_dir = args.bc_model_dir
        if bc_model_dir is None and args.bc_model_base_dir:
            # Construct path from base directory
            bc_model_dir = os.path.join(args.bc_model_base_dir, args.layout)
        elif bc_model_dir is None and args.use_default_bc_models:
            split = PARTNER_TYPE_TO_SPLIT[args.partner_type]
            bc_model_dir = os.path.join(BC_SAVE_DIR, split, args.layout)
        
        run_name = args.ex_name or get_default_run_name(
            "ppo_bc", layout=args.layout, partner_type=args.partner_type
        )
        if not canonical_paper_entrypoint:
            run_name = _ablation_name(run_name)
        results = train_ppo_bc(
            layout=args.layout,
            seed=args.seed,
            bc_model_dir=bc_model_dir,
            results_dir=args.results_dir,
            use_legacy_results_layout=args.use_legacy_results_layout,
            ex_name=run_name,
            timestamp_dir=args.timestamp_dir,
            ppo_data_dir=args.ppo_data_dir,
            partner_type=args.partner_type,
            agent_name=args.agent_name,
            verbose=verbose,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            canonical_paper_entrypoint=canonical_paper_entrypoint,
            **local_overrides
        )
        print(f"\nTraining complete. Results: {results}")
        
    elif args.all_layouts:
        # Train all layouts
        ex_name_prefix = args.ex_name
        if not canonical_paper_entrypoint:
            default_prefix = f"ablation__ppo_bc__partner-{args.partner_type}"
            ex_name_prefix = _ablation_name(ex_name_prefix) if ex_name_prefix else default_prefix
        results = train_all_layouts(
            seeds=seeds,
            layouts=PAPER_LAYOUTS,
            results_dir=args.results_dir,
            use_legacy_results_layout=args.use_legacy_results_layout,
            ex_name_prefix=ex_name_prefix,
            timestamp_dir=args.timestamp_dir,
            ppo_data_dir=args.ppo_data_dir,
            partner_type=args.partner_type,
            agent_name=args.agent_name,
            verbose=verbose,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            bc_model_base_dir=args.bc_model_base_dir,
            canonical_paper_entrypoint=canonical_paper_entrypoint,
        )
        print_summary(results)
        
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()

