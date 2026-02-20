"""
Train PPO agents with GAIL partner (PPO_GAIL).

Similar to PPO_BC but uses a GAIL-trained policy as the partner instead of BC.
Supports two modes:

  --controlled (default): Uses the EXACT same PPO hyperparameters as PPO_BC
      (Paper Table 3). The partner model is the sole independent variable.
      This is the primary, fair comparison against PPO_BC.

  --optimized: Uses Bayesian-optimized hyperparameters but matches PPO_BC on
      reward shaping horizon and total timesteps (10M). This is an ablation
      that shows the ceiling of GAIL with tuned HPs.

Usage:
    # Fair comparison (default: controlled mode, same HPs as PPO_BC)
    python -m human_aware_rl.ppo.train_ppo_gail --layout cramped_room --seed 0

    # All layouts with all seeds (controlled mode)
    python -m human_aware_rl.ppo.train_ppo_gail --all_layouts --seeds 0,10,20,30,40

    # Optimized ablation mode
    python -m human_aware_rl.ppo.train_ppo_gail --layout cramped_room --optimized

    # Custom GAIL model directory
    python -m human_aware_rl.ppo.train_ppo_gail --layout cramped_room \\
        --gail_model_base_dir path/to/gail_models
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

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.ppo.configs.paper_configs import (
    PAPER_LAYOUTS,
    LAYOUT_TO_ENV,
    get_ppo_gail_config,
    get_ppo_gail_optimized_config,
    PAPER_PPO_GAIL_CONFIGS,
    BAYESIAN_OPTIMIZED_GAIL_PARAMS,
)

# Output directories
PPO_GAIL_SAVE_DIR = os.path.join(DATA_DIR, "ppo_gail_runs")

# Default GAIL model paths
try:
    from human_aware_rl.imitation.gail import GAIL_SAVE_DIR
except ImportError:
    GAIL_SAVE_DIR = os.path.join(DATA_DIR, "gail_runs")

DEFAULT_GAIL_MODEL_PATHS = {
    layout: os.path.join(GAIL_SAVE_DIR, layout)
    for layout in PAPER_LAYOUTS
}


def _load_gail_agent(gail_model_dir: str, layout: str, env_layout: str):
    """
    Load a GAIL model and wrap it as an agent compatible with PPOTrainer.

    Args:
        gail_model_dir: Path to GAIL model directory for this layout
        layout: Paper layout name
        env_layout: Environment layout name (legacy)

    Returns:
        GAILAgentWrapper instance that can be used as PPOTrainer.bc_agent
    """
    import torch
    from human_aware_rl.imitation.gail import GAILPolicy
    from overcooked_ai_py.agents.benchmarking import AgentEvaluator
    from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
    from overcooked_ai_py.mdp.actions import Action

    gail_checkpoint_path = os.path.join(gail_model_dir, "model.pt")

    if not os.path.exists(gail_checkpoint_path):
        raise FileNotFoundError(
            f"No GAIL model found at {gail_checkpoint_path}. "
            f"Run: python -m human_aware_rl.imitation.gail --layout {layout}"
        )

    # Setup environment to get dimensions
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params=DEFAULT_ENV_PARAMS,
    )

    def featurize_fn(state):
        return ae.env.featurize_state_mdp(state)

    # Get state/action dims
    dummy_state = ae.env.mdp.get_standard_start_state()
    obs_shape = featurize_fn(dummy_state)[0].shape
    state_dim = int(np.prod(obs_shape))
    action_dim = len(Action.ALL_ACTIONS)

    # Load GAIL policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gail_policy = GAILPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
    ).to(device)

    checkpoint = torch.load(gail_checkpoint_path, map_location=device)
    gail_policy.load_state_dict(checkpoint["policy_state_dict"])
    gail_policy.eval()

    class GAILAgentWrapper:
        """Wrapper for GAIL policy to work with PPO training (same interface as BCAgent)."""

        def __init__(self, policy, featurize_fn, stochastic=True):
            self.policy = policy
            self.featurize_fn = featurize_fn
            self.stochastic = stochastic
            self.agent_index = 1  # GAIL partner is usually player 1

        def action(self, state):
            """Get action for state. Returns (action, info) like BCAgent."""
            obs = self.featurize_fn(state)[self.agent_index]
            obs_flat = obs.flatten()
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = self.policy(obs_tensor)
                action_probs = torch.softmax(logits, dim=-1)

                if self.stochastic:
                    action_idx = torch.multinomial(action_probs, 1).item()
                else:
                    action_idx = action_probs.argmax(dim=-1).item()

            action = Action.INDEX_TO_ACTION[action_idx]
            info = {"action_probs": action_probs.cpu().numpy()}
            return action, info

        def set_agent_index(self, index):
            self.agent_index = index

        def reset(self):
            pass

    return GAILAgentWrapper(gail_policy, featurize_fn, stochastic=True)


def train_ppo_gail(
    layout: str,
    seed: int = 0,
    gail_model_dir: Optional[str] = None,
    results_dir: str = "results/ppo_gail",
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    optimized: bool = False,
    **overrides
) -> Dict[str, Any]:
    """
    Train a PPO agent with GAIL partner for a specific layout.

    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        seed: Random seed
        gail_model_dir: Path to GAIL model directory (default: use default path)
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        optimized: If True, use Bayesian-optimized HPs (ablation). Default False
                   uses Paper Table 3 HPs (fair controlled comparison).
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

    # Use default GAIL model path if not specified
    if gail_model_dir is None:
        gail_model_dir = DEFAULT_GAIL_MODEL_PATHS.get(layout)
        if gail_model_dir is None:
            raise ValueError(f"No default GAIL model path for layout: {layout}")

    # Get configuration based on mode
    mode_name = "optimized" if optimized else "controlled"
    if optimized:
        config_dict = get_ppo_gail_optimized_config(
            layout=layout,
            seed=seed,
            gail_model_dir=gail_model_dir,
            results_dir=results_dir,
            **overrides
        )
    else:
        config_dict = get_ppo_gail_config(
            layout=layout,
            seed=seed,
            gail_model_dir=gail_model_dir,
            results_dir=results_dir,
            **overrides
        )

    # Get environment layout name
    env_layout = config_dict["layout_name"]

    if verbose:
        print("\n" + "=" * 60)
        print(f"Training PPO_GAIL ({mode_name} mode)")
        print("=" * 60)
        print(f"Layout: {layout} -> {env_layout}")
        print(f"Seed: {seed}")
        print(f"GAIL model: {gail_model_dir}")
        print(f"Total timesteps: {config_dict['total_timesteps']:,}")
        print(f"Learning rate: {config_dict['learning_rate']}")
        lr_factor = config_dict.get('lr_annealing_factor', 1.0)
        use_lr_ann = config_dict.get('use_lr_annealing', False)
        if use_lr_ann and lr_factor > 1.0:
            final_lr = config_dict['learning_rate'] / lr_factor
            print(f"LR annealing: factor={lr_factor} "
                  f"({config_dict['learning_rate']:.2e} -> {final_lr:.2e})")
        else:
            print(f"LR annealing: disabled (constant)")
        print(f"VF coef: {config_dict['vf_coef']}")
        print(f"Reward shaping horizon: {config_dict.get('reward_shaping_horizon', 'inf'):,.0f}")
        print(f"Num minibatches: {config_dict.get('num_minibatches', 6)}")
        num_envs = config_dict.get('num_workers', 30)
        print(f"Num envs: {num_envs} (batch={num_envs * 400:,})")
        print(f"Entropy: start={config_dict.get('entropy_coeff_start', 0.1)}, "
              f"end={config_dict.get('entropy_coeff_end', 0.1)}, "
              f"annealing={config_dict.get('use_entropy_annealing', False)}")
        print(f"BC schedule: {config_dict['bc_schedule']}")
        print(f"Results dir: {results_dir}")
        if optimized:
            print(f"  [OPTIMIZED MODE: Bayesian HPs with controlled budget]")
        else:
            print(f"  [CONTROLLED MODE: Paper Table 3 HPs, partner=GAIL]")
        print("=" * 60 + "\n")

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
        bc_schedule_tuples = [(int(bc_schedule[i]), float(bc_schedule[i + 1]))
                              for i in range(0, len(bc_schedule), 2)]

    # Create PPO config (mirrors train_ppo_bc.py PPOConfig creation)
    num_envs = config_dict.get("num_workers", 30)

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
        # Entropy: fixed at 0.1 for controlled (original default)
        entropy_coeff_start=config_dict.get("entropy_coeff_start", 0.1),
        entropy_coeff_end=config_dict.get("entropy_coeff_end", 0.1),
        entropy_coeff_horizon=config_dict.get("entropy_coeff_horizon", 0),
        use_entropy_annealing=config_dict.get("use_entropy_annealing", False),
        # LR annealing
        use_lr_annealing=config_dict.get("use_lr_annealing", False),
        lr_annealing_factor=config_dict.get("lr_annealing_factor", 1.0),
        num_epochs=config_dict.get("num_sgd_iter", 8),
        log_interval=config_dict.get("log_interval", 1),
        save_interval=config_dict.get("save_interval", 50),
        eval_interval=config_dict.get("eval_interval", 25),
        early_stop_patience=config_dict.get("early_stop_patience", 100),
        bc_schedule=bc_schedule_tuples,
        bc_model_dir=None,  # Don't auto-load BC; GAIL agent injected after construction
        verbose=verbose,
        results_dir=results_dir,
        experiment_name=config_dict["experiment_name"],
        seed=seed,
    )

    # Load GAIL agent
    gail_agent = _load_gail_agent(gail_model_dir, layout, env_layout)

    # Create trainer and inject GAIL agent as the BC partner
    trainer = PPOTrainer(ppo_config)
    trainer.bc_agent = gail_agent  # Use GAIL instead of BC

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
        json_config["bc_schedule"] = str(bc_schedule_tuples)
        json_config["gail_model_dir"] = gail_model_dir
        json_config["mode"] = mode_name
        json.dump(json_config, f, indent=2)

    # Finish WandB
    if use_wandb:
        import wandb
        wandb.finish()

    return results


def train_all_layouts(
    seeds: List[int],
    layouts: Optional[List[str]] = None,
    results_dir: str = "results/ppo_gail",
    verbose: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "overcooked-ai",
    gail_model_base_dir: Optional[str] = None,
    optimized: bool = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Train PPO_GAIL agents for all layouts with multiple seeds.

    Args:
        seeds: List of random seeds
        layouts: List of layouts to train (default: all paper layouts)
        results_dir: Directory to save results
        verbose: Whether to print progress
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        gail_model_base_dir: Base directory for GAIL models
        optimized: If True, use optimized ablation mode

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

        # Get GAIL model path for this layout
        if gail_model_base_dir:
            gail_model_dir = os.path.join(gail_model_base_dir, layout)
        else:
            gail_model_dir = DEFAULT_GAIL_MODEL_PATHS.get(layout)

        for seed in seeds:
            current_run += 1

            if verbose:
                print(f"\n{'#' * 60}")
                print(f"# Run {current_run}/{total_runs}: {layout} (seed={seed})")
                print(f"{'#' * 60}")

            try:
                results = train_ppo_gail(
                    layout=layout,
                    seed=seed,
                    gail_model_dir=gail_model_dir,
                    results_dir=results_dir,
                    verbose=verbose,
                    use_wandb=use_wandb,
                    wandb_project=wandb_project,
                    optimized=optimized,
                )
                all_results[layout][seed] = results

            except Exception as e:
                print(f"Error training {layout} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                all_results[layout][seed] = {"error": str(e)}

    return all_results


def print_summary(results: Dict[str, Dict[int, Dict[str, Any]]]):
    """Print a summary of training results."""
    print("\n" + "=" * 60)
    print("PPO_GAIL TRAINING SUMMARY")
    print("=" * 60)

    for layout, seed_results in results.items():
        print(f"\n{layout}:")
        print("-" * 40)

        for seed, result in seed_results.items():
            if "error" in result:
                print(f"  Seed {seed}: ERROR - {result['error']}")
            else:
                timesteps = result.get("total_timesteps", "N/A")
                print(f"  Seed {seed}: {timesteps:,} timesteps completed")


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agents with GAIL partners for Overcooked AI",
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

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--controlled",
        action="store_true",
        default=True,
        help="Controlled mode: exact PPO_BC Table 3 HPs, partner=GAIL (default)"
    )
    mode_group.add_argument(
        "--optimized",
        action="store_true",
        help="Optimized ablation: Bayesian HPs with controlled budget (10M, proper shaping)"
    )

    # GAIL model paths
    parser.add_argument(
        "--gail_model_dir",
        type=str,
        default=None,
        help="Path to GAIL model directory (for single layout)"
    )

    parser.add_argument(
        "--gail_model_base_dir",
        type=str,
        default=None,
        help="Base directory for GAIL models (for all layouts)"
    )

    # Output and logging
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/ppo_gail",
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

    # Overrides for testing
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use reduced settings for local testing (10k steps)"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings (1M steps)"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps"
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
            "total_timesteps": 1000000,
            "use_early_stopping": True,
            "early_stop_patience": 100,
            "save_interval": 25,
            "log_interval": 1,
        }

    if args.timesteps:
        local_overrides["total_timesteps"] = args.timesteps

    if args.num_training_iters:
        # Convert iterations to timesteps (30 envs * 400 steps = 12000 per iter)
        local_overrides["total_timesteps"] = args.num_training_iters * 12000

    if args.use_early_stopping:
        local_overrides["use_early_stopping"] = True

    if args.layout:
        # Train single layout
        gail_model_dir = args.gail_model_dir
        if gail_model_dir is None and args.gail_model_base_dir:
            gail_model_dir = os.path.join(args.gail_model_base_dir, args.layout)

        results = train_ppo_gail(
            layout=args.layout,
            seed=args.seed,
            gail_model_dir=gail_model_dir,
            results_dir=args.results_dir,
            verbose=verbose,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            optimized=args.optimized,
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
            gail_model_base_dir=args.gail_model_base_dir,
            optimized=args.optimized,
        )
        print_summary(results)

    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()
