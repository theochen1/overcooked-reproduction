#!/usr/bin/env python
"""
Paper Reproduction Training Script (PPO Self-Play)

Trains PPO self-play agents using per-layout hyperparameters from the paper's Table 2.
Pulls configs from paper_configs.py to avoid hardcoded values drifting out of sync.

Key settings matching the canonical HARL PPO path:
1. Uses modern 26-channel lossless_state_encoding (not legacy 20-channel)
2. Uses per-minibatch advantage normalization (not per-batch)
3. Entropy coefficient = 0.1 (constant, no annealing)
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
15. PPO-SP clip schedule is strict paper parity:
    clip_eps=0.05, cliprange_schedule=constant

Usage:
    cd overcooked_ai-master/src
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout cramped_room
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout forced_coordination
    python -m human_aware_rl.jaxmarl.train_paper_reproduction --all_layouts --seeds 0,10,20,30,40
"""

import argparse
import os
import sys
from typing import Any, Dict

# Ensure we can import from the correct location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from human_aware_rl.jaxmarl.ppo import PPOConfig, PPOTrainer
from human_aware_rl.ppo.configs.paper_configs import (
    PAPER_LAYOUTS,
    LAYOUT_TO_ENV,
    PAPER_PPO_SP_CONFIGS,
    get_ppo_sp_config,
)

STRICT_SP_PARITY_KEYS = {
    "clip_eps": 0.05,
    "max_grad_norm": 0.1,
    "gamma": 0.99,
    "gae_lambda": 0.98,
    "num_workers": 30,
    "rollout_fragment_length": 400,
    "train_batch_size": 12000,
    "num_minibatches": 6,
    "num_sgd_iter": 8,
    "entropy_coeff_start": 0.1,
    "entropy_coeff_end": 0.1,
    "cliprange_schedule": "constant",
    "use_legacy_encoding": False,
    # NOTE: strict parity here follows the TF1 ground-truth codepath,
    # which uses hidden size 64 in practice.
    "hidden_dim": 64,
}

ALLOWED_PAPER_KEYS = {
    "layout_name", "seed", "experiment_name", "results_dir",
    "horizon", "old_dynamics", "use_legacy_encoding",
    "total_timesteps", "learning_rate", "gamma", "gae_lambda",
    "clip_eps", "clip_eps_end", "clip_end_fraction", "cliprange_schedule",
    "vf_coef", "max_grad_norm", "kl_coeff",
    "num_workers", "rollout_fragment_length", "num_minibatches", "num_sgd_iter",
    "train_batch_size", "reward_shaping_factor", "reward_shaping_horizon", "use_phi",
    "num_hidden_layers", "hidden_dim", "num_filters", "num_conv_layers", "use_lstm", "cell_size",
    "entropy_coeff_start", "entropy_coeff_end", "entropy_coeff_horizon", "use_entropy_annealing",
    "use_lr_annealing", "lr_annealing_factor", "lr_schedule_mode",
    "evaluation_interval", "evaluation_num_games",
    "eval_deterministic",
    "bc_schedule", "bc_model_dir",
    "verbose", "verbose_debug", "grad_diagnostics", "log_interval", "save_interval",
    "use_early_stopping", "early_stop_patience", "early_stop_min_reward",
    "num_training_iters",
}


def _assert_sp_parity_contract(layout: str, config_dict: Dict[str, Any]) -> None:
    """Fail fast when a strict paper parity invariant is violated."""
    for k, expected in STRICT_SP_PARITY_KEYS.items():
        actual = config_dict.get(k)
        if actual != expected:
            raise ValueError(f"Strict parity violation for '{k}': expected {expected}, got {actual}")

    layout_cfg = PAPER_PPO_SP_CONFIGS[layout]
    if config_dict.get("learning_rate") != layout_cfg["learning_rate"]:
        raise ValueError(
            f"Strict parity violation for learning_rate on {layout}: "
            f"expected {layout_cfg['learning_rate']}, got {config_dict.get('learning_rate')}"
        )
    if config_dict.get("vf_coef") != layout_cfg["vf_coef"]:
        raise ValueError(
            f"Strict parity violation for vf_coef on {layout}: "
            f"expected {layout_cfg['vf_coef']}, got {config_dict.get('vf_coef')}"
        )
    if config_dict.get("total_timesteps") != layout_cfg["total_timesteps"]:
        raise ValueError(
            f"Strict parity violation for total_timesteps on {layout}: "
            f"expected {layout_cfg['total_timesteps']}, got {config_dict.get('total_timesteps')}"
        )


def ppo_config_from_paper_dict(
    paper_cfg: Dict[str, Any],
    *,
    strict_parity: bool = True,
    layout: str,
) -> PPOConfig:
    """Translate paper config dict to PPOConfig with explicit field mapping."""
    unknown_keys = sorted(set(paper_cfg.keys()) - ALLOWED_PAPER_KEYS)
    if strict_parity and unknown_keys:
        raise ValueError(f"Unknown paper config keys under strict parity: {unknown_keys}")

    if strict_parity:
        _assert_sp_parity_contract(layout, paper_cfg)

    num_envs = paper_cfg["num_workers"]
    num_steps = paper_cfg["rollout_fragment_length"]
    if strict_parity and num_envs * num_steps != paper_cfg["train_batch_size"]:
        raise ValueError(
            f"Strict parity violation for batch shape: num_workers*rollout_fragment_length="
            f"{num_envs * num_steps}, expected train_batch_size={paper_cfg['train_batch_size']}"
        )

    return PPOConfig(
        layout_name=paper_cfg["layout_name"],
        horizon=paper_cfg.get("horizon", 400),
        num_envs=num_envs,
        old_dynamics=paper_cfg.get("old_dynamics", True),
        total_timesteps=paper_cfg["total_timesteps"],
        learning_rate=paper_cfg["learning_rate"],
        num_steps=num_steps,
        num_minibatches=paper_cfg["num_minibatches"],
        num_epochs=paper_cfg["num_sgd_iter"],
        gamma=paper_cfg["gamma"],
        gae_lambda=paper_cfg["gae_lambda"],
        clip_eps=paper_cfg["clip_eps"],
        clip_eps_end=paper_cfg.get("clip_eps_end", 0.0),
        clip_end_fraction=paper_cfg.get("clip_end_fraction", 1.0),
        cliprange_schedule=paper_cfg.get("cliprange_schedule", "constant"),
        ent_coef=paper_cfg.get("entropy_coeff_start", 0.1),
        vf_coef=paper_cfg["vf_coef"],
        max_grad_norm=paper_cfg["max_grad_norm"],
        kl_coeff=paper_cfg.get("kl_coeff", 0.2),
        use_lr_annealing=paper_cfg.get("use_lr_annealing", False),
        lr_annealing_factor=paper_cfg.get("lr_annealing_factor", 1.0),
        lr_schedule_mode=paper_cfg.get("lr_schedule_mode", "tf_factor"),
        use_entropy_annealing=paper_cfg.get("use_entropy_annealing", False),
        entropy_coeff_start=paper_cfg.get("entropy_coeff_start", 0.1),
        entropy_coeff_end=paper_cfg.get("entropy_coeff_end", 0.1),
        entropy_coeff_horizon=paper_cfg.get("entropy_coeff_horizon", 0),
        clip_vf=True,
        reward_shaping_factor=paper_cfg.get("reward_shaping_factor", 1.0),
        reward_shaping_horizon=paper_cfg["reward_shaping_horizon"],
        use_phi=paper_cfg.get("use_phi", False),
        use_legacy_encoding=paper_cfg.get("use_legacy_encoding", False),
        num_hidden_layers=paper_cfg.get("num_hidden_layers", 3),
        hidden_dim=paper_cfg.get("hidden_dim", 64),
        num_filters=paper_cfg.get("num_filters", 25),
        num_conv_layers=paper_cfg.get("num_conv_layers", 3),
        use_lstm=paper_cfg.get("use_lstm", False),
        cell_size=paper_cfg.get("cell_size", 256),
        log_interval=paper_cfg.get("log_interval", 1),
        save_interval=paper_cfg.get("save_interval", 50),
        eval_interval=paper_cfg.get("evaluation_interval", 50),
        eval_num_games=paper_cfg.get("evaluation_num_games", 50),
        eval_deterministic=paper_cfg.get("eval_deterministic", False),
        verbose=paper_cfg.get("verbose", True),
        verbose_debug=paper_cfg.get("verbose_debug", False),
        grad_diagnostics=paper_cfg.get("grad_diagnostics", False),
        use_early_stopping=paper_cfg.get("use_early_stopping", False),
        early_stop_patience=paper_cfg.get("early_stop_patience", 100),
        early_stop_min_reward=paper_cfg.get("early_stop_min_reward", float("inf")),
        bc_schedule=paper_cfg.get("bc_schedule", [(0, 0.0), (float("inf"), 0.0)]),
        results_dir=paper_cfg["results_dir"],
        experiment_name=paper_cfg["experiment_name"],
        seed=paper_cfg["seed"],
        canonical_paper_entrypoint=True,
    )


def get_paper_reproduction_config(
    layout: str,
    seed: int = 0,
    total_timesteps: int = None,
    results_dir: str = "paper_reproduction_results",
    strict_parity: bool = True,
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

    return ppo_config_from_paper_dict(
        config_dict,
        strict_parity=strict_parity,
        layout=layout,
    )


def train_paper_ppo_sp(
    layout: str,
    seed: int,
    out_dir: str,
    strict_parity: bool = True,
    total_timesteps: int = None,
):
    """Canonical strict-parity PPO-SP training entrypoint."""
    config = get_paper_reproduction_config(
        layout=layout,
        seed=seed,
        total_timesteps=total_timesteps,
        results_dir=out_dir,
        strict_parity=strict_parity,
    )
    trainer = PPOTrainer(config)
    return trainer.train(), config


def train_layout(layout: str, seed: int, total_timesteps: int = None,
                 results_dir: str = "paper_reproduction_results",
                 strict_parity: bool = True):
    """Train a single layout/seed combination."""
    results, config = train_paper_ppo_sp(
        layout=layout,
        seed=seed,
        out_dir=results_dir,
        strict_parity=strict_parity,
        total_timesteps=total_timesteps,
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
    print(f"  Clip epsilon (start): {config.clip_eps}")
    print(f"  Clip schedule: {config.cliprange_schedule}")
    print(f"  Clip epsilon end: {config.clip_eps_end}")
    print(f"  Clip end fraction: {config.clip_end_fraction}")
    print(f"  Max grad norm: {config.max_grad_norm}")
    print(f"  GAE lambda: {config.gae_lambda}")
    print(f"  Gamma: {config.gamma}")
    print()
    print("Reward shaping:")
    print(f"  Reward shaping factor: {config.reward_shaping_factor}")
    print(f"  Reward shaping horizon: {config.reward_shaping_horizon:,.0f} (anneals to 0 by this step)")
    print("=" * 60)
    print()

    # Print final results
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final mean shaped return (train): {results.get('train_shaped_return_mean', results.get('final_mean_reward', 0.0)):.2f}")
    print(f"Final mean sparse return (train): {results.get('train_sparse_return_mean', 0.0):.2f}")
    print(f"Best mean shaped return (train): {results.get('best_mean_reward', 0.0):.2f}")
    print(f"Final eval sparse return: {results.get('final_eval_reward', 0.0):.2f}")
    print(f"Eval policy mode: {results.get('eval_policy', 'unknown')}")
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
    parser.add_argument(
        '--no-strict-parity',
        action='store_true',
        help='Disable strict paper-parity assertions and unknown-key checks',
    )
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
                    train_layout(layout, seed, args.timesteps, args.results_dir, not args.no_strict_parity)
                except Exception as e:
                    print(f"ERROR training {layout} seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
    elif args.layout:
        layout = layout_aliases.get(args.layout, args.layout)
        train_layout(layout, args.seed, args.timesteps, args.results_dir, not args.no_strict_parity)
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)


if __name__ == "__main__":
    main()
