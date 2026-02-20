#!/usr/bin/env python3
"""
Evaluate PPO-SP models from the models/ directory structure.

This script evaluates models stored in:
  models/ppo_sp/{layout_name}/seed{N}/params.pkl

Unlike the paper_reproduction models which use legacy layout names
(random0_legacy, random3_legacy), these models use the modern layout names
(forced_coordination, counter_circuit, etc.).
"""

import os
import sys
import pickle
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import jax
import jax.numpy as jnp
from jax import random

from human_aware_rl.jaxmarl.ppo import ActorCritic, PPOConfig
from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig


# Paper reported values for comparison (SP+SP)
PAPER_RESULTS = {
    "forced_coordination": {"mean": 160, "std": 20},
    "counter_circuit": {"mean": 120, "std": 15},
    "cramped_room": {"mean": 200, "std": 10},
    "asymmetric_advantages": {"mean": 200, "std": 15},
    "coordination_ring": {"mean": 150, "std": 20},
}


def load_checkpoint(seed_dir: str) -> Tuple[dict, object]:
    """Load model parameters and config from a seed directory.
    
    Args:
        seed_dir: Path to seed directory containing params.pkl and config.pkl
        
    Returns:
        Tuple of (params, config)
    """
    params_path = os.path.join(seed_dir, "params.pkl")
    config_path = os.path.join(seed_dir, "config.pkl")
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.pkl not found in {seed_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.pkl not found in {seed_dir}")
    
    # Patch ShapedArray for JAX version compatibility
    try:
        from jax._src.core import ShapedArray
        original_init = ShapedArray.__init__
        
        def patched_init(self, shape, dtype, weak_type=False, **kwargs):
            kwargs.pop('named_shape', None)
            return original_init(self, shape, dtype, weak_type, **kwargs)
        
        ShapedArray.__init__ = patched_init
        
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        
        ShapedArray.__init__ = original_init
        
    except Exception as e:
        raise RuntimeError(f"Failed to load params: {e}")
    
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    return params, config


def evaluate_sp_sp(
    params: dict,
    config: object,
    num_games: int = 50,
    verbose: bool = True,
    rng_key: Optional[jax.Array] = None,
) -> Dict[str, float]:
    """
    Evaluate SP+SP (Self-Play with Self-Play).
    
    Uses the SAME layout that the model was trained on (from config).
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)
    
    # Get layout from config
    layout_name = config.layout_name
    use_legacy_encoding = getattr(config, 'use_legacy_encoding', True)
    old_dynamics = getattr(config, 'old_dynamics', True)
    horizon = getattr(config, 'horizon', 400)
    
    if verbose:
        print(f"    Layout: {layout_name}")
        print(f"    Legacy encoding: {use_legacy_encoding}")
        print(f"    Old dynamics: {old_dynamics}")
    
    # Create evaluation environment (no reward shaping)
    eval_config = OvercookedJaxEnvConfig(
        layout_name=layout_name,
        horizon=horizon,
        old_dynamics=old_dynamics,
        reward_shaping_factor=0.0,  # Only sparse rewards
        use_phi=False,
        use_legacy_encoding=use_legacy_encoding,
    )
    env = OvercookedJaxEnv(config=eval_config)
    
    # Create network matching training architecture
    hidden_dim = getattr(config, 'hidden_dim', 64)
    num_hidden_layers = getattr(config, 'num_hidden_layers', 3)
    num_filters = getattr(config, 'num_filters', 25)
    num_conv_layers = getattr(config, 'num_conv_layers', 3)
    
    network = ActorCritic(
        action_dim=6,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_filters=num_filters,
        num_conv_layers=num_conv_layers,
    )
    
    episode_rewards = []
    episode_lengths = []
    
    for game_idx in range(num_games):
        rng_key, game_key = random.split(rng_key)
        states, obs = env.reset()
        
        episode_reward = 0.0
        step_count = 0
        done = False
        
        while not done:
            obs_0 = jnp.array(obs["agent_0"])[None]
            obs_1 = jnp.array(obs["agent_1"])[None]
            
            logits_0, _ = network.apply(params, obs_0)
            logits_1, _ = network.apply(params, obs_1)
            
            # Stochastic action selection
            rng_key, key_0, key_1 = random.split(rng_key, 3)
            action_0 = int(random.categorical(key_0, logits_0[0]))
            action_1 = int(random.categorical(key_1, logits_1[0]))
            
            actions = {"agent_0": action_0, "agent_1": action_1}
            states, obs, rewards, dones, infos = env.step(states, actions)
            
            reward = rewards["agent_0"]
            if hasattr(reward, '__getitem__'):
                episode_reward += float(reward[0])
            else:
                episode_reward += float(reward)
            
            step_count += 1
            
            done_flag = dones["__all__"]
            if hasattr(done_flag, '__getitem__'):
                done = bool(done_flag[0])
            else:
                done = bool(done_flag)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if verbose and (game_idx + 1) % 10 == 0:
            current_mean = np.mean(episode_rewards)
            current_std = np.std(episode_rewards)
            print(f"    Game {game_idx + 1}/{num_games}: Mean = {current_mean:.1f} ± {current_std:.1f}")
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "all_rewards": episode_rewards,
    }


def evaluate_layout(
    models_dir: str,
    layout_name: str,
    num_games: int = 50,
    verbose: bool = True,
) -> Dict[str, any]:
    """Evaluate all seeds for a given layout in the models directory."""
    
    layout_dir = os.path.join(models_dir, "ppo_sp", layout_name)
    
    if not os.path.exists(layout_dir):
        print(f"Layout directory not found: {layout_dir}")
        return None
    
    # Find all seed directories
    seed_dirs = [d for d in os.listdir(layout_dir) if d.startswith("seed")]
    seed_dirs.sort(key=lambda x: int(x.replace("seed", "")))
    
    if not seed_dirs:
        print(f"No seed directories found in: {layout_dir}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {layout_name}")
    print(f"Found {len(seed_dirs)} seeds: {seed_dirs}")
    print(f"{'='*70}")
    
    seed_results = {}
    all_rewards = []
    
    for seed_dir_name in seed_dirs:
        seed_path = os.path.join(layout_dir, seed_dir_name)
        seed_num = seed_dir_name.replace("seed", "")
        
        print(f"\n  Seed {seed_num}:")
        
        try:
            params, config = load_checkpoint(seed_path)
            
            results = evaluate_sp_sp(
                params=params,
                config=config,
                num_games=num_games,
                verbose=verbose,
            )
            
            seed_results[seed_num] = results
            all_rewards.extend(results["all_rewards"])
            
            print(f"    Result: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
            print(f"    Range: [{results['min_reward']:.0f}, {results['max_reward']:.0f}]")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    if not seed_results:
        return None
    
    seed_means = [r["mean_reward"] for r in seed_results.values()]
    
    return {
        "layout_name": layout_name,
        "num_seeds": len(seed_results),
        "num_games_per_seed": num_games,
        "overall_mean": np.mean(all_rewards),
        "overall_std": np.std(all_rewards),
        "seed_mean": np.mean(seed_means),
        "seed_std": np.std(seed_means),
        "seed_results": seed_results,
    }


def print_summary(results: Dict[str, dict]):
    """Print evaluation summary with comparison to paper."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY (SP+SP)")
    print("="*70)
    print(f"\n{'Layout':<25} {'Our Result':<20} {'Paper Result':<20} {'Match?':<10}")
    print("-"*75)
    
    for layout_name, result in results.items():
        if result is None:
            continue
        
        our_mean = result["overall_mean"]
        our_std = result["overall_std"]
        our_str = f"{our_mean:.1f} ± {our_std:.1f}"
        
        if layout_name in PAPER_RESULTS:
            paper_mean = PAPER_RESULTS[layout_name]["mean"]
            paper_std = PAPER_RESULTS[layout_name]["std"]
            paper_str = f"{paper_mean:.0f} ± {paper_std:.0f}"
            
            match = abs(our_mean - paper_mean) < 2 * paper_std
            match_str = "✓" if match else "✗"
        else:
            paper_str = "N/A"
            match_str = "-"
        
        print(f"{layout_name:<25} {our_str:<20} {paper_str:<20} {match_str:<10}")
    
    print("-"*75)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO-SP models from models/ directory"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to models directory",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="all",
        help="Layout to evaluate (or 'all')",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of evaluation games per seed",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-game output",
    )
    args = parser.parse_args()
    
    # Find models directory
    if args.models_dir:
        models_dir = args.models_dir
    else:
        models_dir = os.path.join(PROJECT_ROOT, "models")
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        sys.exit(1)
    
    ppo_sp_dir = os.path.join(models_dir, "ppo_sp")
    if not os.path.exists(ppo_sp_dir):
        print(f"ppo_sp directory not found in: {models_dir}")
        sys.exit(1)
    
    print("="*70)
    print("PPO-SP Model Evaluation (models/ directory)")
    print("="*70)
    print(f"\nModels directory: {models_dir}")
    print(f"Games per seed: {args.games}")
    
    # Available layouts
    available_layouts = [d for d in os.listdir(ppo_sp_dir) 
                         if os.path.isdir(os.path.join(ppo_sp_dir, d))]
    print(f"Available layouts: {available_layouts}")
    
    # Determine which layouts to evaluate
    if args.layout == "all":
        layouts = available_layouts
    else:
        layouts = [args.layout]
    
    # Evaluate each layout
    all_results = {}
    for layout in layouts:
        result = evaluate_layout(
            models_dir=models_dir,
            layout_name=layout,
            num_games=args.games,
            verbose=not args.quiet,
        )
        all_results[layout] = result
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    save_path = os.path.join(models_dir, "ppo_sp_evaluation_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()


