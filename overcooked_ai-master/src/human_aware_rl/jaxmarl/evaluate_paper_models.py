#!/usr/bin/env python3
"""
Evaluate trained PPO-SP models following the paper's evaluation methodology.

This evaluates Self-Play + Self-Play (SP+SP) performance where both agents
are copies of the same trained policy, using only sparse rewards (no shaping).

Paper Reference: "On the Utility of Learning about Humans for Human-AI Coordination" (NeurIPS 2019)

Paper Reported Results (Table 1, SP+SP column):
- Cramped Room: ~200
- Asymmetric Advantages: ~200  
- Coordination Ring: ~150
- Forced Coordination: ~160
- Counter Circuit: ~120
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
from flax import linen as nn

from human_aware_rl.jaxmarl.ppo import ActorCritic, PPOConfig
from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig


# Paper reported values for comparison (SP+SP, i.e. self-play with self)
PAPER_RESULTS = {
    "forced_coordination": {"mean": 160, "std": 20},  # From Table 1
    "counter_circuit": {"mean": 120, "std": 15},       # From Table 1 (Counter Circuit)
    "cramped_room": {"mean": 200, "std": 10},
    "asymmetric_advantages": {"mean": 200, "std": 15},
    "coordination_ring": {"mean": 150, "std": 20},
}

# Layout name mapping (legacy names -> paper names)
LAYOUT_MAPPING = {
    "random0_legacy": "forced_coordination",
    "random3_legacy": "counter_circuit",
}


@dataclass
class EvalConfig:
    """Configuration for model evaluation."""
    layout_name: str
    checkpoint_path: str
    num_games: int = 50  # Paper uses many games for reliable statistics
    horizon: int = 400
    old_dynamics: bool = True
    use_legacy_encoding: bool = True
    seed: int = 42


class JaxArrayUnpickler(pickle.Unpickler):
    """Custom unpickler that handles JAX version incompatibility.
    
    Newer JAX versions removed the 'named_shape' parameter from ShapedArray.
    This unpickler intercepts the array reconstruction and removes unsupported kwargs.
    """
    
    def find_class(self, module, name):
        # Import the class normally
        cls = super().find_class(module, name)
        
        # Wrap ShapedArray to ignore named_shape
        if module == 'jax._src.core' and name == 'ShapedArray':
            original_init = cls.__init__
            
            def patched_init(self, shape, dtype, weak_type=False, **kwargs):
                # Remove named_shape if present (not supported in newer JAX)
                kwargs.pop('named_shape', None)
                return original_init(self, shape, dtype, weak_type, **kwargs)
            
            cls.__init__ = patched_init
        
        return cls


def load_checkpoint(checkpoint_path: str) -> Tuple[dict, dict]:
    """Load model parameters and config from checkpoint.
    
    Handles JAX version incompatibility with custom unpickler.
    """
    params_path = os.path.join(checkpoint_path, "params.pkl")
    config_path = os.path.join(checkpoint_path, "config.pkl")
    
    # First try to patch ShapedArray to handle version incompatibility
    try:
        from jax._src.core import ShapedArray
        original_init = ShapedArray.__init__
        
        def patched_init(self, shape, dtype, weak_type=False, **kwargs):
            kwargs.pop('named_shape', None)
            return original_init(self, shape, dtype, weak_type, **kwargs)
        
        ShapedArray.__init__ = patched_init
        
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        
        # Restore original
        ShapedArray.__init__ = original_init
        
    except Exception as e:
        raise RuntimeError(f"Failed to load params: {e}")
    
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    return params, config


def evaluate_sp_sp(
    params: dict,
    config: dict,
    layout_name: str,
    num_games: int = 50,
    horizon: int = 400,
    verbose: bool = True,
    rng_key: Optional[jax.Array] = None,
) -> Dict[str, float]:
    """
    Evaluate SP+SP (Self-Play with Self-Play) - same agent plays both positions.
    
    This is the primary evaluation metric from the paper (Table 1, SP+SP column).
    Uses only sparse rewards (soup deliveries), no reward shaping.
    
    Args:
        params: Trained network parameters
        config: Training config dict
        layout_name: Layout to evaluate on
        num_games: Number of evaluation games
        horizon: Episode length
        verbose: Whether to print progress
        rng_key: Optional JAX random key
        
    Returns:
        Dictionary with evaluation statistics
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)
    
    # Create evaluation environment (no reward shaping - only sparse rewards count)
    eval_config = OvercookedJaxEnvConfig(
        layout_name=layout_name,
        horizon=horizon,
        old_dynamics=True,
        reward_shaping_factor=0.0,  # CRITICAL: No shaping during evaluation
        use_phi=False,
        use_legacy_encoding=True,
    )
    env = OvercookedJaxEnv(config=eval_config)
    
    # Create network (matching training architecture from config)
    # The ActorCritic uses action_dim, not num_actions
    network = ActorCritic(
        action_dim=6,  # 6 actions: N, S, E, W, Stay, Interact
        hidden_dim=64,
        num_hidden_layers=3,
        num_filters=25,
        num_conv_layers=3,
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
            # Get observations for both agents
            obs_0 = jnp.array(obs["agent_0"])[None]  # Add batch dim
            obs_1 = jnp.array(obs["agent_1"])[None]
            
            # Get action logits from network
            logits_0, _ = network.apply(params, obs_0)
            logits_1, _ = network.apply(params, obs_1)
            
            # Stochastic action selection (matching training behavior)
            rng_key, key_0, key_1 = random.split(rng_key, 3)
            action_0 = int(random.categorical(key_0, logits_0[0]))
            action_1 = int(random.categorical(key_1, logits_1[0]))
            
            # Step environment
            actions = {"agent_0": action_0, "agent_1": action_1}
            states, obs, rewards, dones, infos = env.step(states, actions)
            
            # Accumulate sparse reward only
            reward = rewards["agent_0"]
            if hasattr(reward, '__getitem__'):
                episode_reward += float(reward[0])
            else:
                episode_reward += float(reward)
            
            step_count += 1
            
            # Check done
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
            print(f"  Game {game_idx + 1}/{num_games}: Running mean = {current_mean:.1f} ± {current_std:.1f}")
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "all_rewards": episode_rewards,
    }


def find_best_checkpoint(model_dir: str) -> str:
    """Find the checkpoint with highest update number (final model)."""
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint_")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")
    
    # Sort by checkpoint number and get the last one
    checkpoints.sort(key=lambda x: int(x.split("_")[1]))
    return os.path.join(model_dir, checkpoints[-1])


def evaluate_layout(
    results_dir: str,
    layout_name: str,
    num_games: int = 50,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Evaluate all seeds for a given layout.
    
    Args:
        results_dir: Path to paper_reproduction results directory
        layout_name: Layout name (e.g., "random0_legacy")
        num_games: Number of games per seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with aggregated results
    """
    # Find all model directories for this layout
    model_dirs = [
        d for d in os.listdir(results_dir)
        if d.startswith(f"ppo_sp_{layout_name}_seed")
    ]
    
    if not model_dirs:
        print(f"No models found for layout: {layout_name}")
        return None
    
    model_dirs.sort()  # Sort by seed
    
    paper_name = LAYOUT_MAPPING.get(layout_name, layout_name)
    print(f"\n{'='*70}")
    print(f"Evaluating: {layout_name}")
    print(f"Paper name: {paper_name}")
    print(f"Found {len(model_dirs)} trained models")
    print(f"{'='*70}")
    
    seed_results = {}
    all_rewards = []
    
    for model_dir in model_dirs:
        seed = model_dir.split("_seed")[1]
        model_path = os.path.join(results_dir, model_dir)
        
        try:
            checkpoint_path = find_best_checkpoint(model_path)
            params, config = load_checkpoint(checkpoint_path)
            
            print(f"\nSeed {seed}:")
            print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
            
            results = evaluate_sp_sp(
                params=params,
                config=config,
                layout_name=layout_name,
                num_games=num_games,
                verbose=verbose,
            )
            
            seed_results[seed] = results
            all_rewards.extend(results["all_rewards"])
            
            print(f"  Result: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
            print(f"  Range: [{results['min_reward']:.0f}, {results['max_reward']:.0f}]")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    if not seed_results:
        return None
    
    # Aggregate across seeds
    seed_means = [r["mean_reward"] for r in seed_results.values()]
    
    aggregated = {
        "layout_name": layout_name,
        "paper_name": paper_name,
        "num_seeds": len(seed_results),
        "num_games_per_seed": num_games,
        "overall_mean": np.mean(all_rewards),
        "overall_std": np.std(all_rewards),
        "seed_mean": np.mean(seed_means),
        "seed_std": np.std(seed_means),
        "seed_results": seed_results,
    }
    
    return aggregated


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
        
        paper_name = result["paper_name"]
        our_mean = result["overall_mean"]
        our_std = result["overall_std"]
        
        our_str = f"{our_mean:.1f} ± {our_std:.1f}"
        
        if paper_name in PAPER_RESULTS:
            paper_mean = PAPER_RESULTS[paper_name]["mean"]
            paper_std = PAPER_RESULTS[paper_name]["std"]
            paper_str = f"{paper_mean:.0f} ± {paper_std:.0f}"
            
            # Check if within reasonable range (within 2 std of paper)
            match = abs(our_mean - paper_mean) < 2 * paper_std
            match_str = "✓" if match else "✗"
        else:
            paper_str = "N/A"
            match_str = "-"
        
        print(f"{paper_name:<25} {our_str:<20} {paper_str:<20} {match_str:<10}")
    
    print("-"*75)
    print("\nNotes:")
    print("  - Results are for SP+SP (Self-Play + Self-Play)")
    print("  - Only sparse rewards (soup deliveries) are counted")
    print("  - Paper results from Table 1, NeurIPS 2019")
    print("  - '✓' indicates our result is within 2 std of paper result")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO-SP models (paper reproduction)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to paper_reproduction results directory",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["random0_legacy", "random3_legacy", "all"],
        default="all",
        help="Layout to evaluate",
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
        help="Suppress per-game progress output",
    )
    args = parser.parse_args()
    
    # Find results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Default path
        results_dir = os.path.join(
            PROJECT_ROOT, "results", "paper_reproduction"
        )
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("\nPlease run training first or specify --results-dir")
        sys.exit(1)
    
    print("="*70)
    print("PPO-SP Model Evaluation (Paper Reproduction)")
    print("="*70)
    print(f"\nResults directory: {results_dir}")
    print(f"Games per seed: {args.games}")
    
    # Determine layouts to evaluate
    if args.layout == "all":
        layouts = ["random0_legacy", "random3_legacy"]
    else:
        layouts = [args.layout]
    
    # Evaluate each layout
    all_results = {}
    for layout in layouts:
        result = evaluate_layout(
            results_dir=results_dir,
            layout_name=layout,
            num_games=args.games,
            verbose=not args.quiet,
        )
        all_results[layout] = result
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    save_path = os.path.join(results_dir, "evaluation_results_detailed.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nDetailed results saved to: {save_path}")


if __name__ == "__main__":
    main()

