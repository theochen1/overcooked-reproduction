#!/usr/bin/env python
"""
Evaluate PPO models from local_models/ppo_runs directory.

This script evaluates the pre-trained models on forced_coordination and counter_circuit layouts.
The models in local_models/ppo_runs use the old naming convention:
  - ppo_sp_random0 -> forced_coordination
  - ppo_sp_random3 -> counter_circuit

Usage:
    cd human_aware_rl/human_aware_rl
    python evaluate_local_models.py
    python evaluate_local_models.py --layout forced_coordination
    python evaluate_local_models.py --games 50
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from collections import defaultdict

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overcooked_ai_py.utils import load_pickle
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from baselines_utils import get_agent_from_saved_model

# Path to local models (relative to human_aware_rl/human_aware_rl/)
LOCAL_MODELS_DIR = '../local_models/ppo_runs/'

# Layout name mappings (old name -> new name for display)
LAYOUT_MAPPINGS = {
    "random0": "forced_coordination",
    "random3": "counter_circuit",
}

# Model configurations
MODEL_CONFIGS = {
    "forced_coordination": {
        "run_name": "ppo_sp_random0",
        "layout_name": "random0",  # The layout name used when training
        "seeds": [2229, 7649, 7225, 9807, 386]
    },
    "counter_circuit": {
        "run_name": "ppo_sp_random3",
        "layout_name": "random3",  # The layout name used when training
        "seeds": [2229, 7649, 7225, 9807, 386]
    }
}

# Expected results from paper (SP+SP self-play)
PAPER_RESULTS = {
    "forced_coordination": {"mean": 0.8, "std": 3.9},
    "counter_circuit": {"mean": 0.0, "std": 0.0}
}


def get_agent_from_local_model(run_name, seed, best=False):
    """Load a PPO agent from local_models directory."""
    base_dir = os.path.join(os.path.dirname(__file__), LOCAL_MODELS_DIR)
    seed_dir = os.path.join(base_dir, run_name, f'seed{seed}')
    config_path = os.path.join(seed_dir, 'config.pickle')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    config = load_pickle(config_path)
    
    # Get the model directory
    if best:
        model_dir = os.path.join(seed_dir, 'best')
    else:
        model_dir = os.path.join(seed_dir, 'ppo_agent')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model not found at {model_dir}")
    
    # Load agent
    agent = get_agent_from_saved_model(model_dir, config.get("sim_threads", 30))
    
    return agent, config


def evaluate_self_play(layout_name, run_name, seeds, num_games=10, display=False):
    """Evaluate PPO-SP agent playing with itself."""
    print(f"\n{'='*60}")
    print(f"Evaluating PPO-SP on layout: {layout_name}")
    print(f"Run name: {run_name}")
    print(f"{'='*60}")
    
    results = defaultdict(list)
    
    for seed in seeds:
        # Reset TensorFlow graph for each seed
        tf.reset_default_graph()
        
        try:
            agent, config = get_agent_from_local_model(run_name, seed, best=True)
            
            # Get layout name from config
            mdp_params = config.get("mdp_params", {})
            env_params = config.get("env_params", {"horizon": 400})
            
            # Create evaluator
            evaluator = AgentEvaluator(
                mdp_params=mdp_params,
                env_params=env_params
            )
            
            # Self-play evaluation (same agent plays both positions)
            agent_pair = AgentPair(agent, agent, allow_duplicate_agents=True)
            eval_results = evaluator.evaluate_agent_pair(
                agent_pair,
                num_games=num_games,
                display=display
            )
            
            avg_reward = np.mean(eval_results['ep_returns'])
            std_reward = np.std(eval_results['ep_returns'])
            
            results['seeds'].append(seed)
            results['avg_rewards'].append(avg_reward)
            results['std_rewards'].append(std_reward)
            results['all_rewards'].append(eval_results['ep_returns'])
            
            print(f"  Seed {seed}: {avg_reward:.2f} ± {std_reward:.2f}")
            
        except Exception as e:
            print(f"  Seed {seed}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    if results['avg_rewards']:
        overall_avg = np.mean(results['avg_rewards'])
        overall_std = np.std(results['avg_rewards'])
        print(f"\n  Overall: {overall_avg:.2f} ± {overall_std:.2f}")
    
    return results


def print_comparison_with_paper(results, layout_name):
    """Print comparison between our results and paper results."""
    if layout_name in PAPER_RESULTS:
        paper = PAPER_RESULTS[layout_name]
        our_avg = np.mean(results['avg_rewards']) if results['avg_rewards'] else 0
        our_std = np.std(results['avg_rewards']) if results['avg_rewards'] else 0
        
        print(f"\n  Comparison with paper:")
        print(f"    Paper:  {paper['mean']:.2f} ± {paper['std']:.2f}")
        print(f"    Ours:   {our_avg:.2f} ± {our_std:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate local PPO-SP models')
    parser.add_argument('--layout', type=str, default=None,
                        choices=['forced_coordination', 'counter_circuit'],
                        help='Specific layout to evaluate (default: all)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games per evaluation (default: 10)')
    parser.add_argument('--display', action='store_true',
                        help='Display game visualization')
    parser.add_argument('--best', action='store_true', default=True,
                        help='Use best checkpoint instead of final model')
    args = parser.parse_args()
    
    # Determine which layouts to evaluate
    layouts_to_eval = [args.layout] if args.layout else list(MODEL_CONFIGS.keys())
    
    all_results = {}
    
    print("="*60)
    print("PPO-SP Model Evaluation")
    print("="*60)
    print(f"\nModels directory: {LOCAL_MODELS_DIR}")
    print(f"Layouts to evaluate: {layouts_to_eval}")
    print(f"Games per seed: {args.games}")
    
    for layout in layouts_to_eval:
        if layout not in MODEL_CONFIGS:
            print(f"Unknown layout: {layout}")
            continue
        
        config = MODEL_CONFIGS[layout]
        
        # Run evaluation
        results = evaluate_self_play(
            layout_name=layout,
            run_name=config['run_name'],
            seeds=config['seeds'],
            num_games=args.games,
            display=args.display
        )
        
        all_results[layout] = results
        
        # Print comparison with paper
        print_comparison_with_paper(results, layout)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n{'Layout':<25} {'Our Result':<20} {'Paper Result':<20}")
    print("-"*65)
    
    for layout, results in all_results.items():
        if results['avg_rewards']:
            our_avg = np.mean(results['avg_rewards'])
            our_std = np.std(results['avg_rewards'])
            our_str = f"{our_avg:.2f} ± {our_std:.2f}"
        else:
            our_str = "N/A"
        
        if layout in PAPER_RESULTS:
            paper = PAPER_RESULTS[layout]
            paper_str = f"{paper['mean']:.2f} ± {paper['std']:.2f}"
        else:
            paper_str = "N/A"
        
        print(f"{layout:<25} {our_str:<20} {paper_str:<20}")
    
    print("\nNote: Paper results show SP+SP (self-play) performance is very low")
    print("on these 'hard' layouts. This is expected - self-play alone doesn't")
    print("learn good coordination on forced_coordination and counter_circuit.")


if __name__ == "__main__":
    main()



