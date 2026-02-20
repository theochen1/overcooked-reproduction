#!/usr/bin/env python
"""
Evaluate trained PPO Self-Play models.
Run this inside the Docker container after training.

Usage:
    python evaluate_ppo_sp.py                    # Evaluate all layouts
    python evaluate_ppo_sp.py --layout simple    # Evaluate specific layout
    python evaluate_ppo_sp.py --games 50         # Run 50 games per evaluation
"""

import argparse
import numpy as np
from collections import defaultdict

from ppo.ppo import get_ppo_agent, load_training_data, PPO_DATA_DIR
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_pickle

# PPO-SP experiment configurations
PPO_SP_CONFIGS = {
    "simple": {
        "run_name": "ppo_sp_simple",
        "seeds": [2229, 7649, 7225, 9807, 386]
    },
    "unident_s": {
        "run_name": "ppo_sp_unident_s",
        "seeds": [2229, 7649, 7225, 9807, 386]
    },
    "random0": {
        "run_name": "ppo_sp_random0",
        "seeds": [2229, 7649, 7225, 9807, 386]
    },
    "random1": {
        "run_name": "ppo_sp_random1",
        "seeds": [2229, 7649, 7225, 9807, 386]
    },
    "random3": {
        "run_name": "ppo_sp_random3",
        "seeds": [2229, 7649, 7225, 9807, 386]
    }
}


def evaluate_ppo_sp_self_play(layout_name, run_name, seeds, num_games=10, display=False):
    """Evaluate PPO-SP agent playing with itself."""
    print(f"\n{'='*60}")
    print(f"Evaluating PPO-SP on layout: {layout_name}")
    print(f"{'='*60}")
    
    results = defaultdict(list)
    
    for seed in seeds:
        try:
            agent, config = get_ppo_agent(run_name, seed=seed, best=False)
            
            # Create evaluator from config
            evaluator = AgentEvaluator(
                mdp_params=config["mdp_params"], 
                env_params=config["env_params"]
            )
            
            # Self-play evaluation
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
    
    if results['avg_rewards']:
        overall_avg = np.mean(results['avg_rewards'])
        overall_std = np.std(results['avg_rewards'])
        print(f"\n  Overall: {overall_avg:.2f} ± {overall_std:.2f}")
    
    return results


def print_training_summary(run_name, seeds):
    """Print training summary from saved training info."""
    try:
        train_infos, config = load_training_data(run_name, seeds)
        print(f"\nTraining Summary for {run_name}:")
        print(f"  Total timesteps: {config.get('PPO_RUN_TOT_TIMESTEPS', 'N/A')}")
        print(f"  Learning rate: {config.get('LR', 'N/A')}")
        
        for i, (seed, info) in enumerate(zip(seeds, train_infos)):
            if 'ep_sparse_rew_mean' in info:
                final_reward = info['ep_sparse_rew_mean'][-1]
                print(f"  Seed {seed} final sparse reward: {final_reward:.2f}")
    except Exception as e:
        print(f"  Could not load training info: {e}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO-SP models')
    parser.add_argument('--layout', type=str, default=None,
                        help='Specific layout to evaluate (default: all)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games per evaluation (default: 10)')
    parser.add_argument('--display', action='store_true',
                        help='Display game visualization')
    parser.add_argument('--save', action='store_true',
                        help='Save results to pickle file')
    args = parser.parse_args()
    
    layouts_to_eval = [args.layout] if args.layout else list(PPO_SP_CONFIGS.keys())
    
    all_results = {}
    
    for layout in layouts_to_eval:
        if layout not in PPO_SP_CONFIGS:
            print(f"Unknown layout: {layout}")
            continue
            
        config = PPO_SP_CONFIGS[layout]
        
        # Print training summary
        print_training_summary(config['run_name'], config['seeds'])
        
        # Run evaluation
        results = evaluate_ppo_sp_self_play(
            layout_name=layout,
            run_name=config['run_name'],
            seeds=config['seeds'],
            num_games=args.games,
            display=args.display
        )
        
        all_results[layout] = results
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for layout, results in all_results.items():
        if results['avg_rewards']:
            avg = np.mean(results['avg_rewards'])
            std = np.std(results['avg_rewards'])
            print(f"{layout}: {avg:.2f} ± {std:.2f}")
    
    if args.save:
        save_path = PPO_DATA_DIR + "ppo_sp_evaluation_results"
        save_pickle(all_results, save_path)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()






