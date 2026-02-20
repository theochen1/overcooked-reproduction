"""
Evaluate BC vs Human Proxy (matching paper Figure 4).

This script evaluates BC agents paired with Human Proxy models,
using the exact methodology from the paper:
- 400 timestep episodes
- Both agent orderings (BC as agent 0 and agent 1)
- Multiple games for computing mean and standard error

Usage:
    python -m human_aware_rl.evaluation.evaluate_bc_hp --all_layouts
    python -m human_aware_rl.evaluation.evaluate_bc_hp --layout cramped_room
"""

import argparse
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.imitation.behavior_cloning import (
    BC_SAVE_DIR,
    load_bc_model,
)
from human_aware_rl.imitation.bc_agent import BCAgent
from human_aware_rl.ppo.configs.paper_configs import PAPER_LAYOUTS, LAYOUT_TO_ENV


def load_bc_agent(
    model_dir: str,
    layout_name: str,
    agent_index: int,
    stochastic: bool = True,
) -> BCAgent:
    """Load a BC agent from a saved model."""
    model, bc_params = load_bc_model(model_dir)
    
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name, "old_dynamics": True},
        env_params={"horizon": 400}
    )
    
    featurize_fn = lambda state: ae.env.featurize_state_mdp(state)
    
    return BCAgent(
        model=model,
        bc_params=bc_params,
        featurize_fn=featurize_fn,
        agent_index=agent_index,
        stochastic=stochastic,
    )


def evaluate_bc_vs_hp(
    layout: str,
    bc_model_dir: str,
    hp_model_dir: str,
    num_games: int = 100,
    horizon: int = 400,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate BC agent paired with Human Proxy.
    
    Matches paper Figure 4 methodology:
    - Both orderings (BC as agent 0 and agent 1)
    - 400 timestep episodes
    - Returns mean and std for computing standard error
    
    Args:
        layout: Layout name
        bc_model_dir: Path to BC model
        hp_model_dir: Path to Human Proxy model
        num_games: Number of games per ordering
        horizon: Episode length (400 in paper)
        verbose: Print progress
        
    Returns:
        Dictionary with results for both orderings
    """
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Create agent evaluator
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": horizon}
    )
    
    results = {}
    
    # Ordering 1: BC as agent 0, HP as agent 1
    if verbose:
        print(f"\n  Evaluating BC (agent 0) + HP (agent 1)...")
    
    bc_agent_0 = load_bc_agent(bc_model_dir, env_layout, agent_index=0)
    hp_agent_1 = load_bc_agent(hp_model_dir, env_layout, agent_index=1)
    
    agent_pair_0 = AgentPair(bc_agent_0, hp_agent_1)
    eval_results_0 = ae.evaluate_agent_pair(
        agent_pair_0,
        num_games=num_games,
        display=False,
        info=False
    )
    
    rewards_0 = eval_results_0["ep_returns"]
    results["bc_0_hp_1"] = {
        "mean": float(np.mean(rewards_0)),
        "std": float(np.std(rewards_0)),
        "stderr": float(np.std(rewards_0) / np.sqrt(len(rewards_0))),
        "all_rewards": [float(r) for r in rewards_0],
    }
    
    if verbose:
        print(f"    BC(0)+HP(1): {results['bc_0_hp_1']['mean']:.1f} ± {results['bc_0_hp_1']['stderr']:.1f}")
    
    # Ordering 2: HP as agent 0, BC as agent 1 (swapped)
    if verbose:
        print(f"  Evaluating HP (agent 0) + BC (agent 1) [swapped]...")
    
    hp_agent_0 = load_bc_agent(hp_model_dir, env_layout, agent_index=0)
    bc_agent_1 = load_bc_agent(bc_model_dir, env_layout, agent_index=1)
    
    agent_pair_1 = AgentPair(hp_agent_0, bc_agent_1)
    eval_results_1 = ae.evaluate_agent_pair(
        agent_pair_1,
        num_games=num_games,
        display=False,
        info=False
    )
    
    rewards_1 = eval_results_1["ep_returns"]
    results["hp_0_bc_1"] = {
        "mean": float(np.mean(rewards_1)),
        "std": float(np.std(rewards_1)),
        "stderr": float(np.std(rewards_1) / np.sqrt(len(rewards_1))),
        "all_rewards": [float(r) for r in rewards_1],
    }
    
    if verbose:
        print(f"    HP(0)+BC(1): {results['hp_0_bc_1']['mean']:.1f} ± {results['hp_0_bc_1']['stderr']:.1f}")
    
    # Combined (average of both orderings)
    all_rewards = rewards_0 + rewards_1
    results["combined"] = {
        "mean": float(np.mean(all_rewards)),
        "std": float(np.std(all_rewards)),
        "stderr": float(np.std(all_rewards) / np.sqrt(len(all_rewards))),
    }
    
    if verbose:
        print(f"    Combined: {results['combined']['mean']:.1f} ± {results['combined']['stderr']:.1f}")
    
    return results


def evaluate_all_layouts(
    layouts: Optional[List[str]] = None,
    bc_base_dir: Optional[str] = None,
    hp_base_dir: Optional[str] = None,
    num_games: int = 100,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate BC vs HP for all layouts.
    
    Args:
        layouts: Layouts to evaluate (default: all paper layouts)
        bc_base_dir: Base directory for BC models
        hp_base_dir: Base directory for HP models
        num_games: Games per ordering per layout
        verbose: Print progress
        
    Returns:
        Dictionary mapping layout to results
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    
    if bc_base_dir is None:
        bc_base_dir = os.path.join(BC_SAVE_DIR, "train")
    
    if hp_base_dir is None:
        hp_base_dir = os.path.join(BC_SAVE_DIR, "test")
    
    all_results = {}
    
    for layout in layouts:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layout: {layout}")
            print(f"{'='*60}")
        
        bc_model_dir = os.path.join(bc_base_dir, layout)
        hp_model_dir = os.path.join(hp_base_dir, layout)
        
        # Check models exist
        if not os.path.exists(bc_model_dir):
            print(f"  WARNING: BC model not found at {bc_model_dir}")
            continue
        if not os.path.exists(hp_model_dir):
            print(f"  WARNING: HP model not found at {hp_model_dir}")
            continue
        
        try:
            results = evaluate_bc_vs_hp(
                layout=layout,
                bc_model_dir=bc_model_dir,
                hp_model_dir=hp_model_dir,
                num_games=num_games,
                verbose=verbose,
            )
            all_results[layout] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[layout] = {"error": str(e)}
    
    return all_results


def print_paper_table(results: Dict[str, Dict]):
    """Print results in paper table format."""
    print("\n" + "="*80)
    print("BC + HProxy Evaluation Results (Figure 4 - Gray Bars)")
    print("="*80)
    print(f"{'Layout':<25} {'BC(0)+HP(1)':<20} {'HP(0)+BC(1)':<20} {'Combined':<20}")
    print("-"*80)
    
    for layout in PAPER_LAYOUTS:
        if layout not in results:
            print(f"{layout:<25} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
            continue
        
        r = results[layout]
        if "error" in r:
            print(f"{layout:<25} {'ERROR':<20} {'ERROR':<20} {'ERROR':<20}")
            continue
        
        bc_hp = f"{r['bc_0_hp_1']['mean']:.1f} ± {r['bc_0_hp_1']['stderr']:.1f}"
        hp_bc = f"{r['hp_0_bc_1']['mean']:.1f} ± {r['hp_0_bc_1']['stderr']:.1f}"
        combined = f"{r['combined']['mean']:.1f} ± {r['combined']['stderr']:.1f}"
        
        print(f"{layout:<25} {bc_hp:<20} {hp_bc:<20} {combined:<20}")
    
    print("="*80)
    print("\nNote: Results show mean ± standard error")
    print("BC(0)+HP(1) = BC as chef 0, Human Proxy as chef 1")
    print("HP(0)+BC(1) = Swapped positions (hashed bars in paper)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BC vs Human Proxy (Paper Figure 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        choices=PAPER_LAYOUTS,
        help="Evaluate single layout"
    )
    
    parser.add_argument(
        "--all_layouts",
        action="store_true",
        help="Evaluate all paper layouts"
    )
    
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games per agent ordering"
    )
    
    parser.add_argument(
        "--bc_dir",
        type=str,
        default=None,
        help="BC models directory"
    )
    
    parser.add_argument(
        "--hp_dir",
        type=str,
        default=None,
        help="Human Proxy models directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.layout:
        layouts = [args.layout]
    elif args.all_layouts:
        layouts = PAPER_LAYOUTS
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        return
    
    results = evaluate_all_layouts(
        layouts=layouts,
        bc_base_dir=args.bc_dir,
        hp_base_dir=args.hp_dir,
        num_games=args.num_games,
        verbose=verbose,
    )
    
    print_paper_table(results)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

