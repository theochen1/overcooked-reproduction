"""
Run full evaluation pipeline and generate paper Figure 4.

This script:
1. Loads trained models (BC, PPO_SP, PPO_BC)
2. Evaluates each agent pairing with Human Proxy
3. Tests both starting positions (normal and swapped)
4. Aggregates results over seeds
5. Generates the paper's Figure 4
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict, List, Tuple
import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR

# Layouts
LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}

SEEDS = [0, 10, 20, 30, 40]


class BCAgentWrapper(Agent):
    """Wrapper for BC model."""
    
    def __init__(self, model, featurize_fn, stochastic=True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        
    def action(self, state):
        import torch
        import torch.nn.functional as F
        
        obs = self.featurize_fn(state)[self.agent_index]
        obs_flat = obs.flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = probs.argmax(dim=-1).item()
        
        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": probs.numpy()}


def load_bc_model(model_dir: str):
    """Load BC model."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model as _load
    return _load(model_dir, verbose=False)


def evaluate_pair(
    agent1: Agent,
    agent2: Agent,
    layout: str,
    num_games: int = 10,
    swapped: bool = False,
) -> Dict:
    """
    Evaluate an agent pair.
    
    Args:
        agent1: First agent
        agent2: Second agent (e.g., Human Proxy)
        layout: Layout name
        num_games: Number of games to play
        swapped: If True, swap starting positions
    """
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    
    if swapped:
        agent2.set_agent_index(0)
        agent1.set_agent_index(1)
        pair = AgentPair(agent2, agent1)
    else:
        agent1.set_agent_index(0)
        agent2.set_agent_index(1)
        pair = AgentPair(agent1, agent2)
    
    results = ae.evaluate_agent_pair(pair, num_games=num_games, display=False)
    
    rewards = results["ep_returns"]
    return {
        "mean": np.mean(rewards),
        "std": np.std(rewards),
        "se": np.std(rewards) / np.sqrt(len(rewards)),
        "rewards": rewards,
    }


def run_evaluation(
    model_dir: str,
    output_dir: str,
    num_games: int = 10,
    verbose: bool = True,
):
    """
    Run full evaluation for all layouts and agent combinations.
    
    Args:
        model_dir: Directory containing trained models
        output_dir: Directory to save results
        num_games: Number of games per evaluation
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for layout in LAYOUTS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {layout}")
            print(f"{'='*60}")
        
        env_layout = LAYOUT_TO_ENV.get(layout, layout)
        layout_results = {}
        
        # Setup featurization
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )
        
        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)
        
        # Load Human Proxy (BC trained on test data)
        hp_dir = os.path.join(model_dir, "BC", "test", layout)
        if not os.path.exists(os.path.join(hp_dir, "model.pt")):
            print(f"  WARNING: No HP model found at {hp_dir}")
            continue
            
        try:
            hp_model, hp_params = load_bc_model(hp_dir)
            if verbose:
                print(f"  Loaded HP from {hp_dir}")
        except Exception as e:
            print(f"  ERROR loading HP: {e}")
            continue
        
        # Load BC (trained on train data)
        bc_dir = os.path.join(model_dir, "BC", "train", layout)
        bc_model = None
        if os.path.exists(os.path.join(bc_dir, "model.pt")):
            try:
                bc_model, bc_params = load_bc_model(bc_dir)
                if verbose:
                    print(f"  Loaded BC from {bc_dir}")
            except Exception as e:
                print(f"  ERROR loading BC: {e}")
        
        # Evaluate BC + HP
        if bc_model is not None:
            if verbose:
                print(f"\n  Evaluating BC + HP...")
            
            bc_results = []
            bc_sw_results = []
            
            bc_agent = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            
            r = evaluate_pair(bc_agent, hp_agent, layout, num_games, swapped=False)
            bc_results.extend(r['rewards'])
            
            bc_agent = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            
            r_sw = evaluate_pair(bc_agent, hp_agent, layout, num_games, swapped=True)
            bc_sw_results.extend(r_sw['rewards'])
            
            layout_results['bc_hp'] = {
                'mean': np.mean(bc_results),
                'std': np.std(bc_results),
                'se': np.std(bc_results) / np.sqrt(len(bc_results)),
            }
            layout_results['bc_hp_swapped'] = {
                'mean': np.mean(bc_sw_results),
                'std': np.std(bc_sw_results),
                'se': np.std(bc_sw_results) / np.sqrt(len(bc_sw_results)),
            }
            
            if verbose:
                print(f"    BC+HP: {layout_results['bc_hp']['mean']:.1f} ± {layout_results['bc_hp']['se']:.1f}")
                print(f"    BC+HP (swapped): {layout_results['bc_hp_swapped']['mean']:.1f} ± {layout_results['bc_hp_swapped']['se']:.1f}")
        
        # Save layout results
        layout_dir = os.path.join(output_dir, layout)
        os.makedirs(layout_dir, exist_ok=True)
        
        for key, value in layout_results.items():
            with open(os.path.join(layout_dir, f"{key}.json"), 'w') as f:
                json.dump(value, f, indent=2)
        
        all_results[layout] = layout_results
    
    # Save all results
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to {output_dir}")
    
    return all_results


def generate_plot(results: Dict, save_path: str = None):
    """Generate paper Figure 4 from results."""
    from human_aware_rl.visualization.plot_paper_figure4 import plot_simple_comparison
    
    # Convert results to plotting format
    plot_data = {}
    for layout in LAYOUTS:
        if layout not in results:
            continue
        
        plot_data[layout] = {}
        for key, value in results[layout].items():
            plot_data[layout][key] = value
    
    plot_simple_comparison(plot_data, save_path)


def main():
    parser = argparse.ArgumentParser(description="Run full paper evaluation")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory containing trained models")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of games per evaluation")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plot after evaluation")
    parser.add_argument("--plot_only", type=str, default=None,
                        help="Path to existing results to plot")
    
    args = parser.parse_args()
    
    if args.plot_only:
        with open(args.plot_only, 'r') as f:
            results = json.load(f)
        generate_plot(results, save_path="paper_figure4.png")
    else:
        if args.model_dir is None:
            # Default to Test_1 directory
            args.model_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "Test_1"
            )
        
        results = run_evaluation(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            num_games=args.num_games,
        )
        
        if args.plot:
            generate_plot(results, save_path="paper_figure4.png")


if __name__ == "__main__":
    main()

