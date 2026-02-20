"""
Evaluate Run_2 trained models against Human Proxy.

Available models:
- BC (train) - Behavior Cloning trained on training data
- BC (test) / HP - Human Proxy trained on test data  
- GAIL - Generative Adversarial Imitation Learning

Evaluations:
- BC + HP: BC model paired with Human Proxy
- GAIL + HP: GAIL model paired with Human Proxy
"""

import os
import json
import argparse
from typing import Dict
import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

import torch
import torch.nn.functional as F

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

# Run_2 directory
RUN2_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Run_2", "models")


class BCAgentWrapper(Agent):
    """Wrapper for BC model."""
    
    def __init__(self, model, featurize_fn, stochastic=True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        
    def action(self, state):
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


class GAILAgentWrapper(Agent):
    """Wrapper for GAIL model."""
    
    def __init__(self, policy, featurize_fn, stochastic=True):
        super().__init__()
        self.policy = policy
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        
    def action(self, state):
        obs = self.featurize_fn(state)[self.agent_index]
        obs_flat = obs.flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        
        with torch.no_grad():
            # GAIL policy returns (logits, value)
            logits, _ = self.policy(obs_tensor)
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


def load_gail_model(model_dir: str, state_dim: int, action_dim: int = 6):
    """Load GAIL model."""
    from human_aware_rl.imitation.gail import GAILPolicy
    
    checkpoint_path = os.path.join(model_dir, "model.pt")
    
    policy = GAILPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    return policy


def evaluate_pair(
    agent1: Agent,
    agent2: Agent,
    layout: str,
    num_games: int = 10,
    swapped: bool = False,
) -> Dict:
    """Evaluate an agent pair."""
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


def evaluate_all(num_games: int = 10, verbose: bool = True):
    """Evaluate all available models in Run_2."""
    results = {}
    
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
        
        # Get state dimension
        dummy_state = ae.env.mdp.get_standard_start_state()
        obs_shape = featurize_fn(dummy_state)[0].shape
        state_dim = int(np.prod(obs_shape))
        
        # Load Human Proxy (BC/test)
        hp_dir = os.path.join(RUN2_DIR, "bc_runs", "test", layout)
        try:
            hp_model, _ = load_bc_model(hp_dir)
            if verbose:
                print(f"  ✓ Loaded HP from {hp_dir}")
        except Exception as e:
            print(f"  ✗ ERROR loading HP: {e}")
            continue
        
        # Load BC (train)
        bc_dir = os.path.join(RUN2_DIR, "bc_runs", "train", layout)
        bc_model = None
        try:
            bc_model, _ = load_bc_model(bc_dir)
            if verbose:
                print(f"  ✓ Loaded BC from {bc_dir}")
        except Exception as e:
            print(f"  ✗ ERROR loading BC: {e}")
        
        # Load GAIL
        gail_dir = os.path.join(RUN2_DIR, "gail_runs", layout)
        gail_policy = None
        try:
            gail_policy = load_gail_model(gail_dir, state_dim)
            if verbose:
                print(f"  ✓ Loaded GAIL from {gail_dir}")
        except Exception as e:
            print(f"  ✗ ERROR loading GAIL: {e}")
        
        # Evaluate BC + HP
        if bc_model is not None:
            if verbose:
                print(f"\n  Evaluating BC + HP...")
            
            bc_agent = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            
            r = evaluate_pair(bc_agent, hp_agent, layout, num_games, swapped=False)
            layout_results['bc_hp'] = {
                'mean': float(r['mean']),
                'std': float(r['std']),
                'se': float(r['se']),
            }
            
            # Swapped
            bc_agent = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            
            r_sw = evaluate_pair(bc_agent, hp_agent, layout, num_games, swapped=True)
            layout_results['bc_hp_swapped'] = {
                'mean': float(r_sw['mean']),
                'std': float(r_sw['std']),
                'se': float(r_sw['se']),
            }
            
            if verbose:
                print(f"    BC+HP: {layout_results['bc_hp']['mean']:.1f} ± {layout_results['bc_hp']['se']:.1f}")
                print(f"    BC+HP (swapped): {layout_results['bc_hp_swapped']['mean']:.1f} ± {layout_results['bc_hp_swapped']['se']:.1f}")
        
        # Evaluate GAIL + HP
        if gail_policy is not None:
            if verbose:
                print(f"\n  Evaluating GAIL + HP...")
            
            gail_agent = GAILAgentWrapper(gail_policy, featurize_fn, stochastic=True)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            
            r = evaluate_pair(gail_agent, hp_agent, layout, num_games, swapped=False)
            layout_results['gail_hp'] = {
                'mean': float(r['mean']),
                'std': float(r['std']),
                'se': float(r['se']),
            }
            
            # Swapped
            gail_agent = GAILAgentWrapper(gail_policy, featurize_fn, stochastic=True)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            
            r_sw = evaluate_pair(gail_agent, hp_agent, layout, num_games, swapped=True)
            layout_results['gail_hp_swapped'] = {
                'mean': float(r_sw['mean']),
                'std': float(r_sw['std']),
                'se': float(r_sw['se']),
            }
            
            if verbose:
                print(f"    GAIL+HP: {layout_results['gail_hp']['mean']:.1f} ± {layout_results['gail_hp']['se']:.1f}")
                print(f"    GAIL+HP (swapped): {layout_results['gail_hp_swapped']['mean']:.1f} ± {layout_results['gail_hp_swapped']['se']:.1f}")
        
        results[layout] = layout_results
    
    return results


def print_summary(results: Dict):
    """Print summary table."""
    print("\n" + "="*80)
    print("RUN_2 EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Layout':<25} {'BC+HP':<20} {'GAIL+HP':<20}")
    print("-"*80)
    
    for layout in LAYOUTS:
        if layout not in results:
            continue
        
        lr = results[layout]
        
        bc = f"{lr['bc_hp']['mean']:.1f}±{lr['bc_hp']['se']:.1f}" if 'bc_hp' in lr else "N/A"
        gail = f"{lr['gail_hp']['mean']:.1f}±{lr['gail_hp']['se']:.1f}" if 'gail_hp' in lr else "N/A"
        
        print(f"{layout:<25} {bc:<20} {gail:<20}")
    
    print("="*80)
    
    # Compare BC vs GAIL
    print("\n" + "="*80)
    print("BC vs GAIL COMPARISON")
    print("="*80)
    
    bc_wins = 0
    gail_wins = 0
    
    for layout in LAYOUTS:
        if layout not in results:
            continue
        lr = results[layout]
        
        if 'bc_hp' in lr and 'gail_hp' in lr:
            bc_mean = lr['bc_hp']['mean']
            gail_mean = lr['gail_hp']['mean']
            
            if bc_mean > gail_mean:
                winner = "BC"
                bc_wins += 1
            elif gail_mean > bc_mean:
                winner = "GAIL"
                gail_wins += 1
            else:
                winner = "TIE"
            
            print(f"{layout:<25} BC: {bc_mean:.1f}  GAIL: {gail_mean:.1f}  Winner: {winner}")
    
    print(f"\nOverall: BC wins {bc_wins}, GAIL wins {gail_wins}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Run_2 models")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of games per evaluation")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Run_2 Model Evaluation")
    print("="*60)
    print(f"Models directory: {RUN2_DIR}")
    print(f"Number of games: {args.num_games}")
    
    results = evaluate_all(num_games=args.num_games)
    print_summary(results)
    
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()

