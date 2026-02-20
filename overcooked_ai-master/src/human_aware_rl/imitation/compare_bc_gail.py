"""
Compare BC vs GAIL as human proxies.

Tests each model paired with Human Proxy (HP) to see which better
coordinates with human-like behavior.

This is a quick evaluation before doing full PPO training.
"""

import os
import argparse
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR, load_bc_model
from human_aware_rl.imitation.gail import GAIL_SAVE_DIR, GAILPolicy

# Layout name mapping (paper name -> environment name)
LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",  # Different env name!
}


class BCAgentWrapper(Agent):
    """Wrapper for BC model as an Agent."""
    
    def __init__(self, model, featurize_fn, stochastic=True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        self.device = next(model.parameters()).device
    
    def action(self, state):
        obs = self.featurize_fn(state)[self.agent_index]
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = torch.argmax(probs, dim=-1).item()
        
        return Action.INDEX_TO_ACTION[action_idx], {}
    
    def reset(self):
        pass


class GAILAgentWrapper(Agent):
    """Wrapper for GAIL policy as an Agent."""
    
    def __init__(self, policy, featurize_fn, stochastic=True):
        super().__init__()
        self.policy = policy
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        self.device = next(policy.parameters()).device
    
    def action(self, state):
        obs = self.featurize_fn(state)[self.agent_index]
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.policy(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = torch.argmax(probs, dim=-1).item()
        
        return Action.INDEX_TO_ACTION[action_idx], {}
    
    def reset(self):
        pass


def load_gail_model(gail_dir: str, state_dim: int, action_dim: int, device: str = "cpu"):
    """Load GAIL policy from checkpoint."""
    checkpoint_path = os.path.join(gail_dir, "model.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No GAIL model found at {checkpoint_path}")
    
    # Create policy with same architecture
    policy = GAILPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    return policy


def evaluate_agent_with_hp(
    agent: Agent,
    hp_agent: Agent,
    layout: str,
    num_games: int = 10,
    horizon: int = 400,
) -> Dict[str, float]:
    """
    Evaluate an agent paired with Human Proxy.
    
    Tests both agent orderings (agent as player 0 and player 1).
    """
    # Map layout name to environment name
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": horizon},
    )
    
    results = {"rewards": [], "rewards_p0": [], "rewards_p1": []}
    
    # Test agent as player 0, HP as player 1
    agent.set_agent_index(0)
    hp_agent.set_agent_index(1)
    agent_pair = AgentPair(agent, hp_agent)
    
    eval_results = ae.evaluate_agent_pair(agent_pair, num_games=num_games, display=False)
    results["rewards_p0"] = eval_results["ep_returns"]
    results["rewards"].extend(eval_results["ep_returns"])
    
    # Test HP as player 0, agent as player 1
    hp_agent.set_agent_index(0)
    agent.set_agent_index(1)
    agent_pair = AgentPair(hp_agent, agent)
    
    eval_results = ae.evaluate_agent_pair(agent_pair, num_games=num_games, display=False)
    results["rewards_p1"] = eval_results["ep_returns"]
    results["rewards"].extend(eval_results["ep_returns"])
    
    return {
        "mean": np.mean(results["rewards"]),
        "std": np.std(results["rewards"]),
        "mean_p0": np.mean(results["rewards_p0"]),
        "mean_p1": np.mean(results["rewards_p1"]),
        "all_rewards": results["rewards"],
    }


def compare_bc_gail(layout: str, num_games: int = 10, verbose: bool = True):
    """Compare BC and GAIL models on a layout."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    bc_train_dir = os.path.join(BC_SAVE_DIR, "train", layout)
    bc_test_dir = os.path.join(BC_SAVE_DIR, "test", layout)  # Human Proxy
    gail_dir = os.path.join(GAIL_SAVE_DIR, layout)
    
    # Check files exist
    if not os.path.exists(os.path.join(bc_train_dir, "model.pt")):
        print(f"ERROR: No BC (train) model for {layout}")
        return None
    if not os.path.exists(os.path.join(bc_test_dir, "model.pt")):
        print(f"ERROR: No BC (test/HP) model for {layout}")
        print("Run: python -m human_aware_rl.imitation.train_bc_models --all_layouts")
        return None
    
    has_gail = os.path.exists(os.path.join(gail_dir, "model.pt"))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing BC vs GAIL: {layout}")
        print(f"{'='*60}")
    
    # Map layout name to environment name
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Setup environment for featurization
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
    
    # Load Human Proxy (BC trained on test data)
    hp_model, _ = load_bc_model(bc_test_dir, device=device)
    hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
    
    # Load BC (trained on train data)
    bc_model, _ = load_bc_model(bc_train_dir, device=device)
    bc_agent = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
    
    results = {}
    
    # Evaluate BC + HP
    if verbose:
        print(f"\nEvaluating BC + HP ({num_games * 2} games)...")
    bc_results = evaluate_agent_with_hp(bc_agent, hp_agent, layout, num_games)
    results["bc"] = bc_results
    if verbose:
        print(f"  BC + HP: {bc_results['mean']:.1f} ± {bc_results['std']:.1f}")
        print(f"    As P0: {bc_results['mean_p0']:.1f}, As P1: {bc_results['mean_p1']:.1f}")
    
    # Evaluate GAIL + HP (if available)
    if has_gail:
        gail_policy = load_gail_model(gail_dir, state_dim, action_dim, device)
        gail_agent = GAILAgentWrapper(gail_policy, featurize_fn, stochastic=True)
        
        if verbose:
            print(f"\nEvaluating GAIL + HP ({num_games * 2} games)...")
        gail_results = evaluate_agent_with_hp(gail_agent, hp_agent, layout, num_games)
        results["gail"] = gail_results
        if verbose:
            print(f"  GAIL + HP: {gail_results['mean']:.1f} ± {gail_results['std']:.1f}")
            print(f"    As P0: {gail_results['mean_p0']:.1f}, As P1: {gail_results['mean_p1']:.1f}")
    else:
        if verbose:
            print(f"\nNo GAIL model found for {layout}")
        results["gail"] = None
    
    # Summary
    if verbose:
        print(f"\n{'='*40}")
        print(f"SUMMARY: {layout}")
        print(f"{'='*40}")
        print(f"  BC + HP:   {results['bc']['mean']:.1f} ± {results['bc']['std']:.1f}")
        if results["gail"]:
            print(f"  GAIL + HP: {results['gail']['mean']:.1f} ± {results['gail']['std']:.1f}")
            diff = results["gail"]["mean"] - results["bc"]["mean"]
            print(f"  Difference: {diff:+.1f} ({'GAIL better' if diff > 0 else 'BC better'})")
    
    return results


def compare_all_layouts(num_games: int = 10):
    """Compare BC vs GAIL on all layouts."""
    layouts = [
        "cramped_room",
        "asymmetric_advantages", 
        "coordination_ring",
        "forced_coordination",
        "counter_circuit",
    ]
    
    print("\n" + "="*70)
    print("BC vs GAIL COMPARISON (Paired with Human Proxy)")
    print("="*70)
    
    all_results = {}
    for layout in layouts:
        results = compare_bc_gail(layout, num_games, verbose=True)
        if results:
            all_results[layout] = results
    
    # Final summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Layout':<25} {'BC + HP':>12} {'GAIL + HP':>12} {'Winner':>10}")
    print("-"*70)
    
    for layout, results in all_results.items():
        bc_score = results["bc"]["mean"]
        if results["gail"]:
            gail_score = results["gail"]["mean"]
            winner = "GAIL" if gail_score > bc_score else "BC"
            print(f"{layout:<25} {bc_score:>12.1f} {gail_score:>12.1f} {winner:>10}")
        else:
            print(f"{layout:<25} {bc_score:>12.1f} {'N/A':>12} {'BC':>10}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare BC vs GAIL with Human Proxy")
    parser.add_argument("--layout", type=str, default=None, help="Specific layout to test")
    parser.add_argument("--num_games", type=int, default=10, help="Games per agent ordering")
    parser.add_argument("--all", action="store_true", help="Test all layouts")
    args = parser.parse_args()
    
    if args.all or args.layout is None:
        compare_all_layouts(args.num_games)
    else:
        compare_bc_gail(args.layout, args.num_games)

