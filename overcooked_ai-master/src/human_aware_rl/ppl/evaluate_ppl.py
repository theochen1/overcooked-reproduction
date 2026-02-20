"""
Evaluation Script for PPL-based Models

This script evaluates PPL models against:
1. Standard BC models
2. PPO_BC models  
3. PPO_GAIL models

Metrics:
- Cross-entropy on held-out human data
- Entropy of predictions (uncertainty)
- Rollout performance (when paired with self/BC/PPO)
"""

import argparse
import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TEST


def load_test_data(layout: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data for a layout."""
    data_params = {
        "layouts": [layout],
        "check_trajectories": False,
        "featurize_states": True,
        "data_path": CLEAN_2019_HUMAN_DATA_TEST,
    }
    
    processed_trajs = get_human_human_trajectories(**data_params, silent=True)
    
    states, actions = [], []
    for ep_idx in range(len(processed_trajs["ep_states"])):
        for t in range(len(processed_trajs["ep_states"][ep_idx])):
            states.append(processed_trajs["ep_states"][ep_idx][t].flatten())
            actions.append(int(processed_trajs["ep_actions"][ep_idx][t]))
    
    return np.array(states), np.array(actions)


def compute_metrics(
    action_probs: np.ndarray,
    true_actions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        action_probs: (N, num_actions) predicted probabilities
        true_actions: (N,) true action indices
        
    Returns:
        Dict of metrics
    """
    N = len(true_actions)
    num_actions = action_probs.shape[1]
    
    # Cross-entropy loss
    true_probs = action_probs[np.arange(N), true_actions]
    cross_entropy = -np.mean(np.log(true_probs + 1e-8))
    
    # Accuracy
    predictions = np.argmax(action_probs, axis=1)
    accuracy = np.mean(predictions == true_actions)
    
    # Entropy of predictions (uncertainty)
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1)
    mean_entropy = np.mean(entropy)
    
    # Perplexity
    perplexity = np.exp(cross_entropy)
    
    return {
        "cross_entropy": cross_entropy,
        "accuracy": accuracy,
        "mean_entropy": mean_entropy,
        "perplexity": perplexity,
    }


def evaluate_bayesian_bc(model_dir: str, test_states: np.ndarray, test_actions: np.ndarray) -> Dict[str, float]:
    """Evaluate Bayesian BC model."""
    import pyro
    from human_aware_rl.ppl.bayesian_bc import load_bayesian_bc
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, guide, config = load_bayesian_bc(model_dir, device)
    
    states_tensor = torch.tensor(test_states, dtype=torch.float32, device=device)
    
    # Get predictions with uncertainty
    num_samples = 50
    all_probs = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            guide()  # Sample from posterior
            logits = model(states_tensor)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.stack(all_probs, axis=0)  # (num_samples, N, num_actions)
    mean_probs = np.mean(all_probs, axis=0)  # (N, num_actions)
    std_probs = np.std(all_probs, axis=0)
    
    # Compute base metrics
    metrics = compute_metrics(mean_probs, test_actions)
    
    # Add uncertainty metrics
    metrics["mean_prediction_std"] = np.mean(std_probs)
    metrics["epistemic_uncertainty"] = np.mean(np.std(all_probs, axis=0).sum(axis=1))
    
    return metrics


def evaluate_rational_agent(model_dir: str, test_states: np.ndarray, test_actions: np.ndarray) -> Dict[str, float]:
    """Evaluate Rational Agent model."""
    import pickle
    import pyro
    from human_aware_rl.ppl.rational_agent import RationalAgentModel, QNetwork
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    # Load Q-network
    q_network = QNetwork(
        config["state_dim"],
        config["action_dim"],
        config["q_hidden_dims"],
    ).to(device)
    q_network.load_state_dict(
        torch.load(os.path.join(model_dir, "q_network.pt"), map_location=device)
    )
    q_network.eval()
    
    # Get beta from params (if learned)
    beta = 1.0  # Default
    params_path = os.path.join(model_dir, "params.pt")
    if os.path.exists(params_path):
        pyro.clear_param_store()
        pyro.get_param_store().load(params_path, map_location=device)
        # Try to extract beta
        for name, param in pyro.get_param_store().items():
            if "beta" in name.lower():
                beta = float(param.detach().cpu().numpy())
                break
    
    states_tensor = torch.tensor(test_states, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        q_values = q_network(states_tensor)
        action_probs = F.softmax(beta * q_values, dim=-1).cpu().numpy()
    
    metrics = compute_metrics(action_probs, test_actions)
    metrics["learned_beta"] = beta
    
    return metrics


def evaluate_hierarchical_bc(model_dir: str, test_states: np.ndarray, test_actions: np.ndarray) -> Dict[str, float]:
    """Evaluate Hierarchical BC model."""
    import pickle
    from human_aware_rl.ppl.hierarchical_bc import HierarchicalBCModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    model = HierarchicalBCModel(
        state_dim=config["state_dim"],
        num_goals=config["num_goals"],
        action_dim=config["action_dim"],
        goal_hidden_dims=config["goal_hidden_dims"],
        policy_hidden_dims=config["policy_hidden_dims"],
    ).to(device)
    
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    )
    model.eval()
    
    states_tensor = torch.tensor(test_states, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        goal_probs, action_probs = model(states_tensor)
        action_probs = action_probs.cpu().numpy()
        goal_probs = goal_probs.cpu().numpy()
    
    metrics = compute_metrics(action_probs, test_actions)
    
    # Goal usage statistics
    mean_goal_probs = np.mean(goal_probs, axis=0)
    metrics["goal_entropy"] = -np.sum(mean_goal_probs * np.log(mean_goal_probs + 1e-8))
    metrics["num_active_goals"] = np.sum(mean_goal_probs > 0.1)
    
    return metrics


def evaluate_all_models(
    layout: str,
    results_dir: str = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all PPL models on a layout.
    
    Returns:
        Dict mapping model names to their metrics
    """
    results_dir = results_dir or os.path.join(DATA_DIR, "ppl_runs")
    
    if verbose:
        print(f"\nEvaluating models on {layout}...")
    
    # Load test data
    try:
        test_states, test_actions = load_test_data(layout)
        if verbose:
            print(f"  Loaded {len(test_states)} test samples")
    except Exception as e:
        print(f"  Error loading test data: {e}")
        return {}
    
    all_metrics = {}
    
    # Evaluate Bayesian BC
    model_dir = os.path.join(results_dir, "bayesian_bc", layout)
    if os.path.exists(model_dir):
        try:
            metrics = evaluate_bayesian_bc(model_dir, test_states, test_actions)
            all_metrics["bayesian_bc"] = metrics
            if verbose:
                print(f"  Bayesian BC: acc={metrics['accuracy']:.3f}, CE={metrics['cross_entropy']:.3f}")
        except Exception as e:
            print(f"  Error evaluating Bayesian BC: {e}")
    
    # Evaluate Rational Agent
    model_dir = os.path.join(results_dir, "rational_agent", layout)
    if os.path.exists(model_dir):
        try:
            metrics = evaluate_rational_agent(model_dir, test_states, test_actions)
            all_metrics["rational_agent"] = metrics
            if verbose:
                print(f"  Rational Agent: acc={metrics['accuracy']:.3f}, CE={metrics['cross_entropy']:.3f}, β={metrics.get('learned_beta', 'N/A')}")
        except Exception as e:
            print(f"  Error evaluating Rational Agent: {e}")
    
    # Evaluate Hierarchical BC
    model_dir = os.path.join(results_dir, "hierarchical_bc", layout)
    if os.path.exists(model_dir):
        try:
            metrics = evaluate_hierarchical_bc(model_dir, test_states, test_actions)
            all_metrics["hierarchical_bc"] = metrics
            if verbose:
                print(f"  Hierarchical BC: acc={metrics['accuracy']:.3f}, CE={metrics['cross_entropy']:.3f}, active_goals={metrics.get('num_active_goals', 'N/A')}")
        except Exception as e:
            print(f"  Error evaluating Hierarchical BC: {e}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPL models")
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--all_layouts", action="store_true")
    parser.add_argument("--results_dir", type=str, default=os.path.join(DATA_DIR, "ppl_runs"))
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()
    
    if args.all_layouts:
        layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring",
                   "forced_coordination", "counter_circuit"]
    else:
        layouts = [args.layout]
    
    all_results = {}
    
    for layout in layouts:
        results = evaluate_all_models(layout, args.results_dir)
        all_results[layout] = results
    
    # Print summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    
    for layout, models in all_results.items():
        print(f"\n{layout}:")
        for model, metrics in models.items():
            print(f"  {model}:")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    Cross-entropy: {metrics['cross_entropy']:.3f}")
            print(f"    Perplexity: {metrics['perplexity']:.2f}")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
