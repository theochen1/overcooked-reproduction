"""
Compare PPL Models with Standard BC/PPO Baselines

This script provides a comprehensive comparison between:
- PPL Models: Bayesian BC, Rational Agent, Hierarchical BC
- Standard Models: BC, PPO_BC, PPO_GAIL

Comparison dimensions:
1. Prediction accuracy on held-out human data
2. Uncertainty calibration
3. Interpretability (goal inference, rationality parameter)
4. Rollout performance
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
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR


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


def evaluate_standard_bc(model_dir: str, test_states: np.ndarray, test_actions: np.ndarray) -> Dict[str, float]:
    """Evaluate standard PyTorch BC model."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, bc_params = load_bc_model(model_dir, device=device)
        model.eval()
        
        states_tensor = torch.tensor(test_states, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            logits = model(states_tensor)
            action_probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        # Compute metrics
        N = len(test_actions)
        true_probs = action_probs[np.arange(N), test_actions]
        
        cross_entropy = -np.mean(np.log(true_probs + 1e-8))
        accuracy = np.mean(np.argmax(action_probs, axis=1) == test_actions)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1).mean()
        perplexity = np.exp(cross_entropy)
        
        return {
            "cross_entropy": cross_entropy,
            "accuracy": accuracy,
            "mean_entropy": entropy,
            "perplexity": perplexity,
            "model_type": "standard_bc",
        }
    except Exception as e:
        return {"error": str(e), "model_type": "standard_bc"}


def evaluate_ppl_model(
    model_type: str,
    model_dir: str,
    test_states: np.ndarray,
    test_actions: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a PPL model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if model_type == "bayesian_bc":
            import pyro
            from human_aware_rl.ppl.bayesian_bc import load_bayesian_bc
            
            model, guide, config = load_bayesian_bc(model_dir, device)
            states_tensor = torch.tensor(test_states, dtype=torch.float32, device=device)
            
            # Sample from posterior
            num_samples = 50
            all_probs = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    guide()
                    logits = model(states_tensor)
                    probs = F.softmax(logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
            
            all_probs = np.stack(all_probs, axis=0)
            mean_probs = np.mean(all_probs, axis=0)
            std_probs = np.std(all_probs, axis=0)
            
        elif model_type == "rational_agent":
            import pickle
            from human_aware_rl.ppl.rational_agent import QNetwork
            
            with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
                config = pickle.load(f)
            
            q_network = QNetwork(
                config["state_dim"],
                config["action_dim"],
                config["q_hidden_dims"],
            ).to(device)
            q_network.load_state_dict(
                torch.load(os.path.join(model_dir, "q_network.pt"), map_location=device)
            )
            q_network.eval()
            
            states_tensor = torch.tensor(test_states, dtype=torch.float32, device=device)
            
            # Use default beta=1.0 for comparison
            beta = 1.0
            
            with torch.no_grad():
                q_values = q_network(states_tensor)
                mean_probs = F.softmax(beta * q_values, dim=-1).cpu().numpy()
                std_probs = np.zeros_like(mean_probs)  # No uncertainty for point estimate
            
        elif model_type == "hierarchical_bc":
            import pickle
            from human_aware_rl.ppl.hierarchical_bc import HierarchicalBCModel
            
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
                mean_probs = action_probs.cpu().numpy()
                std_probs = np.zeros_like(mean_probs)
        
        else:
            return {"error": f"Unknown model type: {model_type}"}
        
        # Compute metrics
        N = len(test_actions)
        true_probs = mean_probs[np.arange(N), test_actions]
        
        cross_entropy = -np.mean(np.log(true_probs + 1e-8))
        accuracy = np.mean(np.argmax(mean_probs, axis=1) == test_actions)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1).mean()
        perplexity = np.exp(cross_entropy)
        
        return {
            "cross_entropy": cross_entropy,
            "accuracy": accuracy,
            "mean_entropy": entropy,
            "perplexity": perplexity,
            "mean_uncertainty": np.mean(std_probs),
            "model_type": model_type,
        }
        
    except Exception as e:
        return {"error": str(e), "model_type": model_type}


def compare_models_on_layout(
    layout: str,
    bc_dir: str = None,
    ppl_dir: str = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare all models on a layout.
    
    Returns:
        Dict mapping model names to metrics
    """
    bc_dir = bc_dir or os.path.join(BC_SAVE_DIR, "train")
    ppl_dir = ppl_dir or os.path.join(DATA_DIR, "ppl_runs")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing models on {layout}")
        print(f"{'='*60}")
    
    # Load test data
    try:
        test_states, test_actions = load_test_data(layout)
        if verbose:
            print(f"Loaded {len(test_states)} test samples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {}
    
    results = {}
    
    # Standard BC
    bc_path = os.path.join(bc_dir, layout)
    if os.path.exists(bc_path):
        metrics = evaluate_standard_bc(bc_path, test_states, test_actions)
        results["standard_bc"] = metrics
        if verbose and "error" not in metrics:
            print(f"  Standard BC: acc={metrics['accuracy']:.3f}, CE={metrics['cross_entropy']:.3f}")
    
    # PPL Models
    ppl_models = ["bayesian_bc", "rational_agent", "hierarchical_bc"]
    
    for model_type in ppl_models:
        model_dir = os.path.join(ppl_dir, model_type, layout)
        if os.path.exists(model_dir):
            metrics = evaluate_ppl_model(model_type, model_dir, test_states, test_actions)
            results[model_type] = metrics
            if verbose and "error" not in metrics:
                print(f"  {model_type}: acc={metrics['accuracy']:.3f}, CE={metrics['cross_entropy']:.3f}")
    
    return results


def create_comparison_table(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Create a markdown table comparing models."""
    
    # Collect all model names
    all_models = set()
    for layout_results in all_results.values():
        all_models.update(layout_results.keys())
    all_models = sorted(all_models)
    
    # Create header
    lines = ["| Layout | " + " | ".join(all_models) + " |"]
    lines.append("|" + "---|" * (len(all_models) + 1))
    
    # Add rows for accuracy
    lines.append("| **Accuracy** |" + " |" * len(all_models))
    for layout, results in all_results.items():
        row = f"| {layout} |"
        for model in all_models:
            if model in results and "accuracy" in results[model]:
                row += f" {results[model]['accuracy']:.3f} |"
            else:
                row += " - |"
        lines.append(row)
    
    # Add rows for cross-entropy
    lines.append("| **Cross-Entropy** |" + " |" * len(all_models))
    for layout, results in all_results.items():
        row = f"| {layout} |"
        for model in all_models:
            if model in results and "cross_entropy" in results[model]:
                row += f" {results[model]['cross_entropy']:.3f} |"
            else:
                row += " - |"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare PPL and baseline models")
    parser.add_argument("--layout", type=str, default=None, help="Single layout to evaluate")
    parser.add_argument("--all_layouts", action="store_true", help="Evaluate all layouts")
    parser.add_argument("--bc_dir", type=str, default=None)
    parser.add_argument("--ppl_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--markdown", action="store_true", help="Output markdown table")
    args = parser.parse_args()
    
    layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring",
               "forced_coordination", "counter_circuit"]
    
    if args.layout:
        layouts = [args.layout]
    elif not args.all_layouts:
        layouts = ["cramped_room"]  # Default
    
    all_results = {}
    
    for layout in layouts:
        results = compare_models_on_layout(
            layout,
            bc_dir=args.bc_dir,
            ppl_dir=args.ppl_dir,
        )
        all_results[layout] = results
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    for layout, results in all_results.items():
        print(f"\n{layout}:")
        for model, metrics in sorted(results.items()):
            if "error" not in metrics:
                print(f"  {model}: acc={metrics['accuracy']:.3f}, CE={metrics['cross_entropy']:.3f}")
    
    # Output markdown table
    if args.markdown:
        print("\n\nMarkdown Table:\n")
        print(create_comparison_table(all_results))
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
