"""
Unified Training Script for PPL-based Models

This script trains all PPL-based models (Bayesian BC, Rational Agent,
Hierarchical BC) on Overcooked layouts.

Usage:
    # Train all models on one layout
    python train_ppl.py --layout cramped_room
    
    # Train specific model
    python train_ppl.py --layout cramped_room --model bayesian_bc
    
    # Train all models on all layouts
    python train_ppl.py --all_layouts --all_models
    
    # Train with custom parameters
    python train_ppl.py --layout cramped_room --model hierarchical_bc --num_goals 12
"""

import argparse
import os
import time
import json
from typing import Dict, Any, List

import numpy as np

from human_aware_rl.ppl.bayesian_bc import train_bayesian_bc, BayesianBCConfig
from human_aware_rl.ppl.rational_agent import train_rational_agent, RationalAgentConfig
from human_aware_rl.ppl.hierarchical_bc import train_hierarchical_bc, HierarchicalBCConfig
from human_aware_rl.data_dir import DATA_DIR


LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages", 
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

MODELS = ["bayesian_bc", "rational_agent", "hierarchical_bc"]


def train_all_models(
    layout: str,
    models: List[str] = None,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all specified models on a layout.
    
    Args:
        layout: Layout name
        models: List of models to train (default: all)
        verbose: Print progress
        **kwargs: Additional arguments passed to trainers
        
    Returns:
        Dict mapping model names to training results
    """
    models = models or MODELS
    results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Training {model} on {layout}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if model == "bayesian_bc":
                # Extract relevant kwargs
                bc_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in ["num_epochs", "prior_scale", "hidden_dims", "batch_size"]
                }
                result = train_bayesian_bc(layout, verbose=verbose, **bc_kwargs)
                
            elif model == "rational_agent":
                ra_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in ["num_epochs", "learn_beta", "beta_init", "batch_size"]
                }
                result = train_rational_agent(layout, verbose=verbose, **ra_kwargs)
                
            elif model == "hierarchical_bc":
                hbc_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in ["num_epochs", "num_goals", "batch_size"]
                }
                result = train_hierarchical_bc(layout, verbose=verbose, **hbc_kwargs)
                
            else:
                print(f"Unknown model: {model}")
                continue
            
            elapsed = time.time() - start_time
            result["training_time"] = elapsed
            results[model] = result
            
            print(f"Completed {model} in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"Error training {model}: {e}")
            results[model] = {"error": str(e)}
    
    return results


def train_all_layouts(
    layouts: List[str] = None,
    models: List[str] = None,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Train models on all layouts.
    
    Returns:
        Nested dict: {layout: {model: results}}
    """
    layouts = layouts or LAYOUTS
    all_results = {}
    
    for layout in layouts:
        print(f"\n{'#'*70}")
        print(f"# Layout: {layout}")
        print(f"{'#'*70}")
        
        results = train_all_models(layout, models, verbose, **kwargs)
        all_results[layout] = results
    
    return all_results


def compare_models(results_dir: str, layouts: List[str] = None) -> Dict[str, Any]:
    """
    Compare trained models by computing metrics.
    
    This is a placeholder - extend with actual evaluation.
    """
    layouts = layouts or LAYOUTS
    comparison = {}
    
    for layout in layouts:
        comparison[layout] = {}
        
        for model in MODELS:
            model_dir = os.path.join(results_dir, model, layout)
            
            if os.path.exists(model_dir):
                # Load training curves
                # Compute validation metrics
                # etc.
                comparison[layout][model] = {"status": "trained"}
            else:
                comparison[layout][model] = {"status": "not_found"}
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Train PPL-based models for Overcooked AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Layout selection
    parser.add_argument(
        "--layout", 
        type=str, 
        default="cramped_room",
        help="Layout to train on",
    )
    parser.add_argument(
        "--all_layouts",
        action="store_true",
        help="Train on all layouts",
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS + ["all"],
        default="all",
        help="Model to train",
    )
    parser.add_argument(
        "--all_models",
        action="store_true", 
        help="Train all models (same as --model all)",
    )
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    
    # Model-specific parameters
    parser.add_argument("--prior_scale", type=float, default=1.0, help="Bayesian BC: weight prior scale")
    parser.add_argument("--num_goals", type=int, default=8, help="Hierarchical BC: number of latent goals")
    parser.add_argument("--learn_beta", action="store_true", help="Rational Agent: learn rationality parameter")
    
    # Output
    parser.add_argument("--results_dir", type=str, default=os.path.join(DATA_DIR, "ppl_runs"))
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    # Determine models to train
    if args.all_models or args.model == "all":
        models = MODELS
    else:
        models = [args.model]
    
    # Determine layouts
    if args.all_layouts:
        layouts = LAYOUTS
    else:
        layouts = [args.layout]
    
    verbose = args.verbose and not args.quiet
    
    # Build kwargs
    kwargs = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "results_dir": args.results_dir,
        "seed": args.seed,
        "prior_scale": args.prior_scale,
        "num_goals": args.num_goals,
        "learn_beta": args.learn_beta,
    }
    
    print(f"\n{'='*70}")
    print(f"PPL Model Training")
    print(f"{'='*70}")
    print(f"Layouts: {layouts}")
    print(f"Models: {models}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Results dir: {args.results_dir}")
    print()
    
    # Train
    all_results = {}
    for layout in layouts:
        results = train_all_models(layout, models, verbose, **kwargs)
        all_results[layout] = results
    
    # Save summary
    summary_path = os.path.join(args.results_dir, "training_summary.json")
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    with open(summary_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
