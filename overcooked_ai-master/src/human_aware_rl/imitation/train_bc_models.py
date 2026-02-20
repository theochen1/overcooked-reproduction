"""
Batch training script for Behavior Cloning models.

This script trains BC models on human demonstration data for all layouts,
creating both training-set models (BC) and test-set models (Human Proxy/HP).

Usage:
    # Train all layouts
    python -m human_aware_rl.imitation.train_bc_models --all_layouts
    
    # Train specific layout
    python -m human_aware_rl.imitation.train_bc_models --layout cramped_room
    
    # Train only human proxy models (test set)
    python -m human_aware_rl.imitation.train_bc_models --all_layouts --hp_only
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.imitation.behavior_cloning import (
    BC_SAVE_DIR,
    get_bc_params,
    train_bc_model,
    evaluate_bc_model,
)
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TRAIN,
    CLEAN_2019_HUMAN_DATA_TEST,
    LAYOUTS_WITH_DATA_2019,
)


# Layout mapping: paper names to data layout names
# The paper uses these 5 layouts:
# - cramped_room
# - asymmetric_advantages  
# - coordination_ring
# - forced_coordination (mapped to random0 in 2019 data)
# - counter_circuit (mapped to random3 in 2019 data, uses counter_circuit_o_1order for training)
PAPER_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages", 
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

# Mapping from paper layout names to data layout names
LAYOUT_TO_DATA = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "random0",
    "counter_circuit": "random3",
}

# Mapping from paper layout names to environment layout names (for evaluation)
LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}


# Paper hyperparameters for BC (Paper Table 1)
# Common params shared across all layouts
PAPER_BC_COMMON = {
    "mlp_params": {
        "num_layers": 2,
        "net_arch": [64, 64],
    },
    "training_params": {
        "validation_split": 0.15,
        "batch_size": 64,
        "learning_rate": 1e-3,     # Paper Table 1: same for all layouts
        "adam_epsilon": 1e-8,       # Paper Table 1: explicit for reproducibility
        "use_class_weights": False,
        "patience": 20,
        "lr_patience": 3,
        "lr_factor": 0.1,
    },
    "evaluation_params": {
        "ep_length": 400,
        "num_games": 5,
        "display": False,
    },
}

# Per-layout epoch counts from Paper Table 1
PAPER_BC_EPOCHS = {
    "cramped_room": 100,
    "asymmetric_advantages": 120,
    "coordination_ring": 120,
    "forced_coordination": 90,
    "counter_circuit": 110,
}


def train_bc_for_layout(
    layout: str,
    data_split: str = "train",
    output_dir: Optional[str] = None,
    verbose: bool = True,
    evaluate: bool = True,
) -> Dict:
    """
    Train a BC model for a specific layout.
    
    Args:
        layout: Paper layout name (e.g., 'cramped_room')
        data_split: 'train' for BC model, 'test' for Human Proxy model
        output_dir: Directory to save model (default: BC_SAVE_DIR/<split>/<layout>)
        verbose: Whether to print progress
        evaluate: Whether to evaluate the model after training
        
    Returns:
        Dictionary with training results
    """
    # Get data layout name
    data_layout = LAYOUT_TO_DATA.get(layout, layout)
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Determine data path
    if data_split == "train":
        data_path = CLEAN_2019_HUMAN_DATA_TRAIN
    elif data_split == "test":
        data_path = CLEAN_2019_HUMAN_DATA_TEST
    else:
        raise ValueError(f"Invalid data_split: {data_split}. Use 'train' or 'test'.")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(BC_SAVE_DIR, data_split, layout)
    
    # Get per-layout epoch count from Paper Table 1
    epochs = PAPER_BC_EPOCHS.get(layout, 100)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training BC model for layout: {layout}")
        print(f"Data split: {data_split}")
        print(f"Data layout: {data_layout}")
        print(f"Environment layout: {env_layout}")
        print(f"Epochs: {epochs} (Paper Table 1)")
        print(f"Learning rate: {PAPER_BC_COMMON['training_params']['learning_rate']}")
        print(f"Adam epsilon: {PAPER_BC_COMMON['training_params']['adam_epsilon']}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
    
    # Get BC parameters with paper settings
    bc_params = get_bc_params(
        layouts=[data_layout],
        data_path=data_path,
        layout_name=env_layout,
        old_dynamics=True,  # Paper uses old dynamics
        epochs=epochs,      # Per-layout from Paper Table 1
        **PAPER_BC_COMMON["mlp_params"],
        **PAPER_BC_COMMON["training_params"],
        **PAPER_BC_COMMON["evaluation_params"],
    )
    
    # Train model
    model = train_bc_model(output_dir, bc_params, verbose=verbose)
    
    results = {
        "layout": layout,
        "data_split": data_split,
        "output_dir": output_dir,
    }
    
    # Evaluate if requested
    if evaluate:
        if verbose:
            print(f"\nEvaluating model...")
        reward = evaluate_bc_model(model, bc_params, verbose=verbose)
        results["evaluation_reward"] = reward
        if verbose:
            print(f"Evaluation reward: {reward:.2f}")
    
    return results


def train_all_layouts(
    data_split: str = "train",
    layouts: Optional[List[str]] = None,
    verbose: bool = True,
    evaluate: bool = True,
    base_output_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Train BC models for all layouts.
    
    Args:
        data_split: 'train' for BC models, 'test' for Human Proxy models
        layouts: List of layouts to train (default: all paper layouts)
        verbose: Whether to print progress
        evaluate: Whether to evaluate models after training
        base_output_dir: Base directory for output (default: BC_SAVE_DIR)
                        Models saved to base_output_dir/<split>/<layout>/
        
    Returns:
        Dictionary mapping layout names to training results
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    
    results = {}
    
    for layout in layouts:
        # Compute output_dir for this layout
        if base_output_dir:
            output_dir = os.path.join(base_output_dir, data_split, layout)
        else:
            output_dir = None  # Will use default BC_SAVE_DIR/<split>/<layout>
        
        try:
            result = train_bc_for_layout(
                layout=layout,
                data_split=data_split,
                output_dir=output_dir,
                verbose=verbose,
                evaluate=evaluate,
            )
            results[layout] = result
        except Exception as e:
            print(f"Error training {layout}: {e}")
            results[layout] = {"error": str(e)}
    
    return results


def train_all_models(
    layouts: Optional[List[str]] = None,
    verbose: bool = True,
    evaluate: bool = True,
    base_output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Dict]]:
    """
    Train both BC and Human Proxy models for all layouts.
    
    Args:
        layouts: List of layouts to train (default: all paper layouts)
        verbose: Whether to print progress
        evaluate: Whether to evaluate models after training
        base_output_dir: Base directory for output (default: BC_SAVE_DIR)
                        Models saved to base_output_dir/<split>/<layout>/
        
    Returns:
        Dictionary with 'bc' and 'hp' keys, each mapping layouts to results
    """
    results = {
        "bc": {},
        "hp": {},
    }
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING BC MODELS (training data)")
        print("="*60)
    
    results["bc"] = train_all_layouts(
        data_split="train",
        layouts=layouts,
        verbose=verbose,
        evaluate=evaluate,
        base_output_dir=base_output_dir,
    )
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING HUMAN PROXY MODELS (test data)")
        print("="*60)
    
    results["hp"] = train_all_layouts(
        data_split="test",
        layouts=layouts,
        verbose=verbose,
        evaluate=evaluate,
        base_output_dir=base_output_dir,
    )
    
    return results


def print_summary(results: Dict[str, Dict[str, Dict]]):
    """Print a summary of training results."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()} Models:")
        print("-"*40)
        
        for layout, result in model_results.items():
            if "error" in result:
                print(f"  {layout}: ERROR - {result['error']}")
            elif "evaluation_reward" in result:
                print(f"  {layout}: reward = {result['evaluation_reward']:.2f}")
            else:
                print(f"  {layout}: trained (no evaluation)")


def main():
    parser = argparse.ArgumentParser(
        description="Train BC models for Overcooked AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        choices=PAPER_LAYOUTS,
        help="Train a single layout"
    )
    
    parser.add_argument(
        "--all_layouts",
        action="store_true",
        help="Train all 5 paper layouts"
    )
    
    parser.add_argument(
        "--bc_only",
        action="store_true",
        help="Train only BC models (training data)"
    )
    
    parser.add_argument(
        "--hp_only",
        action="store_true",
        help="Train only Human Proxy models (test data)"
    )
    
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (only for single layout)"
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base output directory for all layouts (models saved to base/<split>/<layout>/)"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    evaluate = not args.no_eval
    
    if args.layout:
        # Train single layout
        # Use output_dir if specified, otherwise construct from output_base_dir
        output_dir = args.output_dir
        if output_dir is None and args.output_base_dir:
            # For single layout, we construct the path based on split
            pass  # Will be set per-split below
        
        if args.hp_only:
            od = args.output_dir or (os.path.join(args.output_base_dir, "test", args.layout) if args.output_base_dir else None)
            results = {"hp": {args.layout: train_bc_for_layout(
                args.layout, "test", od, verbose, evaluate
            )}}
        elif args.bc_only:
            od = args.output_dir or (os.path.join(args.output_base_dir, "train", args.layout) if args.output_base_dir else None)
            results = {"bc": {args.layout: train_bc_for_layout(
                args.layout, "train", od, verbose, evaluate
            )}}
        else:
            od_train = args.output_dir or (os.path.join(args.output_base_dir, "train", args.layout) if args.output_base_dir else None)
            od_test = args.output_dir or (os.path.join(args.output_base_dir, "test", args.layout) if args.output_base_dir else None)
            results = {
                "bc": {args.layout: train_bc_for_layout(
                    args.layout, "train", od_train, verbose, evaluate
                )},
                "hp": {args.layout: train_bc_for_layout(
                    args.layout, "test", od_test, verbose, evaluate
                )},
            }
    elif args.all_layouts:
        # Train all layouts
        if args.hp_only:
            results = {"hp": train_all_layouts("test", None, verbose, evaluate, args.output_base_dir)}
        elif args.bc_only:
            results = {"bc": train_all_layouts("train", None, verbose, evaluate, args.output_base_dir)}
        else:
            results = train_all_models(None, verbose, evaluate, args.output_base_dir)
    else:
        parser.print_help()
        print("\nError: Must specify --layout or --all_layouts")
        sys.exit(1)
    
    print_summary(results)
    
    return results


if __name__ == "__main__":
    main()

