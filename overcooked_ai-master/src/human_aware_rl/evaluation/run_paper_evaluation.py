"""
Unified Paper Evaluation Pipeline for Overcooked AI.

This script provides a complete pipeline to:
1. Check all required models exist
2. Run all Figure 4 evaluations
3. Generate paper-style figures

Usage:
    # Full evaluation (requires all models trained)
    python -m human_aware_rl.evaluation.run_paper_evaluation --all
    
    # Just Figure 4(a) - Self-play comparison
    python -m human_aware_rl.evaluation.run_paper_evaluation --figure 4a
    
    # Just Figure 4(b) - PBT comparison
    python -m human_aware_rl.evaluation.run_paper_evaluation --figure 4b
    
    # Check model availability without running
    python -m human_aware_rl.evaluation.run_paper_evaluation --check_only
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Any

from human_aware_rl.ppo.configs.paper_configs import PAPER_LAYOUTS
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR


# Default directories
DEFAULT_DIRS = {
    "ppo_sp": "results/ppo_sp",
    "ppo_bc": "results/ppo_bc",
    "ppo_hp": "results/ppo_hp",
    "pbt": "results/pbt",
    "bc": os.path.join(BC_SAVE_DIR, "train"),
    "hp": os.path.join(BC_SAVE_DIR, "test"),
}


def check_model_availability(
    layouts: List[str] = None,
    seeds: List[int] = None,
    dirs: Dict[str, str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, bool]]:
    """
    Check which models are available for evaluation.
    
    Args:
        layouts: Layouts to check (default: all paper layouts)
        seeds: Seeds to check (default: [0, 10, 20, 30, 40])
        dirs: Model directories (default: DEFAULT_DIRS)
        verbose: Whether to print status
        
    Returns:
        Dict mapping model type -> layout -> availability status
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    if seeds is None:
        seeds = [0, 10, 20, 30, 40]
    if dirs is None:
        dirs = DEFAULT_DIRS
    
    availability = {}
    
    # Check BC models (trained on training data)
    availability["bc"] = {}
    for layout in layouts:
        bc_path = os.path.join(dirs["bc"], layout)
        availability["bc"][layout] = os.path.exists(bc_path)
    
    # Check HP models (trained on test data)
    availability["hp"] = {}
    for layout in layouts:
        hp_path = os.path.join(dirs["hp"], layout)
        availability["hp"][layout] = os.path.exists(hp_path)
    
    # Check PPO models (need at least one seed)
    for model_type in ["ppo_sp", "ppo_bc", "ppo_hp", "pbt"]:
        availability[model_type] = {}
        base_dir = dirs[model_type]
        
        for layout in layouts:
            found = False
            if os.path.exists(base_dir):
                for exp_name in os.listdir(base_dir):
                    if layout in exp_name:
                        exp_dir = os.path.join(base_dir, exp_name)
                        if os.path.isdir(exp_dir):
                            checkpoints = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint")]
                            if checkpoints:
                                found = True
                                break
            availability[model_type][layout] = found
    
    if verbose:
        print("\n" + "="*70)
        print("MODEL AVAILABILITY CHECK")
        print("="*70)
        
        for model_type in ["bc", "hp", "ppo_sp", "ppo_bc", "ppo_hp", "pbt"]:
            print(f"\n{model_type.upper()}:")
            for layout in layouts:
                status = "✓" if availability[model_type][layout] else "✗"
                print(f"  {layout}: {status}")
        
        # Summary
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        
        # Figure 4(a) requirements
        fig_4a_ready = all([
            all(availability["bc"].values()),
            all(availability["hp"].values()),
            all(availability["ppo_sp"].values()),
            all(availability["ppo_bc"].values()),
        ])
        fig_4a_gold = all(availability["ppo_hp"].values())
        
        # Figure 4(b) requirements  
        fig_4b_ready = all([
            all(availability["bc"].values()),
            all(availability["hp"].values()),
            all(availability["pbt"].values()),
            all(availability["ppo_bc"].values()),
        ])
        
        print(f"Figure 4(a) - Self-Play Comparison: {'Ready' if fig_4a_ready else 'Missing models'}")
        print(f"  Gold standard (PPO_HP): {'Ready' if fig_4a_gold else 'Missing'}")
        print(f"Figure 4(b) - PBT Comparison: {'Ready' if fig_4b_ready else 'Missing models'}")
        
        if not fig_4a_ready:
            print("\nTo prepare for Figure 4(a), run:")
            if not all(availability["bc"].values()) or not all(availability["hp"].values()):
                print("  python -m human_aware_rl.imitation.train_bc_models --all_layouts")
            if not all(availability["ppo_sp"].values()):
                print("  python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40")
            if not all(availability["ppo_bc"].values()):
                print("  python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0,10,20,30,40")
            if not fig_4a_gold:
                print("  python -m human_aware_rl.ppo.train_ppo_hp --all_layouts --seeds 0,10,20,30,40")
        
        if not fig_4b_ready:
            print("\nTo prepare for Figure 4(b), run:")
            if not all(availability["pbt"].values()):
                print("  python -m human_aware_rl.ppo.train_pbt --all_layouts")
    
    return availability


def run_evaluation(
    figure: str = "all",
    layouts: List[str] = None,
    seeds: List[int] = None,
    dirs: Dict[str, str] = None,
    num_games: int = 10,
    output_file: str = "paper_results.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run paper evaluations.
    
    Args:
        figure: Which figure to evaluate ("4a", "4b", or "all")
        layouts: Layouts to evaluate
        seeds: Seeds to average over
        dirs: Model directories
        num_games: Number of games per evaluation
        output_file: Path to save results
        verbose: Whether to print progress
        
    Returns:
        Evaluation results
    """
    from human_aware_rl.evaluation.evaluate_paper import (
        evaluate_figure_4a,
        evaluate_figure_4b,
        evaluate_all_paper_experiments,
    )
    
    if layouts is None:
        layouts = PAPER_LAYOUTS
    if seeds is None:
        seeds = [0, 10, 20, 30, 40]
    if dirs is None:
        dirs = DEFAULT_DIRS
    
    results = {}
    
    if figure in ["all", "4a"]:
        if verbose:
            print("\n" + "#"*70)
            print("# Running Figure 4(a) Evaluation - Self-Play Comparison")
            print("#"*70)
        
        results["figure_4a"] = evaluate_figure_4a(
            ppo_sp_dir=dirs["ppo_sp"],
            ppo_bc_dir=dirs["ppo_bc"],
            ppo_hp_dir=dirs["ppo_hp"],
            bc_dir=dirs["bc"],
            hp_dir=dirs["hp"],
            layouts=layouts,
            seeds=seeds,
            num_games=num_games,
            verbose=verbose,
        )
    
    if figure in ["all", "4b"]:
        if verbose:
            print("\n" + "#"*70)
            print("# Running Figure 4(b) Evaluation - PBT Comparison")
            print("#"*70)
        
        results["figure_4b"] = evaluate_figure_4b(
            ppo_bc_dir=dirs["ppo_bc"],
            ppo_hp_dir=dirs["ppo_hp"],
            pbt_dir=dirs["pbt"],
            bc_dir=dirs["bc"],
            hp_dir=dirs["hp"],
            layouts=layouts,
            seeds=seeds,
            num_games=num_games,
            verbose=verbose,
        )
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to {output_file}")
    
    return results


def generate_figures(
    results_file: str = "paper_results.json",
    output_dir: str = "figures",
    fmt: str = "pdf",
    verbose: bool = True,
):
    """
    Generate Figure 4 from evaluation results.
    
    Args:
        results_file: Path to evaluation results JSON
        output_dir: Directory to save figures
        fmt: Output format (pdf, png, svg)
        verbose: Whether to print progress
    """
    from human_aware_rl.visualization.plot_results import plot_figure_4
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    if verbose:
        print("\n" + "#"*70)
        print("# Generating Figure 4")
        print("#"*70)
    
    plot_figure_4(
        results=results,
        output_dir=output_dir,
        fmt=fmt,
        show=False,
    )
    
    if verbose:
        print(f"\nFigures saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Paper Evaluation Pipeline for Overcooked AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Action arguments
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full evaluation pipeline"
    )
    
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check model availability, don't run evaluation"
    )
    
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate figures from existing results"
    )
    
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=["4a", "4b", "all"],
        help="Which figure to evaluate"
    )
    
    # Input/output arguments
    parser.add_argument(
        "--results_file",
        type=str,
        default="paper_results.json",
        help="Path to save/load evaluation results"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save figures"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for figures"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--layouts",
        type=str,
        default=None,
        help="Comma-separated list of layouts (default: all)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,10,20,30,40",
        help="Comma-separated list of seeds"
    )
    
    parser.add_argument(
        "--num_games",
        type=int,
        default=10,
        help="Number of games per evaluation"
    )
    
    # Directory overrides
    parser.add_argument("--ppo_sp_dir", type=str, default=DEFAULT_DIRS["ppo_sp"])
    parser.add_argument("--ppo_bc_dir", type=str, default=DEFAULT_DIRS["ppo_bc"])
    parser.add_argument("--ppo_hp_dir", type=str, default=DEFAULT_DIRS["ppo_hp"])
    parser.add_argument("--pbt_dir", type=str, default=DEFAULT_DIRS["pbt"])
    parser.add_argument("--bc_dir", type=str, default=DEFAULT_DIRS["bc"])
    parser.add_argument("--hp_dir", type=str, default=DEFAULT_DIRS["hp"])
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    layouts = args.layouts.split(",") if args.layouts else None
    seeds = [int(s) for s in args.seeds.split(",")]
    
    dirs = {
        "ppo_sp": args.ppo_sp_dir,
        "ppo_bc": args.ppo_bc_dir,
        "ppo_hp": args.ppo_hp_dir,
        "pbt": args.pbt_dir,
        "bc": args.bc_dir,
        "hp": args.hp_dir,
    }
    
    # Check model availability
    availability = check_model_availability(
        layouts=layouts or PAPER_LAYOUTS,
        seeds=seeds,
        dirs=dirs,
        verbose=verbose,
    )
    
    if args.check_only:
        return
    
    if args.plot_only:
        generate_figures(
            results_file=args.results_file,
            output_dir=args.output_dir,
            fmt=args.format,
            verbose=verbose,
        )
        return
    
    # Run evaluation
    if args.all or args.figure:
        results = run_evaluation(
            figure=args.figure,
            layouts=layouts,
            seeds=seeds,
            dirs=dirs,
            num_games=args.num_games,
            output_file=args.results_file,
            verbose=verbose,
        )
        
        # Generate figures
        generate_figures(
            results_file=args.results_file,
            output_dir=args.output_dir,
            fmt=args.format,
            verbose=verbose,
        )
    else:
        parser.print_help()
        print("\nRun with --all to execute full pipeline, or --check_only to verify models")


if __name__ == "__main__":
    main()

