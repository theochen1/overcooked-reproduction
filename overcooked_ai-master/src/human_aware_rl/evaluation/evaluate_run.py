"""
Generic evaluation script for any run (run3, run4, ...).

Features:
- Parameterized by run number
- Auto-detect model locations using discovered directory structure
- Evaluates BC, GAIL, PPO-SP, PPO-BC, PPO-GAIL with Human Proxy
- Optional plotting and configurable output file names
"""

import argparse
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.evaluation import model_utils as mu


def evaluate_pair(agent1, agent2, layout, num_games=5, swapped=False):
    """Evaluate an agent pair on a layout."""
    env_layout = mu.LAYOUT_TO_ENV.get(layout, layout)

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
    return results["ep_returns"]


def load_bc_and_hp(paths: Dict[str, str], layout: str, verbose: bool):
    """Load BC (train) and HP (test) models."""
    bc_model = None
    hp_model = None

    # HP (test)
    hp_dir = os.path.join(paths["bc_test"], layout)
    try:
        hp_model, _ = mu.load_bc_model(hp_dir)
        if verbose:
            print("  ✓ Loaded HP", hp_dir)
    except Exception as e:
        print(f"  ✗ Failed to load HP at {hp_dir}: {e}")

    # BC (train)
    bc_dir = os.path.join(paths["bc_train"], layout)
    try:
        bc_model, _ = mu.load_bc_model(bc_dir)
        if verbose:
            print("  ✓ Loaded BC", bc_dir)
    except Exception as e:
        if verbose:
            print(f"  ✗ Failed to load BC at {bc_dir}: {e}")
        bc_model = None

    return bc_model, hp_model


def load_gail(paths: Dict[str, str], layout: str, featurize_fn, verbose: bool):
    """Load GAIL model if available."""
    gail_dir = os.path.join(paths["gail"], layout)
    gail_model = None
    try:
        # Build a sample observation for sizing the model
        env_layout = mu.LAYOUT_TO_ENV.get(layout, layout)
        tmp_ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )
        sample_state = tmp_ae.env.state
        sample_obs = featurize_fn(sample_state)[0]
        state_dim = sample_obs.flatten().shape[0]
        gail_model = mu.load_gail_model(gail_dir, state_dim=state_dim)
        if verbose:
            print("  ✓ Loaded GAIL", gail_dir)
    except Exception as e:
        if verbose:
            print(f"  ✗ Failed to load GAIL at {gail_dir}: {e}")
        gail_model = None
    return gail_model


def load_ppo_dir(base_dir: str, prefix: str, layout: str, seed: int, verbose: bool):
    """Load PPO model from the expected structure. Returns (params, config) or (None, None)."""
    model_dir = os.path.join(base_dir, f"{prefix}_{layout}_seed{seed}")
    try:
        return mu.load_ppo_model(model_dir, verbose=verbose)
    except Exception:
        # fallback: maybe params are directly under the dir
        try:
            return mu.load_ppo_model_from_files(model_dir, verbose=verbose)
        except Exception as e2:
            if verbose:
                print(f"    Seed {seed}: Error - {e2}")
            return None, None


def evaluate_all_seeds(
    layout: str,
    agent_type: str,
    paths: Dict[str, str],
    hp_model,
    bc_model,
    featurize_fn,
    num_games_per_seed: int,
    swapped: bool,
    verbose: bool,
    gail_model=None,
) -> Dict:
    """Evaluate across all seeds and aggregate."""

    all_rewards = []

    for seed in mu.SEEDS:
        if agent_type == "bc_hp":
            if seed != mu.SEEDS[0]:
                continue
            agent = mu.BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "gail_hp":
            if seed != mu.SEEDS[0]:
                continue
            if gail_model is None:
                if verbose:
                    print("    No GAIL model available")
                continue
            agent = mu.GAILAgentWrapper(gail_model, featurize_fn, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "sp_sp":
            params, config = load_ppo_dir(paths["ppo_sp"], "ppo_sp", layout, seed, verbose=(seed == mu.SEEDS[0] and verbose))
            if params is None:
                continue
            agent1 = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            agent2 = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            rewards = evaluate_pair(agent1, agent2, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "sp_hp":
            params, config = load_ppo_dir(paths["ppo_sp"], "ppo_sp", layout, seed, verbose=False)
            if params is None:
                continue
            sp_agent = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(sp_agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "ppo_bc_hp":
            params, config = load_ppo_dir(paths["ppo_bc"], "ppo_bc", layout, seed, verbose=False)
            if params is None:
                continue
            ppo_bc_agent = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(ppo_bc_agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "ppo_gail_hp":
            params, config = load_ppo_dir(paths["ppo_gail"], "ppo_gail", layout, seed, verbose=False)
            if params is None:
                continue
            ppo_gail_agent = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(ppo_gail_agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

    if len(all_rewards) == 0:
        return {"mean": 0, "std": 0, "se": 0, "n": 0}

    return {
        "mean": float(np.mean(all_rewards)),
        "std": float(np.std(all_rewards)),
        "se": float(np.std(all_rewards) / np.sqrt(len(all_rewards))),
        "n": len(all_rewards),
    }


def run_full_evaluation(
    run_number: int,
    layouts: Sequence[str],
    num_games_per_seed: int = 5,
    models_base_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Run full evaluation for a given run number."""

    paths = mu.get_model_paths(run_number, base_dir=models_base_dir)
    results = {}

    for layout in layouts:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layout: {layout} (run {run_number})")
            print(f"{'='*60}")

        env_layout = mu.LAYOUT_TO_ENV.get(layout, layout)
        layout_results = {}

        # Setup environment
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )

        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)

        # Load models
        bc_model, hp_model = load_bc_and_hp(paths, layout, verbose=verbose)
        if hp_model is None:
            if verbose:
                print("  ✗ Skipping layout due to missing HP model")
            continue

        gail_model = load_gail(paths, layout, featurize_fn, verbose=verbose)

        # Evaluate SP+SP (baseline)
        if verbose:
            print("\n  Evaluating SP+SP...")
        layout_results["sp_sp"] = evaluate_all_seeds(
            layout,
            "sp_sp",
            paths,
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=False,
            verbose=verbose,
            gail_model=gail_model,
        )

        # Evaluate SP+HP
        if verbose:
            print("\n  Evaluating SP+HP...")
        layout_results["sp_hp"] = evaluate_all_seeds(
            layout,
            "sp_hp",
            paths,
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=False,
            verbose=verbose,
            gail_model=gail_model,
        )
        layout_results["sp_hp_swapped"] = evaluate_all_seeds(
            layout,
            "sp_hp",
            paths,
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=True,
            verbose=verbose,
            gail_model=gail_model,
        )

        # Evaluate PPO_BC+HP
        if verbose:
            print("\n  Evaluating PPO_BC+HP...")
        layout_results["ppo_bc_hp"] = evaluate_all_seeds(
            layout,
            "ppo_bc_hp",
            paths,
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=False,
            verbose=verbose,
            gail_model=gail_model,
        )
        layout_results["ppo_bc_hp_swapped"] = evaluate_all_seeds(
            layout,
            "ppo_bc_hp",
            paths,
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=True,
            verbose=verbose,
            gail_model=gail_model,
        )

        # Evaluate BC+HP
        if bc_model is not None:
            if verbose:
                print("\n  Evaluating BC+HP...")
            layout_results["bc_hp"] = evaluate_all_seeds(
                layout,
                "bc_hp",
                paths,
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=False,
                verbose=verbose,
                gail_model=gail_model,
            )
            layout_results["bc_hp_swapped"] = evaluate_all_seeds(
                layout,
                "bc_hp",
                paths,
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=True,
                verbose=verbose,
                gail_model=gail_model,
            )

        # Evaluate GAIL+HP
        if gail_model is not None:
            if verbose:
                print("\n  Evaluating GAIL+HP...")
            layout_results["gail_hp"] = evaluate_all_seeds(
                layout,
                "gail_hp",
                paths,
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=False,
                verbose=verbose,
                gail_model=gail_model,
            )
            layout_results["gail_hp_swapped"] = evaluate_all_seeds(
                layout,
                "gail_hp",
                paths,
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=True,
                verbose=verbose,
                gail_model=gail_model,
            )

        # Evaluate PPO_GAIL+HP
        if verbose:
            print("\n  Evaluating PPO_GAIL+HP...")
        layout_results["ppo_gail_hp"] = evaluate_all_seeds(
            layout,
            "ppo_gail_hp",
            paths,
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=False,
            verbose=verbose,
            gail_model=gail_model,
        )
        layout_results["ppo_gail_hp_swapped"] = evaluate_all_seeds(
            layout,
            "ppo_gail_hp",
            paths,
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=True,
            verbose=verbose,
            gail_model=gail_model,
        )

        results[layout] = layout_results

    return results


def print_table(results: Dict, run_number: int):
    """Print results in a concise table."""
    print("\n" + "=" * 140)
    print(f"Run {run_number} Evaluation Results")
    print("=" * 140)
    print(f"{'Layout':<22} {'SP+SP':<12} {'SP+HP':<12} {'PPO_BC+HP':<12} {'PPO_GAIL+HP':<12} {'GAIL+HP':<12} {'BC+HP':<12}")
    print("-" * 140)

    for layout in mu.LAYOUTS:
        if layout not in results:
            continue
        r = results[layout]

        sp_sp = f"{r['sp_sp']['mean']:.1f}±{r['sp_sp']['se']:.1f}" if 'sp_sp' in r else "N/A"
        sp_hp = f"{r['sp_hp']['mean']:.1f}±{r['sp_hp']['se']:.1f}" if 'sp_hp' in r else "N/A"
        ppo_bc = f"{r['ppo_bc_hp']['mean']:.1f}±{r['ppo_bc_hp']['se']:.1f}" if 'ppo_bc_hp' in r else "N/A"
        ppo_gail = f"{r['ppo_gail_hp']['mean']:.1f}±{r['ppo_gail_hp']['se']:.1f}" if 'ppo_gail_hp' in r else "N/A"
        gail_hp = f"{r.get('gail_hp', {}).get('mean', 0):.1f}±{r.get('gail_hp', {}).get('se', 0):.1f}" if 'gail_hp' in r else "N/A"
        bc_hp = f"{r.get('bc_hp', {}).get('mean', 0):.1f}±{r.get('bc_hp', {}).get('se', 0):.1f}" if 'bc_hp' in r else "N/A"

        print(f"{layout:<22} {sp_sp:<12} {sp_hp:<12} {ppo_bc:<12} {ppo_gail:<12} {gail_hp:<12} {bc_hp:<12}")

    print("=" * 140)


def main():
    parser = argparse.ArgumentParser(description="Evaluate models for a given run")
    parser.add_argument("--run_number", type=int, required=True, help="Run number (e.g., 3, 4)")
    parser.add_argument("--num_games", type=int, default=5, help="Games per seed")
    parser.add_argument("--layouts", nargs="*", default=mu.LAYOUTS, help="Layouts to evaluate")
    parser.add_argument("--models_base_dir", type=str, default=None, help="Custom base dir for models")
    parser.add_argument("--save_results", type=str, default=None, help="Results JSON output path")
    parser.add_argument("--save_plot", type=str, default=None, help="Plot output path")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Default filenames
    results_path = args.save_results or f"run{args.run_number}_results.json"
    plot_path = args.save_plot or f"run{args.run_number}_figure.png"

    print("=" * 60)
    print(f"Run {args.run_number} Evaluation")
    print("=" * 60)
    print(f"Layouts: {args.layouts}")
    print(f"Games per seed: {args.num_games}")

    results = run_full_evaluation(
        run_number=args.run_number,
        layouts=args.layouts,
        num_games_per_seed=args.num_games,
        models_base_dir=args.models_base_dir,
        verbose=args.verbose,
    )

    print_table(results, args.run_number)

    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot (reuse plotting from run3 script if available)
    if not args.no_plot:
        try:
            from human_aware_rl.evaluation.evaluate_run3_paper import plot_figure_4

            plot_figure_4(results, plot_path)
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
