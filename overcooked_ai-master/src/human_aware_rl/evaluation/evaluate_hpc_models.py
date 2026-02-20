"""
Evaluate HPC-trained models with Human Proxy (HP).

Replicates Figure 4-style evaluations for:
- BC + HP
- PPO_SP + HP
- PPO_BC + HP
- PPO_GAIL + HP
Optionally includes GAIL + HP as an extension.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.evaluation import model_utils as mu


# Auto-detect base directory relative to this file's location
# Falls back to environment variable or reasonable default
BASE_MODEL_DIR = os.environ.get(
    "OVERCOOKED_MODEL_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

PATHS = {
    "bc_train": os.path.join(BASE_MODEL_DIR, "bc_runs", "train"),
    "bc_test": os.path.join(BASE_MODEL_DIR, "bc_runs", "test"),
    "gail": os.path.join(BASE_MODEL_DIR, "gail_runs"),
    "ppo_sp": os.path.join(BASE_MODEL_DIR, "results", "ppo_sp"),
    "ppo_bc": os.path.join(BASE_MODEL_DIR, "results", "ppo_bc"),
    "ppo_gail": os.path.join(BASE_MODEL_DIR, "results", "ppo_gail"),
}


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


def load_bc_and_hp(layout: str, verbose: bool):
    """Load BC (train) and HP (test) models."""
    bc_model = None
    hp_model = None

    hp_dir = os.path.join(PATHS["bc_test"], layout)
    try:
        hp_model, _ = mu.load_bc_model(hp_dir)
        if verbose:
            print("  ✓ Loaded HP", hp_dir)
    except Exception as exc:
        print(f"  ✗ Failed to load HP at {hp_dir}: {exc}")

    bc_dir = os.path.join(PATHS["bc_train"], layout)
    try:
        bc_model, _ = mu.load_bc_model(bc_dir)
        if verbose:
            print("  ✓ Loaded BC", bc_dir)
    except Exception as exc:
        if verbose:
            print(f"  ✗ Failed to load BC at {bc_dir}: {exc}")
        bc_model = None

    return bc_model, hp_model


def load_gail(layout: str, featurize_fn, verbose: bool):
    """Load GAIL model if available."""
    gail_dir = os.path.join(PATHS["gail"], layout)
    gail_model = None
    try:
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
    except Exception as exc:
        if verbose:
            print(f"  ✗ Failed to load GAIL at {gail_dir}: {exc}")
        gail_model = None
    return gail_model


def load_ppo_dir(base_dir: str, prefix: str, layout: str, seed: int, verbose: bool):
    """Load PPO model from the expected structure."""
    model_dir = os.path.join(base_dir, f"{prefix}_{layout}_seed{seed}")
    try:
        return mu.load_ppo_model(model_dir, verbose=verbose)
    except Exception:
        try:
            return mu.load_ppo_model_from_files(model_dir, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"    Seed {seed}: Error - {exc}")
            return None, None


def evaluate_all_seeds(
    layout: str,
    agent_type: str,
    hp_model,
    bc_model,
    featurize_fn,
    num_games_per_seed: int,
    swapped: bool,
    verbose: bool,
    gail_model=None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict:
    """Evaluate across all seeds and aggregate."""
    all_rewards: List[float] = []
    seeds = list(seeds or mu.SEEDS)

    for seed in seeds:
        if agent_type == "bc_hp":
            if seed != seeds[0]:
                continue
            agent = mu.BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "gail_hp":
            if seed != seeds[0]:
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
            params, config = load_ppo_dir(
                PATHS["ppo_sp"],
                "ppo_sp",
                layout,
                seed,
                verbose=(seed == seeds[0] and verbose),
            )
            if params is None:
                continue
            agent1 = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            agent2 = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            rewards = evaluate_pair(agent1, agent2, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "sp_hp":
            params, config = load_ppo_dir(PATHS["ppo_sp"], "ppo_sp", layout, seed, verbose=False)
            if params is None:
                continue
            sp_agent = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(sp_agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "ppo_bc_hp":
            params, config = load_ppo_dir(PATHS["ppo_bc"], "ppo_bc", layout, seed, verbose=False)
            if params is None:
                continue
            ppo_bc_agent = mu.PPOAgentWrapper(params, config, layout, stochastic=True)
            hp_agent = mu.BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(ppo_bc_agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)

        elif agent_type == "ppo_gail_hp":
            params, config = load_ppo_dir(PATHS["ppo_gail"], "ppo_gail", layout, seed, verbose=False)
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
    layouts: Sequence[str],
    seeds: Sequence[int],
    num_games_per_seed: int = 5,
    include_gail: bool = True,
    include_ppo_gail: bool = True,
    verbose: bool = True,
):
    """Run full evaluation for all layouts."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for layout in layouts:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layout: {layout}")
            print(f"{'='*60}")

        env_layout = mu.LAYOUT_TO_ENV.get(layout, layout)
        layout_results: Dict[str, Dict[str, float]] = {}

        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )

        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)

        bc_model, hp_model = load_bc_and_hp(layout, verbose=verbose)
        if hp_model is None:
            if verbose:
                print("  ✗ Skipping layout due to missing HP model")
            continue

        gail_model = load_gail(layout, featurize_fn, verbose=verbose) if include_gail else None

        if verbose:
            print("\n  Evaluating SP+SP...")
        layout_results["sp_sp"] = evaluate_all_seeds(
            layout,
            "sp_sp",
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=False,
            verbose=verbose,
            gail_model=gail_model,
            seeds=seeds,
        )

        if verbose:
            print("\n  Evaluating SP+HP...")
        layout_results["sp_hp"] = evaluate_all_seeds(
            layout,
            "sp_hp",
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=False,
            verbose=verbose,
            gail_model=gail_model,
            seeds=seeds,
        )
        layout_results["sp_hp_swapped"] = evaluate_all_seeds(
            layout,
            "sp_hp",
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=True,
            verbose=verbose,
            gail_model=gail_model,
            seeds=seeds,
        )

        if verbose:
            print("\n  Evaluating PPO_BC+HP...")
        layout_results["ppo_bc_hp"] = evaluate_all_seeds(
            layout,
            "ppo_bc_hp",
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=False,
            verbose=verbose,
            gail_model=gail_model,
            seeds=seeds,
        )
        layout_results["ppo_bc_hp_swapped"] = evaluate_all_seeds(
            layout,
            "ppo_bc_hp",
            hp_model,
            bc_model,
            featurize_fn,
            num_games_per_seed,
            swapped=True,
            verbose=verbose,
            gail_model=gail_model,
            seeds=seeds,
        )

        if include_ppo_gail:
            if verbose:
                print("\n  Evaluating PPO_GAIL+HP...")
            layout_results["ppo_gail_hp"] = evaluate_all_seeds(
                layout,
                "ppo_gail_hp",
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed,
                swapped=False,
                verbose=verbose,
                gail_model=gail_model,
                seeds=seeds,
            )
            layout_results["ppo_gail_hp_swapped"] = evaluate_all_seeds(
                layout,
                "ppo_gail_hp",
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed,
                swapped=True,
                verbose=verbose,
                gail_model=gail_model,
                seeds=seeds,
            )

        if bc_model is not None:
            if verbose:
                print("\n  Evaluating BC+HP...")
            layout_results["bc_hp"] = evaluate_all_seeds(
                layout,
                "bc_hp",
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=False,
                verbose=verbose,
                gail_model=gail_model,
                seeds=seeds,
            )
            layout_results["bc_hp_swapped"] = evaluate_all_seeds(
                layout,
                "bc_hp",
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=True,
                verbose=verbose,
                gail_model=gail_model,
                seeds=seeds,
            )

        if include_gail and gail_model is not None:
            if verbose:
                print("\n  Evaluating GAIL+HP...")
            layout_results["gail_hp"] = evaluate_all_seeds(
                layout,
                "gail_hp",
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=False,
                verbose=verbose,
                gail_model=gail_model,
                seeds=seeds,
            )
            layout_results["gail_hp_swapped"] = evaluate_all_seeds(
                layout,
                "gail_hp",
                hp_model,
                bc_model,
                featurize_fn,
                num_games_per_seed * 5,
                swapped=True,
                verbose=verbose,
                gail_model=gail_model,
                seeds=seeds,
            )

        results[layout] = layout_results

    return results


def print_table(results: Dict):
    """Print results in a concise table."""
    print("\n" + "=" * 140)
    print("HPC Evaluation Results")
    print("=" * 140)
    header = (
        f"{'Layout':<22} {'SP+SP':<12} {'SP+HP':<12} "
        f"{'PPO_BC+HP':<12} {'PPO_GAIL+HP':<12} {'GAIL+HP':<12} {'BC+HP':<12}"
    )
    print(header)
    print("-" * 140)

    for layout in mu.LAYOUTS:
        if layout not in results:
            continue
        r = results[layout]

        sp_sp = f"{r['sp_sp']['mean']:.1f}±{r['sp_sp']['se']:.1f}" if "sp_sp" in r else "N/A"
        sp_hp = f"{r['sp_hp']['mean']:.1f}±{r['sp_hp']['se']:.1f}" if "sp_hp" in r else "N/A"
        ppo_bc = f"{r['ppo_bc_hp']['mean']:.1f}±{r['ppo_bc_hp']['se']:.1f}" if "ppo_bc_hp" in r else "N/A"
        ppo_gail = f"{r['ppo_gail_hp']['mean']:.1f}±{r['ppo_gail_hp']['se']:.1f}" if "ppo_gail_hp" in r else "N/A"
        gail_hp = f"{r.get('gail_hp', {}).get('mean', 0):.1f}±{r.get('gail_hp', {}).get('se', 0):.1f}" if "gail_hp" in r else "N/A"
        bc_hp = f"{r.get('bc_hp', {}).get('mean', 0):.1f}±{r.get('bc_hp', {}).get('se', 0):.1f}" if "bc_hp" in r else "N/A"

        print(f"{layout:<22} {sp_sp:<12} {sp_hp:<12} {ppo_bc:<12} {ppo_gail:<12} {gail_hp:<12} {bc_hp:<12}")

    print("=" * 140)


def parse_seeds(raw_seeds: str) -> List[int]:
    """Parse comma-separated seeds string."""
    return [int(seed.strip()) for seed in raw_seeds.split(",") if seed.strip()]


def main():
    global BASE_MODEL_DIR, PATHS
    
    parser = argparse.ArgumentParser(description="Evaluate HPC-trained models with Human Proxy")
    parser.add_argument("--num_games", type=int, default=5, help="Games per seed")
    parser.add_argument("--layouts", nargs="*", default=mu.LAYOUTS, help="Layouts to evaluate")
    parser.add_argument("--seeds", type=str, default="0,10,20,30,40", help="Comma-separated seeds")
    parser.add_argument("--output", type=str, default="hpc_eval_results.json", help="Results JSON output path")
    parser.add_argument("--no_gail", action="store_true", help="Skip GAIL baseline evaluation")
    parser.add_argument("--no_ppo_gail", action="store_true", help="Skip PPO_GAIL evaluation")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base model directory (overrides auto-detected path)")
    args = parser.parse_args()

    # Allow CLI override of base model directory
    if args.base_dir:
        BASE_MODEL_DIR = os.path.expanduser(args.base_dir)
        PATHS = {
            "bc_train": os.path.join(BASE_MODEL_DIR, "bc_runs", "train"),
            "bc_test": os.path.join(BASE_MODEL_DIR, "bc_runs", "test"),
            "gail": os.path.join(BASE_MODEL_DIR, "gail_runs"),
            "ppo_sp": os.path.join(BASE_MODEL_DIR, "results", "ppo_sp"),
            "ppo_bc": os.path.join(BASE_MODEL_DIR, "results", "ppo_bc"),
            "ppo_gail": os.path.join(BASE_MODEL_DIR, "results", "ppo_gail"),
        }

    seeds = parse_seeds(args.seeds)
    verbose = not args.quiet

    print("=" * 60)
    print("HPC Model Evaluation")
    print("=" * 60)
    print(f"Base model dir: {BASE_MODEL_DIR}")
    print(f"Layouts: {args.layouts}")
    print(f"Seeds: {seeds}")
    print(f"Games per seed: {args.num_games}")

    results = run_full_evaluation(
        layouts=args.layouts,
        seeds=seeds,
        num_games_per_seed=args.num_games,
        include_gail=not args.no_gail,
        include_ppo_gail=not args.no_ppo_gail,
        verbose=verbose,
    )

    print_table(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
