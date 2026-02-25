#!/usr/bin/env python
"""Reproduce Figure 4a from Carroll et al. NeurIPS 2019."""

import argparse
import os
from collections import defaultdict

import numpy as np

from overcooked_ai_py.utils import save_pickle, load_pickle, mean_and_std_err
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.utils import reset_tf, set_global_seed
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved, BC_SAVE_DIR
from human_aware_rl.ppo.ppo import get_ppo_agent, PPO_DATA_DIR
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH
from human_aware_rl.experiments.graphing import get_algorithm_color, get_texture


LAYOUTS = ["simple", "unident_s", "random1", "random0", "random3"]
LAYOUT_DISPLAY_NAMES = {
    "simple": "Cramped Room",
    "unident_s": "Asymmetric Advantages",
    "random1": "Coordination Ring",
    "random0": "Forced Coordination",
    "random3": "Counter Circuit",
}

# Seeds exactly as used in each training script
PPO_SP_SEEDS = [2229, 7649, 7225, 9807, 386]  # from ppo_sp_experiments.sh
PPO_BC_TRAIN_SEEDS = [9456, 1887, 5578, 5987, 516]  # from ppo_bc_experiments.sh
PPO_BC_TEST_SEEDS = [2888, 7424, 7360, 4467, 184]  # from ppo_bc_experiments.sh

NUM_ROUNDS = 100  # paper uses 100 rollouts per seed
RESULTS_SAVE_PATH = PPO_DATA_DIR + "figure4_results"
FIGURE_SAVE_PATH = "figure4_reproduction.png"


def get_best_bc_model_paths():
    """
    Returns dict:
      {"train": {layout: model_name}, "test": {layout: model_name}}

    model_name is the string passed to get_bc_agent_from_saved(), which
    prepends BC_SAVE_DIR to form the full path.

    If bc_experiments.py's run_all_bc_experiments() was completed, the
    pickle at BEST_BC_MODELS_PATH already exists. Otherwise fall back to
    the original paper's manual selection indices — adjust these if
    YOUR seed ordering differs.
    """
    if os.path.exists(BEST_BC_MODELS_PATH):
        return load_pickle(BEST_BC_MODELS_PATH)
    if os.path.exists(BC_SAVE_DIR + "best_bc_model_paths"):
        return load_pickle(BC_SAVE_DIR + "best_bc_model_paths")

    # Original paper's selections: {layout: (train_seed_idx, test_seed_idx)}
    # Indices are positions 0-4 in the BC training seeds list [5415,2652,6440,1965,6647]
    selected = {
        "simple": (0, 1),
        "unident_s": (0, 0),
        "random1": (4, 2),
        "random0": (2, 1),
        "random3": (3, 3),
    }
    paths = {"train": {}, "test": {}}
    for layout, (ti, xi) in selected.items():
        paths["train"][layout] = f"{layout}_bc_train_seed{ti}"
        paths["test"][layout] = f"{layout}_bc_test_seed{xi}"
    return paths


def load_ppo_agent_with_best_fallback(ex_name, seed):
    """Load PPO from best checkpoint if available, else from final model."""
    best_dir = f"{PPO_DATA_DIR}{ex_name}/seed{seed}/best"
    use_best = os.path.exists(best_dir)
    try:
        return get_ppo_agent(ex_name, seed, best=use_best)
    except Exception:
        if use_best:
            return get_ppo_agent(ex_name, seed, best=False)
        raise


def evaluate_ppo_sp_for_layout(layout, bc_test_path, seeds=PPO_SP_SEEDS, num_rounds=NUM_ROUNDS):
    """
    Returns dict: {algorithm_name: [per_seed_mean, ...]}
    The list has len(seeds) entries.
    """
    reset_tf()
    _, bc_params = get_bc_agent_from_saved(bc_test_path)
    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    sp_results = defaultdict(list)
    for seed in seeds:
        reset_tf()  # MUST reset between seeds — TF1 default graph/session reuse.
        agent_ppo, _ = load_ppo_agent_with_best_fallback(f"ppo_sp_{layout}", seed)
        # Re-load BC agent after reset_tf.
        agent_bc_test, _ = get_bc_agent_from_saved(bc_test_path)

        # White bar: PPO_SP paired with itself.
        sp_sp = evaluator.evaluate_agent_pair(
            AgentPair(agent_ppo, agent_ppo, allow_duplicate_agents=True),
            num_games=num_rounds,
        )
        sp_results["PPO_SP+PPO_SP"].append(np.mean(sp_sp["ep_returns"]))

        # Dark teal bars: SP paired with BC_test in both player positions.
        ppo_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo, agent_bc_test), num_games=num_rounds)
        sp_results["PPO_SP+BC_test_0"].append(np.mean(ppo_bc["ep_returns"]))

        bc_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo), num_games=num_rounds)
        sp_results["PPO_SP+BC_test_1"].append(np.mean(bc_ppo["ep_returns"]))

    return dict(sp_results)


def evaluate_bc_for_layout(layout, bc_train_path, bc_test_path, num_rounds=NUM_ROUNDS):
    """
    Returns dict: {algorithm_name: (mean, stderr)}
    Note: BC has no "seeds" — there is exactly one BC_train and one BC_test
    model selected per layout. The evaluator runs num_rounds games total.
    """
    _ = layout  # Maintained for API symmetry with other evaluators.
    reset_tf()
    agent_bc_train, bc_train_params = get_bc_agent_from_saved(bc_train_path)
    agent_bc_test, _ = get_bc_agent_from_saved(bc_test_path)

    evaluator = AgentEvaluator(
        mdp_params=bc_train_params["mdp_params"],
        env_params=bc_train_params["env_params"],
    )

    traj0 = evaluator.evaluate_agent_pair(AgentPair(agent_bc_train, agent_bc_test), num_games=num_rounds)
    traj1 = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_bc_train), num_games=num_rounds)

    return {
        "BC_train+BC_test_0": mean_and_std_err(traj0["ep_returns"]),
        "BC_train+BC_test_1": mean_and_std_err(traj1["ep_returns"]),
    }


def evaluate_ppo_bc_for_layout(
    layout,
    bc_test_path,
    bc_train_seeds=PPO_BC_TRAIN_SEEDS,
    bc_test_seeds=PPO_BC_TEST_SEEDS,
    num_rounds=NUM_ROUNDS,
):
    """
    Returns dict: {algorithm_name: [per_seed_mean, ...]}
    """
    assert len(bc_train_seeds) == len(bc_test_seeds), "Seed lists must be same length"
    results = defaultdict(list)

    for seed_idx in range(len(bc_train_seeds)):
        reset_tf()
        agent_bc_test, bc_params = get_bc_agent_from_saved(bc_test_path)
        evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

        # Orange bars: PPO trained with BC_train, evaluated vs BC_test.
        agent_ppo_bc_train, _ = load_ppo_agent_with_best_fallback(
            f"ppo_bc_train_{layout}", bc_train_seeds[seed_idx]
        )
        traj0 = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=num_rounds)
        results["PPO_BC_train+BC_test_0"].append(np.mean(traj0["ep_returns"]))

        traj1 = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_train), num_games=num_rounds)
        results["PPO_BC_train+BC_test_1"].append(np.mean(traj1["ep_returns"]))

        # Gold standard: PPO trained with BC_test (has access to true proxy).
        agent_ppo_bc_test, _ = load_ppo_agent_with_best_fallback(
            f"ppo_bc_test_{layout}", bc_test_seeds[seed_idx]
        )
        traj0_gs = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_test, agent_bc_test), num_games=num_rounds)
        results["PPO_BC_test+BC_test_0"].append(np.mean(traj0_gs["ep_returns"]))

        traj1_gs = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_test), num_games=num_rounds)
        results["PPO_BC_test+BC_test_1"].append(np.mean(traj1_gs["ep_returns"]))

    return dict(results)


def collect_all_results(best_bc_model_paths, layouts):
    """
    Returns: all_results[layout][algorithm] = list of per-seed means,
    except BC entries which are already (mean, stderr) 2-tuples.
    """
    all_results = {}
    for layout in layouts:
        print("\n" + ("=" * 50))
        print("Evaluating layout: {}".format(layout))
        print("=" * 50)

        bc_train_path = best_bc_model_paths["train"][layout]
        bc_test_path = best_bc_model_paths["test"][layout]

        layout_results = {}
        print("  [1/3] Evaluating PPO_SP...")
        layout_results.update(evaluate_ppo_sp_for_layout(layout, bc_test_path))

        print("  [2/3] Evaluating BC baseline...")
        layout_results.update(evaluate_bc_for_layout(layout, bc_train_path, bc_test_path))

        print("  [3/3] Evaluating PPO_BC (+ gold standard)...")
        layout_results.update(evaluate_ppo_bc_for_layout(layout, bc_test_path))

        all_results[layout] = layout_results
        print("  Done: {}".format(layout))

    return all_results


def aggregate_results(all_results):
    """
    Convert: list of per-seed means -> (mean, stderr) 2-tuple.
    BC entries are already (mean, stderr) — pass through unchanged.
    """
    aggregated = {}
    for layout, algo_dict in all_results.items():
        aggregated[layout] = {}
        for algo, values in algo_dict.items():
            if isinstance(values, (list, np.ndarray)):
                arr = np.array(values)
                aggregated[layout][algo] = (np.mean(arr), np.std(arr) / np.sqrt(len(arr)))
            else:
                aggregated[layout][algo] = values
    return aggregated


def plot_figure4a(aggregated_results, layouts, save_path=FIGURE_SAVE_PATH):
    """
    Reproduce Figure 4a: 1x5 subplots, one per layout.
    Bar order per layout matches paper (left to right):
      [SP+SP_white, SP+BC0_teal, SP+BC1_teal_hatch,
       BC+BC0_gray, BC+BC1_gray_hatch,
       PPOBC+BC0_orange, PPOBC+BC1_orange_hatch]
    Gold standard as red dotted axhline.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required only for plotting. Install it in the container "
            "or run with --no-plot to skip figure generation."
        ) from exc

    bar_algos = [
        "PPO_SP+PPO_SP",
        "PPO_SP+BC_test_0",
        "PPO_SP+BC_test_1",
        "BC_train+BC_test_0",
        "BC_train+BC_test_1",
        "PPO_BC_train+BC_test_0",
        "PPO_BC_train+BC_test_1",
    ]
    gold_algos = ["PPO_BC_test+BC_test_0", "PPO_BC_test+BC_test_1"]

    n_layouts = len(layouts)
    x_positions = np.arange(len(bar_algos))
    fig, axes = plt.subplots(1, n_layouts, figsize=(4 * n_layouts, 5), sharey=False)
    if n_layouts == 1:
        axes = [axes]
    fig.suptitle("Performance with human proxy model (HProxy)", fontsize=13)

    for ax_idx, layout in enumerate(layouts):
        ax = axes[ax_idx]
        layout_data = aggregated_results[layout]

        means = [layout_data[a][0] for a in bar_algos]
        stderrs = [layout_data[a][1] for a in bar_algos]
        colors = [get_algorithm_color(a) for a in bar_algos]
        hatches = [get_texture(a) for a in bar_algos]

        bars = ax.bar(
            x_positions,
            means,
            width=0.9,
            color=colors,
            edgecolor="black",
            linewidth=0.7,
            yerr=stderrs,
            capsize=3,
            error_kw={"elinewidth": 1.0},
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        gold_means = [layout_data[a][0] for a in gold_algos if a in layout_data]
        if gold_means:
            ax.axhline(np.mean(gold_means), color="red", linestyle="--", linewidth=1.5, label="Gold standard")

        ax.set_title(LAYOUT_DISPLAY_NAMES[layout], fontsize=10)
        ax.set_xticks([])
        ax.set_ylim(bottom=0, top=255)
        if ax_idx == 0:
            ax.set_ylabel("Mean episode reward (400 timesteps)", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("Figure saved to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation, load saved results and just replot",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=LAYOUTS,
        help="Subset of layouts to evaluate",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run evaluation and save results, but skip figure generation",
    )
    args = parser.parse_args()

    invalid_layouts = [layout for layout in args.layouts if layout not in LAYOUTS]
    if invalid_layouts:
        raise ValueError("Unknown layouts: {}. Allowed: {}".format(invalid_layouts, LAYOUTS))

    set_global_seed(124)  # matches ppo_sp_experiments.py

    best_bc_model_paths = get_best_bc_model_paths()
    print("Using BC model paths:", best_bc_model_paths)

    if args.skip_eval:
        print("Loading saved results...")
        all_results = load_pickle(RESULTS_SAVE_PATH)
        if set(args.layouts) != set(all_results.keys()):
            all_results = {layout: all_results[layout] for layout in args.layouts}
    else:
        all_results = collect_all_results(best_bc_model_paths, args.layouts)
        save_pickle(all_results, RESULTS_SAVE_PATH)
        print("Raw results saved to {}".format(RESULTS_SAVE_PATH))

    if args.no_plot:
        print("Skipping plot generation (--no-plot).")
    else:
        aggregated = aggregate_results(all_results)
        plot_figure4a(aggregated, args.layouts)
