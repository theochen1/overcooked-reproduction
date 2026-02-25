"""Build results JSON for figure4a.py from ground_truth_runs checkpoints.

Usage:
    python prepare_results.py
    # → writes results_figure4a.json in the current directory

Then pass that file to the plotting script:
    python figure4a.py --results_path results_figure4a.json --output figure_4a.png
"""

import json
import pickle
import sys
from pathlib import Path

# ── Layout name mapping: run-dir name → figure4a.py LAYOUT_ORDER key ─────────
LAYOUT_MAP = {
    "simple":    "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random0":   "coordination_ring",
    "random3":   "forced_coordination",
    "random1":   "counter_circuit",
}

# Actual seed values per model family, ordered → integer indices 0-4
PPO_SP_SEEDS  = [2229, 386, 7225, 7649, 9807]
PPO_BCT_SEEDS = [184, 2888, 4467, 7360, 7424]   # ppo_bc_test
PPO_BCR_SEEDS = [1887, 516, 5578, 5987, 9456]   # ppo_bc_train

RUNS_ROOT = Path(__file__).resolve().parents[2] / "human_aware_rl" / "ground_truth_runs"
BC_RUNS_ROOT = Path(__file__).resolve().parents[2] / "human_aware_rl" / "human_aware_rl" / "data" / "bc_runs"

NUM_ROLLOUTS = 100  # matches the paper


# ── Path setup: make original human_aware_rl importable ──────────────────────
_HAR_ROOT = Path(__file__).resolve().parents[2] / "human_aware_rl"
for _p in [str(_HAR_ROOT), str(_HAR_ROOT / "human_aware_rl")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from human_aware_rl.ppo.ppo import get_ppo_agent                       # noqa: E402
from human_aware_rl.imitation.behavior_cloning import BehaviorCloningPolicy  # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld    # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv          # noqa: E402
from overcooked_ai_py.agents.benchmarking import AgentEvaluator        # noqa: E402


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_ppo_agent(run_dir: Path, seed: int):
    """Load the best PPO checkpoint for a given run directory and seed."""
    seed_dir = run_dir / f"seed{seed}"
    agent_dir = seed_dir / "best" / "ppo_agent"
    config_path = seed_dir / "config.pickle"
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return get_ppo_agent(str(agent_dir), config)


def load_bc_agent(layout_name: str, split: str):
    """Load a BC model.

    Args:
        layout_name: short layout key, e.g. 'simple', 'random0'.
        split: 'train' or 'test'.
    """
    bc_dir = BC_RUNS_ROOT / f"{layout_name}_{split}"
    if not bc_dir.exists():
        raise FileNotFoundError(
            f"BC model not found at {bc_dir}. "
            "Check that BC_RUNS_ROOT is correct and the directory exists."
        )
    with open(bc_dir / "config.pickle", "rb") as f:
        config = pickle.load(f)
    return BehaviorCloningPolicy.from_pickle(str(bc_dir), config)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_pair(
    agent0,
    agent1,
    layout_name: str,
    num_rollouts: int,
    switched: bool = False,
) -> float:
    """Run num_rollouts episodes and return mean cumulative reward.

    Args:
        agent0: the agent under evaluation (player 0 when switched=False).
        agent1: the partner agent (player 1 when switched=False).
        layout_name: short layout key used by OvercookedGridworld.
        num_rollouts: number of episodes to average over.
        switched: if True, swap player indices (agent0 plays as player 1).
    """
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=400)
    evaluator = AgentEvaluator.from_env(env)
    if switched:
        ap = evaluator.get_agent_pair(agent1, agent0)
    else:
        ap = evaluator.get_agent_pair(agent0, agent1)
    results = evaluator.evaluate_agent_pair(ap, num_games=num_rollouts)
    return float(results["ep_returns_mean"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    out: dict = {}

    for dir_layout, fig_layout in LAYOUT_MAP.items():
        print(f"\n{'='*60}")
        print(f"Layout: {dir_layout}  →  {fig_layout}")
        print("=" * 60)

        out[fig_layout] = {
            "SP_SP":           {},
            "SP_HProxy":       {},
            "PPOBC_HProxy":    {},
            "BC_HProxy":       {},
            "SP_HProxy_sw":    {},
            "PPOBC_HProxy_sw": {},
            "BC_HProxy_sw":    {},
            "gold_standard":   None,
        }

        # HProxy = BC_test (held-out human proxy used at evaluation time)
        print("  Loading HProxy (bc_test)...")
        hproxy = load_bc_agent(dir_layout, "test")

        # BC_train used as the acting agent in the BC_HProxy condition
        print("  Loading BC_train agent...")
        bc_train = load_bc_agent(dir_layout, "train")

        # ── PPO_SP: 5 seeds ───────────────────────────────────────────────
        for i, seed in enumerate(PPO_SP_SEEDS):
            run_dir = RUNS_ROOT / f"ppo_sp_{dir_layout}_seed{seed}"
            print(f"  [PPO_SP] seed={seed} (index {i})...")
            agent = load_ppo_agent(run_dir, seed)

            out[fig_layout]["SP_SP"][i]        = evaluate_pair(agent, agent,  dir_layout, NUM_ROLLOUTS, switched=False)
            out[fig_layout]["SP_HProxy"][i]    = evaluate_pair(agent, hproxy, dir_layout, NUM_ROLLOUTS, switched=False)
            out[fig_layout]["SP_HProxy_sw"][i] = evaluate_pair(agent, hproxy, dir_layout, NUM_ROLLOUTS, switched=True)

        # ── PPO_BC_test: 5 seeds ──────────────────────────────────────────
        for i, seed in enumerate(PPO_BCT_SEEDS):
            run_dir = RUNS_ROOT / f"ppo_bc_test_{dir_layout}_seed{seed}"
            print(f"  [PPO_BC_test] seed={seed} (index {i})...")
            agent = load_ppo_agent(run_dir, seed)

            out[fig_layout]["PPOBC_HProxy"][i]    = evaluate_pair(agent, hproxy, dir_layout, NUM_ROLLOUTS, switched=False)
            out[fig_layout]["PPOBC_HProxy_sw"][i] = evaluate_pair(agent, hproxy, dir_layout, NUM_ROLLOUTS, switched=True)

        # ── BC_train as agent paired with HProxy: replicated 5 times ─────
        # BC is deterministic (given the same env seed), so we run once and
        # replicate across the 5 index slots to match the expected seed-dict format.
        print("  [BC_train] evaluating vs HProxy...")
        bc_mean        = evaluate_pair(bc_train, hproxy, dir_layout, NUM_ROLLOUTS, switched=False)
        bc_mean_sw     = evaluate_pair(bc_train, hproxy, dir_layout, NUM_ROLLOUTS, switched=True)
        for i in range(5):
            out[fig_layout]["BC_HProxy"][i]    = bc_mean
            out[fig_layout]["BC_HProxy_sw"][i] = bc_mean_sw

        # ── Gold standard: PPO_BC_train seed 0 paired with HProxy ────────
        # PPO_BC_train was trained against bc_train, so testing it against
        # bc_test (HProxy) gives the gold-standard upper-bound estimate.
        gs_seed = PPO_BCR_SEEDS[0]
        gs_dir  = RUNS_ROOT / f"ppo_bc_train_{dir_layout}_seed{gs_seed}"
        print(f"  [Gold standard] ppo_bc_train seed={gs_seed}...")
        gs_agent = load_ppo_agent(gs_dir, gs_seed)
        out[fig_layout]["gold_standard"] = evaluate_pair(
            gs_agent, hproxy, dir_layout, NUM_ROLLOUTS, switched=False
        )

        print(f"  ✓ Done: {fig_layout}")

    out_path = Path(__file__).parent / "results_figure4a.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Results saved to: {out_path}")
    print("\nNext step:")
    print("  python figure4a.py --results_path results_figure4a.json --output figure_4a.png")


if __name__ == "__main__":
    main()
