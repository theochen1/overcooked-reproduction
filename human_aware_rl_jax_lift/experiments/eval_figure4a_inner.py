#!/usr/bin/env python
"""
Evaluate all Figure-4a conditions for ONE layout inside the Docker container.

Must be run from /code/human_aware_rl (the working directory set by run_eval_figure4a.sh).

Usage (inside container):
    python eval_figure4a_inner.py --layout simple --num_games 100 --out_dir /results

Writes: {out_dir}/results_{layout}.json  →  {fig_key: {condition: {seed_idx: reward}}}
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf

from ppo.ppo import get_ppo_agent                           # noqa: E402
from overcooked_ai_py.agents.agent import AgentPair         # noqa: E402
from overcooked_ai_py.agents.benchmarking import AgentEvaluator  # noqa: E402

# BC import – the class name differs between repo versions; try both.
try:
    from imitation.behavioral_cloning import BehavioralCloningPolicy
except ImportError:
    try:
        from behavioral_cloning import BehavioralCloningPolicy
    except ImportError as exc:
        raise ImportError(
            "Could not import BehavioralCloningPolicy. "
            "Check that the imitation module is on sys.path inside the container."
        ) from exc

# ── Seed tables (must match ground_truth_runs directory names) ────────────────
PPO_SP_SEEDS  = [2229, 386, 7225, 7649, 9807]
PPO_BCT_SEEDS = [184,  2888, 4467, 7360, 7424]   # ppo_bc_test
PPO_BCR_SEEDS = [1887, 516,  5578, 5987, 9456]   # ppo_bc_train (gold std)

# BC models live at /code/human_aware_rl/data/bc_runs/{layout}_{split}/
# (container mount: host overcooked-reproduction/human_aware_rl → /code)
BC_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "bc_runs"
)

LAYOUT_MAP = {
    "simple":    "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random0":   "coordination_ring",
    "random3":   "forced_coordination",
    "random1":   "counter_circuit",
}


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_ppo(run_name: str, seed: int):
    """Load best PPO checkpoint, resetting the TF1 graph between calls."""
    tf.reset_default_graph()
    return get_ppo_agent(run_name, seed=seed, best=True)


def load_bc(layout: str, split: str):
    """Load a BC agent from data/bc_runs/{layout}_{split}/."""
    bc_dir = os.path.normpath(os.path.join(BC_DATA_DIR, f"{layout}_{split}"))
    if not os.path.isdir(bc_dir):
        raise FileNotFoundError(
            f"BC model directory not found: {bc_dir}\n"
            "Ensure BC_DATA_DIR is correct and the model has been trained."
        )
    return BehavioralCloningPolicy.from_pickle(bc_dir, stochastic=True)


# ── Evaluator factory ─────────────────────────────────────────────────────────

def make_evaluator(config: dict) -> AgentEvaluator:
    env_params = dict(config["env_params"])
    env_params["horizon"] = 400   # paper evaluation horizon
    return AgentEvaluator(
        mdp_params=config["mdp_params"],
        env_params=env_params,
    )


def eval_pair(
    agent0,
    agent1,
    config: dict,
    num_games: int,
    switched: bool = False,
    self_play: bool = False,
) -> float:
    """Return mean cumulative reward over num_games episodes.

    Args:
        agent0:    The agent under evaluation.
        agent1:    The partner (HProxy, or same agent for self-play).
        config:    Dict with 'mdp_params' and 'env_params' (taken from a PPO run).
        num_games: Number of rollout episodes.
        switched:  If True, agent0 plays as player 1 (position swap).
        self_play: If True, creates AgentPair(agent0, agent0) for SP+SP.
    """
    evaluator = make_evaluator(config)
    if self_play:
        pair = AgentPair(agent0, agent0, allow_duplicate_agents=True)
    elif switched:
        pair = AgentPair(agent1, agent0)
    else:
        pair = AgentPair(agent0, agent1)
    result = evaluator.evaluate_agent_pair(pair, num_games=num_games)
    return float(np.mean(result["ep_returns"]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Figure-4a conditions for one layout (runs inside Docker)."
    )
    parser.add_argument(
        "--layout", required=True, choices=list(LAYOUT_MAP.keys()),
        help="Short layout name: simple | unident_s | random0 | random1 | random3",
    )
    parser.add_argument(
        "--num_games", type=int, default=100,
        help="Rollout episodes per (condition, seed) pair (default: 100).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=".",
        help="Directory to write results_{layout}.json (default: cwd).",
    )
    args = parser.parse_args()

    layout  = args.layout
    fig_key = LAYOUT_MAP[layout]

    out: dict = {
        "SP_SP":           {},
        "SP_HProxy":       {},
        "PPOBC_HProxy":    {},
        "BC_HProxy":       {},
        "SP_HProxy_sw":    {},
        "PPOBC_HProxy_sw": {},
        "BC_HProxy_sw":    {},
        "gold_standard":   None,
    }

    print(f"\n{'='*60}")
    print(f"  Layout : {layout}  →  {fig_key}")
    print(f"  Games  : {args.num_games} per condition")
    print(f"{'='*60}")

    # Load shared models once
    print("  Loading HProxy (bc_test) ...")
    hproxy = load_bc(layout, "test")
    print("  Loading BC_train acting agent ...")
    bc_train = load_bc(layout, "train")

    sp_config = None  # will be populated on first PPO_SP seed

    # ── PPO_SP × 5 seeds ───────────────────────────────────────────────────────
    for i, seed in enumerate(PPO_SP_SEEDS):
        print(f"  [PPO_SP] seed={seed} (idx {i}) ...")
        agent, config = load_ppo(f"ppo_sp_{layout}", seed)
        if sp_config is None:
            sp_config = config

        out["SP_SP"][i]        = eval_pair(agent, agent,  config, args.num_games, self_play=True)
        out["SP_HProxy"][i]    = eval_pair(agent, hproxy, config, args.num_games)
        out["SP_HProxy_sw"][i] = eval_pair(agent, hproxy, config, args.num_games, switched=True)
        print(f"    SP_SP={out['SP_SP'][i]:.1f}  "
              f"SP_HP={out['SP_HProxy'][i]:.1f}  "
              f"SP_HP_sw={out['SP_HProxy_sw'][i]:.1f}")

    # ── PPO_BC_test × 5 seeds ──────────────────────────────────────────────────
    for i, seed in enumerate(PPO_BCT_SEEDS):
        print(f"  [PPO_BC_test] seed={seed} (idx {i}) ...")
        agent, config = load_ppo(f"ppo_bc_test_{layout}", seed)
        out["PPOBC_HProxy"][i]    = eval_pair(agent, hproxy, config, args.num_games)
        out["PPOBC_HProxy_sw"][i] = eval_pair(agent, hproxy, config, args.num_games, switched=True)
        print(f"    PPOBC_HP={out['PPOBC_HProxy'][i]:.1f}  "
              f"PPOBC_HP_sw={out['PPOBC_HProxy_sw'][i]:.1f}")

    # ── BC_train as acting agent ───────────────────────────────────────────────
    # BC is deterministic given fixed env randomness, so we run once and
    # replicate across the 5 seed index slots.
    assert sp_config is not None, "sp_config is None – did PPO_SP seeds load?"
    print("  [BC_train vs HProxy] evaluating ...")
    bc_mean    = eval_pair(bc_train, hproxy, sp_config, args.num_games)
    bc_mean_sw = eval_pair(bc_train, hproxy, sp_config, args.num_games, switched=True)
    for i in range(5):
        out["BC_HProxy"][i]    = bc_mean
        out["BC_HProxy_sw"][i] = bc_mean_sw
    print(f"    BC_HP={bc_mean:.1f}  BC_HP_sw={bc_mean_sw:.1f}")

    # ── Gold standard: PPO_BC_train seed 0 vs HProxy ──────────────────────────
    gs_seed = PPO_BCR_SEEDS[0]
    print(f"  [Gold standard] ppo_bc_train seed={gs_seed} ...")
    gs_agent, gs_config = load_ppo(f"ppo_bc_train_{layout}", gs_seed)
    out["gold_standard"] = eval_pair(gs_agent, hproxy, gs_config, args.num_games)
    print(f"    gold_standard={out['gold_standard']:.1f}")

    # ── Save fragment ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"results_{layout}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({fig_key: out}, f, indent=2)
    print(f"\n  ✓ Written → {out_path}")


if __name__ == "__main__":
    main()
