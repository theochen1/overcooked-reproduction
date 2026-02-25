"""
Run all Figure 4a evaluations inside the legacy TF1 environment.

Evaluates 7 conditions x 5 layouts x 5 seeds and writes results_figure4a.json
in the format expected by figure4a.py.

Must be run from /code/human_aware_rl with PYTHONPATH set for overcooked_ai_py.
"""

import copy
import json
import os
import sys

import numpy as np
import tensorflow as tf

from ppo.ppo import get_ppo_agent, PPO_DATA_DIR
from human_aware_rl.imitation.behavioural_cloning import (
    get_bc_agent_from_saved,
    load_bc_model_from_path,
    get_bc_agent_from_model,
    BC_SAVE_DIR,
)
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

# Monkey-patch ExpertDataset.__setstate__ to tolerate version mismatches.
try:
    from stable_baselines.gail.dataset.dataset import ExpertDataset
    _orig_setstate = ExpertDataset.__setstate__

    def _permissive_setstate(self, state):
        try:
            _orig_setstate(self, state)
        except AssertionError:
            self.__dict__.update(state)

    ExpertDataset.__setstate__ = _permissive_setstate
except Exception:
    pass

NUM_GAMES = int(os.environ.get("NUM_GAMES", "100"))
OUT_PATH = os.environ.get("OUT_PATH", "/results/results_figure4a.json")

LAYOUTS = ["simple", "unident_s", "random0", "random1", "random3"]
LAYOUT_MAP = {
    "simple": "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random0": "coordination_ring",
    "random3": "forced_coordination",
    "random1": "counter_circuit",
}

PPO_SP_SEEDS = [2229, 386, 7225, 7649, 9807]
PPO_BCT_SEEDS = [184, 2888, 4467, 7360, 7424]
PPO_BCR_SEEDS = [1887, 516, 5578, 5987, 9456]

GT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ground_truth_runs")


def ensure_bc_model_symlinks():
    """GAIL.load expects 'model' (no ext) but files are 'model.pkl'.
    Create 'model' -> 'model.pkl' symlinks where needed."""
    bc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bc_runs")
    if not os.path.isdir(bc_dir):
        return
    for entry in os.listdir(bc_dir):
        subdir = os.path.join(bc_dir, entry)
        if not os.path.isdir(subdir):
            continue
        model_pkl = os.path.join(subdir, "model.pkl")
        model_bare = os.path.join(subdir, "model")
        if os.path.exists(model_pkl) and not os.path.exists(model_bare):
            os.symlink("model.pkl", model_bare)


def ensure_ppo_symlinks():
    """Create symlinks from data/ppo_runs/{name} -> ground_truth_runs/{name}
    so that get_ppo_agent can find the checkpoints."""
    os.makedirs(PPO_DATA_DIR, exist_ok=True)
    for entry in os.listdir(GT_DIR):
        src = os.path.join(GT_DIR, entry)
        dst = os.path.join(PPO_DATA_DIR, entry)
        if os.path.isdir(src) and not os.path.exists(dst) and not os.path.islink(dst):
            os.symlink(os.path.abspath(src), dst)


def make_evaluator(config):
    env_params = dict(config["env_params"])
    env_params["horizon"] = 400
    return AgentEvaluator(
        mdp_params=config["mdp_params"],
        env_params=env_params,
    )


def eval_pair(agent0, agent1, config, num_games, switched=False, self_play=False):
    evaluator = make_evaluator(config)
    if self_play:
        pair = AgentPair(agent0, agent0, allow_duplicate_agents=True)
    elif switched:
        pair = AgentPair(agent1, agent0)
    else:
        pair = AgentPair(agent0, agent1)
    result = evaluator.evaluate_agent_pair(pair, num_games=num_games)
    return float(np.mean(result["ep_returns"]))


def load_best_bc(layout, split):
    """Load the best BC model for a layout/split.

    best_bc_model_paths.pickle maps split -> layout -> model_name
    (e.g. "simple_bc_test_seed2"). Fallback to seed0 if missing.
    """
    try:
        from overcooked_ai_py.utils import load_pickle
        best_paths = load_pickle(BC_SAVE_DIR + "best_bc_model_paths")
        model_name = best_paths[split][layout]
    except Exception:
        model_name = "{}_bc_{}_seed0".format(layout, split)

    agent, bc_params = get_bc_agent_from_saved(model_name, no_waits=False)
    return agent, bc_params


def main():
    ensure_ppo_symlinks()
    ensure_bc_model_symlinks()

    results = {}

    for layout in LAYOUTS:
        fig_key = LAYOUT_MAP[layout]
        print("\n" + "=" * 60)
        print("  Layout: {} -> {}".format(layout, fig_key))
        print("  Games per eval: {}".format(NUM_GAMES))
        print("=" * 60)

        out = {
            "SP_SP": {},
            "SP_HProxy": {},
            "PPOBC_HProxy": {},
            "BC_HProxy": {},
            "SP_HProxy_sw": {},
            "PPOBC_HProxy_sw": {},
            "BC_HProxy_sw": {},
            "gold_standard": None,
        }

        # Load BC agents
        print("  Loading HProxy (bc_test) ...")
        hproxy, _ = load_best_bc(layout, "test")
        print("  Loading BC_train ...")
        bc_train, _ = load_best_bc(layout, "train")

        sp_config = None

        # PPO_SP x 5 seeds
        for i, seed in enumerate(PPO_SP_SEEDS):
            print("  [PPO_SP] seed={} (idx {}) ...".format(seed, i))
            tf.reset_default_graph()
            agent, config = get_ppo_agent("ppo_sp_{}".format(layout), seed=seed, best=True)
            if sp_config is None:
                sp_config = config

            out["SP_SP"][i] = eval_pair(agent, agent, config, NUM_GAMES, self_play=True)
            out["SP_HProxy"][i] = eval_pair(agent, hproxy, config, NUM_GAMES)
            out["SP_HProxy_sw"][i] = eval_pair(agent, hproxy, config, NUM_GAMES, switched=True)
            print("    SP_SP={:.1f}  SP_HP={:.1f}  SP_HP_sw={:.1f}".format(
                out["SP_SP"][i], out["SP_HProxy"][i], out["SP_HProxy_sw"][i]))

        # PPO_BC_test x 5 seeds
        for i, seed in enumerate(PPO_BCT_SEEDS):
            print("  [PPO_BC_test] seed={} (idx {}) ...".format(seed, i))
            tf.reset_default_graph()
            agent, config = get_ppo_agent("ppo_bc_test_{}".format(layout), seed=seed, best=True)
            out["PPOBC_HProxy"][i] = eval_pair(agent, hproxy, config, NUM_GAMES)
            out["PPOBC_HProxy_sw"][i] = eval_pair(agent, hproxy, config, NUM_GAMES, switched=True)
            print("    PPOBC_HP={:.1f}  PPOBC_HP_sw={:.1f}".format(
                out["PPOBC_HProxy"][i], out["PPOBC_HProxy_sw"][i]))

        # BC_train as acting agent (no per-seed variance; replicate across slots)
        assert sp_config is not None
        print("  [BC_train vs HProxy] ...")
        bc_mean = eval_pair(bc_train, hproxy, sp_config, NUM_GAMES)
        bc_mean_sw = eval_pair(bc_train, hproxy, sp_config, NUM_GAMES, switched=True)
        for i in range(5):
            out["BC_HProxy"][i] = bc_mean
            out["BC_HProxy_sw"][i] = bc_mean_sw
        print("    BC_HP={:.1f}  BC_HP_sw={:.1f}".format(bc_mean, bc_mean_sw))

        # Gold standard: PPO_BC_train seed 0
        gs_seed = PPO_BCR_SEEDS[0]
        print("  [Gold standard] ppo_bc_train seed={} ...".format(gs_seed))
        tf.reset_default_graph()
        gs_agent, gs_config = get_ppo_agent("ppo_bc_train_{}".format(layout), seed=gs_seed, best=True)
        out["gold_standard"] = eval_pair(gs_agent, hproxy, gs_config, NUM_GAMES)
        print("    gold_standard={:.1f}".format(out["gold_standard"]))

        results[fig_key] = out

    # Write results
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults written to: {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
