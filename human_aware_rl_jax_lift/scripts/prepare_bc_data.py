"""
Convert legacy human-human trajectory DataFrames into pre-featurized 64-dim BC datasets
compatible with featurize_state_64 and train_bc.py.

Layout name mapping (DataFrame → engine/.layout file):
  cramped_room       → simple
  asymmetric_adv...  → unident_s
  coordination_ring  → random1
  random0            → random0
  random3            → random3

NOTE on action format:
  Legacy joint actions are tuples of overcooked_ai_py Action values, e.g.
    ((0, -1), 'interact')  or  ((1, 0), (0, 0))
  These are NOT integers.  Use Action.ACTION_TO_INDEX to convert to 0-5.
  This mirrors what joint_state_trajectory_to_single() does internally.

NOTE on trajectory structure from df_traj_to_python_joint_traj:
  ep_observations = np.array([list_of_OvercookedState])  shape (1,), dtype object
  ep_actions      = [list_of_joint_action_tuples]        plain Python list
  Both are already in the "outer wrapper = episodes" format, iterate directly.

NOTE on get_overcooked_traj_for_worker_layout:
  Internally calls PYTHON_LAYOUT_NAME_TO_JS_NAME[layout_name], so MUST receive
  the engine/JAX name (e.g. 'simple'), NOT the DataFrame name ('cramped_room').

Pipeline per timestep (both players):
  DataFrame row
    → get_overcooked_traj_for_worker_layout(df, worker_id, jax_layout)
    → from_legacy_state(terrain, state)         (JAX OvercookedState)
    → featurize_state_64(terrain, jax_state)    (feats_p0, feats_p1), each float32[64]
    → {"features": np.ndarray[N,64], "actions": np.ndarray[N]}

Usage (from repo root):
  python human_aware_rl_jax_lift/scripts/prepare_bc_data.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Layout name mapping ────────────────────────────────────────────────────────
JAX_TO_LEGACY: dict[str, str] = {
    "simple":    "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random1":   "coordination_ring",
    "random0":   "random0",
    "random3":   "random3",
}
LAYOUTS = list(JAX_TO_LEGACY.keys())

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "human_aware_rl"))
sys.path.insert(0, str(REPO_ROOT / "human_aware_rl" / "overcooked_ai"))

_DEFAULT_TRAIN_PKL = (
    REPO_ROOT
    / "human_aware_rl"
    / "human_aware_rl"
    / "data"
    / "human"
    / "anonymized"
    / "clean_train_trials.pkl"
)
_DEFAULT_TEST_PKL = _DEFAULT_TRAIN_PKL.parent / "clean_test_trials.pkl"
_DEFAULT_OUT_DIR  = REPO_ROOT / "human_aware_rl_jax_lift" / "data" / "bc_data"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _legacy_action_to_int(action) -> int:
    """
    Convert one legacy overcooked_ai_py action to its integer index.

    Legacy actions are overcooked_ai_py Action values:
      direction tuples : (0,-1) N, (0,1) S, (1,0) E, (-1,0) W, (0,0) STAY
      string           : 'interact'
    Action.ACTION_TO_INDEX maps all of these to 0-5.
    """
    from overcooked_ai_py.mdp.actions import Action
    # Normalise: numpy strings (np.str_) or lists need to become the exact type
    # that Action.ACTION_TO_INDEX uses as keys.
    if isinstance(action, np.ndarray):
        action = action.item()          # e.g. np.str_('interact') → 'interact'
    if isinstance(action, list):
        action = tuple(action)          # [0, -1] → (0, -1)
    return int(Action.ACTION_TO_INDEX[action])


def _joint_action_to_ints(joint_act) -> tuple[int, int]:
    """Convert a (action_p0, action_p1) legacy joint action to (int, int)."""
    return _legacy_action_to_int(joint_act[0]), _legacy_action_to_int(joint_act[1])


def _featurize_worker(worker_id, df, jax_layout: str, terrain) -> tuple[list, list]:
    """
    Featurize all timesteps for one worker on one layout.

    `jax_layout` is the engine/JAX name (e.g. 'simple').  It is passed directly
    to get_overcooked_traj_for_worker_layout which uses PYTHON_LAYOUT_NAME_TO_JS_NAME
    internally to translate to the DataFrame name.

    Returns (features_list, actions_list): each feature is float32[64], each
    action is an int.  Entries for both players are interleaved: p0 then p1.
    """
    from human_aware_rl.human.process_dataframes import get_overcooked_traj_for_worker_layout
    from human_aware_rl_jax_lift.env.compat import from_legacy_state
    from human_aware_rl_jax_lift.encoding.bc_features import featurize_state_64

    traj, _ = get_overcooked_traj_for_worker_layout(df, worker_id, jax_layout)
    if traj is None:
        return [], []

    # ep_observations = np.array([list_of_OvercookedState])  — iterate as episodes
    # ep_actions      = [list_of_joint_action_tuples]        — already episode-wrapped
    ep_obs  = traj["ep_observations"]
    ep_acts = traj["ep_actions"]

    features: list[np.ndarray] = []
    actions:  list[int]        = []

    for obs_ep, acts_ep in zip(ep_obs, ep_acts):
        for state, joint_act in zip(obs_ep, acts_ep):
            try:
                jax_state = from_legacy_state(terrain, state)
                feats_p0, feats_p1 = featurize_state_64(terrain, jax_state)
                a0, a1 = _joint_action_to_ints(joint_act)
            except Exception as exc:
                if not features:
                    print(
                        f"\n    [WARN] worker {worker_id} / {jax_layout}: "
                        f"{type(exc).__name__}: {exc}  (further failures silenced)"
                    )
                continue

            features.append(np.asarray(feats_p0, dtype=np.float32))
            features.append(np.asarray(feats_p1, dtype=np.float32))
            actions.append(a0)
            actions.append(a1)

    return features, actions


def featurize_layout_split(df, jax_layout: str, split_label: str, terrain) -> dict:
    legacy_layout = JAX_TO_LEGACY[jax_layout]
    layout_df     = df[df["layout_name"] == legacy_layout]
    workers       = list(layout_df["workerid_num"].unique())

    print(
        f"  [{split_label:5s}] {jax_layout:10s} ({legacy_layout}): "
        f"{len(workers)} workers",
        end="",
        flush=True,
    )

    all_features: list[np.ndarray] = []
    all_actions:  list[int]        = []
    skipped = 0

    for worker_id in workers:
        feats, acts = _featurize_worker(worker_id, layout_df, jax_layout, terrain)
        if not feats:
            skipped += 1
            continue
        all_features.extend(feats)
        all_actions.extend(acts)

    if skipped:
        print(f" (skipped {skipped} workers with no data)", end="")

    if not all_features:
        raise ValueError(
            f"No features extracted for layout '{jax_layout}' / split '{split_label}'. "
            f"Looked for '{legacy_layout}' in DataFrame (found: "
            f"{sorted(df['layout_name'].unique())})."
        )

    features_arr = np.stack(all_features)
    actions_arr  = np.array(all_actions, dtype=np.int32)

    print(f" → {features_arr.shape[0]:,} timesteps, feat_dim={features_arr.shape[1]}")

    assert features_arr.shape[1] == 64, (
        f"Expected 64-dim features, got {features_arr.shape}."
    )
    assert actions_arr.min() >= 0 and actions_arr.max() <= 5, (
        f"Action out of [0,5]: min={actions_arr.min()}, max={actions_arr.max()}."
    )

    return {"features": features_arr, "actions": actions_arr}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy human trajectory DataFrames to 64-dim BC feature files."
    )
    parser.add_argument("--train_pkl", default=str(_DEFAULT_TRAIN_PKL))
    parser.add_argument("--test_pkl",  default=str(_DEFAULT_TEST_PKL))
    parser.add_argument("--out_dir",   default=str(_DEFAULT_OUT_DIR))
    parser.add_argument(
        "--layouts", nargs="+", default=LAYOUTS, choices=LAYOUTS, metavar="LAYOUT",
        help=f"Layouts to process: {LAYOUTS}  (default: all)",
    )
    args = parser.parse_args()

    import pandas as pd
    from human_aware_rl_jax_lift.env.layouts import parse_layout

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DataFrames...")
    df_train = pd.read_pickle(args.train_pkl)
    df_test  = pd.read_pickle(args.test_pkl)
    print(f"  train : {len(df_train):>8,} rows")
    print(f"  test  : {len(df_test):>8,} rows")
    print(f"  Layouts in train: {sorted(df_train['layout_name'].unique())}")
    print()

    for jax_layout in args.layouts:
        print(f"=== {jax_layout} ===")
        terrain = parse_layout(jax_layout)
        payload = {
            "train": featurize_layout_split(df_train, jax_layout, "train", terrain),
            "test":  featurize_layout_split(df_test,  jax_layout, "test",  terrain),
        }
        out_path = out_dir / f"{jax_layout}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"  Saved → {out_path}\n")

    print("Done. Verify with:")
    print(
        "  python -c \""
        "import pickle; "
        "[print(L, s, v['features'].shape, v['actions'].shape) "
        "for L in ['simple','unident_s','random0','random1','random3'] "
        "for s,v in pickle.load(open(f'human_aware_rl_jax_lift/data/bc_data/{L}.pkl','rb')).items()]"
        "\""
    )


if __name__ == "__main__":
    main()
