"""
Convert legacy human-human trajectory DataFrames into pre-featurized 64-dim BC datasets
compatible with featurize_state_64 and train_bc.py.

Layout name mapping (DataFrame → engine/.layout file):
  cramped_room       → simple
  asymmetric_adv...  → unident_s
  coordination_ring  → random1
  random0            → random0
  random3            → random3

Pipeline per timestep (both players):
  DataFrame row
    → get_overcooked_traj_for_worker_layout()  (legacy OvercookedState objects)
    → from_legacy_state(terrain, state)         (JAX OvercookedState)
    → featurize_state_64(terrain, jax_state)    (feats_p0, feats_p1), each float32[64]
    → {"features": np.ndarray[N,64], "actions": np.ndarray[N]}

Usage (from repo root):
  python human_aware_rl_jax_lift/scripts/prepare_bc_data.py

  # or with explicit paths / subset of layouts:
  python human_aware_rl_jax_lift/scripts/prepare_bc_data.py \\
    --train_pkl human_aware_rl/human_aware_rl/data/human/anonymized/clean_train_trials.pkl \\
    --test_pkl  human_aware_rl/human_aware_rl/data/human/anonymized/clean_test_trials.pkl \\
    --out_dir   human_aware_rl_jax_lift/data/bc_data \\
    --layouts   simple unident_s random0 random1 random3
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Layout name mapping ────────────────────────────────────────────────────────
# Key   = JAX/engine name (used by parse_layout and train_bc.py --layout)
# Value = legacy DataFrame name (used by the human study web interface)
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
# Make both packages importable when running as a script
sys.path.insert(0, str(REPO_ROOT / "human_aware_rl"))  # legacy human_aware_rl package
sys.path.insert(0, str(REPO_ROOT))                      # JAX package root

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

def _joint_action_to_ints(joint_act) -> tuple[int, int]:
    """Safely extract (a0, a1) ints from whatever format the legacy traj stores."""
    a0 = int(np.asarray(joint_act[0]).reshape(-1)[0])
    a1 = int(np.asarray(joint_act[1]).reshape(-1)[0])
    return a0, a1


def _featurize_worker(worker_id, df, legacy_layout: str, terrain) -> tuple[list, list]:
    """
    Featurize all timesteps for one worker on one legacy layout.
    Returns (features_list, actions_list) where each feature is float32[64]
    and each action is an int. Entries for both players are interleaved.
    """
    from human_aware_rl.human.process_dataframes import get_overcooked_traj_for_worker_layout
    from human_aware_rl_jax_lift.env.compat import from_legacy_state
    from human_aware_rl_jax_lift.encoding.bc_features import featurize_state_64

    traj, _ = get_overcooked_traj_for_worker_layout(df, worker_id, legacy_layout)
    if traj is None:
        return [], []

    ep_obs  = traj["ep_observations"]
    ep_acts = traj["ep_actions"]

    # Normalise: some versions wrap in an extra list (multi-episode), some don't.
    # Detect by checking whether the first element is itself a list of states.
    if isinstance(ep_obs[0], list):
        obs_episodes  = ep_obs
        acts_episodes = ep_acts
    else:
        obs_episodes  = [ep_obs]
        acts_episodes = [ep_acts]

    features, actions = [], []
    for obs_ep, acts_ep in zip(obs_episodes, acts_episodes):
        for state, joint_act in zip(obs_ep, acts_ep):
            try:
                jax_state = from_legacy_state(terrain, state)
                feats_p0, feats_p1 = featurize_state_64(terrain, jax_state)
            except Exception:
                # Skip corrupted / truncated states at episode boundaries.
                continue

            a0, a1 = _joint_action_to_ints(joint_act)
            features.append(np.asarray(feats_p0, dtype=np.float32))
            features.append(np.asarray(feats_p1, dtype=np.float32))
            actions.append(a0)
            actions.append(a1)

    return features, actions


def featurize_layout_split(df, jax_layout: str, split_label: str, terrain) -> dict:
    """
    Build features + actions arrays for all workers in `df` on `jax_layout`.
    `split_label` is used only for progress printing.
    """
    legacy_layout = JAX_TO_LEGACY[jax_layout]
    layout_df     = df[df["layout_name"] == legacy_layout]
    workers       = list(layout_df["workerid_num"].unique())
    n_workers     = len(workers)

    print(
        f"  [{split_label:5s}] {jax_layout:10s} ({legacy_layout}): "
        f"{n_workers} workers",
        end="",
        flush=True,
    )

    all_features: list[np.ndarray] = []
    all_actions:  list[int]        = []
    skipped = 0

    for worker_id in workers:
        feats, acts = _featurize_worker(worker_id, layout_df, legacy_layout, terrain)
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
            f"Check JAX_TO_LEGACY mapping: looked for '{legacy_layout}' in the DataFrame "
            f"but found layouts: {sorted(df['layout_name'].unique())}."
        )

    features_arr = np.stack(all_features)          # [N, 64]
    actions_arr  = np.array(all_actions, dtype=np.int32)  # [N]

    print(f" → {features_arr.shape[0]:,} timesteps, feat_dim={features_arr.shape[1]}")

    # Sanity checks
    assert features_arr.shape[1] == 64, (
        f"Expected 64-dim features, got shape {features_arr.shape}. "
        "Verify featurize_state_64 in encoding/bc_features.py."
    )
    assert 0 <= actions_arr.min() and actions_arr.max() <= 5, (
        f"Action values out of [0,5] range: min={actions_arr.min()}, max={actions_arr.max()}."
    )

    return {"features": features_arr, "actions": actions_arr}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert legacy human-human trajectory DataFrames into 64-dim "
            "pre-featurized BC datasets for human_aware_rl_jax_lift."
        )
    )
    parser.add_argument(
        "--train_pkl",
        default=str(_DEFAULT_TRAIN_PKL),
        help="Path to clean_train_trials.pkl  (default: %(default)s)",
    )
    parser.add_argument(
        "--test_pkl",
        default=str(_DEFAULT_TEST_PKL),
        help="Path to clean_test_trials.pkl   (default: %(default)s)",
    )
    parser.add_argument(
        "--out_dir",
        default=str(_DEFAULT_OUT_DIR),
        help="Output directory for data/bc_data/*.pkl  (default: %(default)s)",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=LAYOUTS,
        choices=LAYOUTS,
        metavar="LAYOUT",
        help=f"Layouts to process. Choices: {LAYOUTS}  (default: all)",
    )
    args = parser.parse_args()

    # Lazy imports so the script fails fast on bad paths before heavy imports.
    import pandas as pd
    from human_aware_rl_jax_lift.env.layouts import parse_layout

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load DataFrames ──
    print("Loading DataFrames...")
    df_train = pd.read_pickle(args.train_pkl)
    df_test  = pd.read_pickle(args.test_pkl)
    print(f"  train : {len(df_train):>8,} rows")
    print(f"  test  : {len(df_test):>8,} rows")
    print(f"  Layout names found in train: {sorted(df_train['layout_name'].unique())}")
    print()

    # ── Process each layout ──
    for jax_layout in args.layouts:
        print(f"=== {jax_layout} ===")
        # parse_layout uses the JAX/engine name to read the .layout file
        terrain = parse_layout(jax_layout)

        payload: dict = {}
        payload["train"] = featurize_layout_split(df_train, jax_layout, "train", terrain)
        payload["test"]  = featurize_layout_split(df_test,  jax_layout, "test",  terrain)

        out_path = out_dir / f"{jax_layout}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f)
        print(f"  Saved → {out_path}\n")

    # ── Final verification hint ──
    print("Done. Verify with:")
    print(
        "  python -c \""
        "import pickle; "
        "[print(L, s, v['features'].shape, v['actions'].shape) "
        "for L in ['simple','unident_s','random0','random1','random3'] "
        "for s,v in pickle.load(open(f'human_aware_rl_jax_lift/data/bc_data/{L}.pkl','rb')).items()]"
        "\""
    )
    print()
    print("Expected output (~9 000 timesteps per layout/split, feature_dim=64):")
    print("  simple     train (9000, 64) (9000,)")
    print("  simple     test  (9000, 64) (9000,)")
    print("  ...")


if __name__ == "__main__":
    main()
