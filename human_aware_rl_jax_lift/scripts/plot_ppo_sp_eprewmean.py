"""Plot average true_eprew (ep_sparse_rew_mean) across seeds for each PPO_SP layout (5 layouts, 5 seeds each).

Requires: numpy, matplotlib. Run from repo root with conda/env active, e.g.:
  conda activate overcooked
  cd human_aware_rl_jax_lift && python scripts/plot_ppo_sp_eprewmean.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LAYOUTS = ["simple", "unident_s", "random0", "random1", "random3"]
PPO_SP_SEEDS = [2229, 7649, 7225, 9807, 386]
# From paper: num_envs=30, horizon=400 → steps per update = 12000
STEPS_PER_UPDATE = 30 * 400


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO_SP eprewmean by layout (mean over seeds).")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("data/ppo_runs"),
        help="Root dir containing ppo_sp_jax_<layout>/seed<N>/training_info.pkl",
    )
    parser.add_argument("--output", type=Path, default=Path("data/ppo_sp_true_eprew_by_layout.png"))
    parser.add_argument("--title", type=str, default="PPO SP: mean true_eprew across seeds")
    args = parser.parse_args()

    layout_curves = {}  # layout -> dict with "steps", "mean", "std", "n_seeds"

    for layout in LAYOUTS:
        run_dir = args.save_dir / f"ppo_sp_jax_{layout}"
        if not run_dir.exists():
            print(f"Skip {layout}: missing {run_dir}")
            continue

        all_eprew = []  # list of arrays (one per seed), possibly different lengths
        for seed in PPO_SP_SEEDS:
            pkl_path = run_dir / f"seed{seed}" / "training_info.pkl"
            if not pkl_path.exists():
                print(f"Skip {layout} seed {seed}: missing {pkl_path}")
                continue
            with pkl_path.open("rb") as f:
                logs = pickle.load(f)
            # true_eprew = ep_sparse_rew_mean (sparse reward, no shaping)
            eprew = logs.get("ep_sparse_rew_mean", [])
            if eprew:
                all_eprew.append(np.asarray(eprew, dtype=np.float64))

        if not all_eprew:
            print(f"No data for layout {layout}")
            continue

        # Align by minimum length so we average over same number of updates for all seeds
        min_len = min(len(a) for a in all_eprew)
        stacked = np.array([a[:min_len] for a in all_eprew])
        steps = (np.arange(min_len) + 1) * STEPS_PER_UPDATE
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)

        layout_curves[layout] = {
            "steps": steps,
            "mean": mean,
            "std": std,
            "n_seeds": stacked.shape[0],
        }

    if not layout_curves:
        print("No data found. Exiting.")
        return

    n_plots = len(LAYOUTS)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]
    for ax, layout in zip(axes, LAYOUTS):
        if layout not in layout_curves:
            ax.set_visible(False)
            continue
        d = layout_curves[layout]
        ax.plot(d["steps"], d["mean"])
        if d["n_seeds"] > 1 and np.any(d["std"] > 0):
            ax.fill_between(
                d["steps"],
                d["mean"] - d["std"],
                d["mean"] + d["std"],
                alpha=0.25,
            )
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("true_eprew (mean ± std)" if ax == axes[0] else "")
        ax.set_title(f"{layout} (n={d['n_seeds']})")
        ax.grid(True, alpha=0.3)
    fig.suptitle(args.title, y=1.02)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
