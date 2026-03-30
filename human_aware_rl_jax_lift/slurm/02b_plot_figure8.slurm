#!/bin/bash
#SBATCH --job-name=plot_figure8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

# Usage:
#   sbatch 02b_plot_figure8.slurm
# Optional overrides:
#   sbatch --export=ALL,REPO_ROOT=/path/to/human_aware_rl_jax_lift,RUNS_DIR=/path/to/ppo_runs,OUT_DIR=/path/to/out 02b_plot_figure8.slurm
#
# This script is self-contained and runs under Slurm without Docker.
# It scans PPO self-play training outputs (training_info.pkl per seed), groups by layout,
# aggregates over seeds, and produces Figure 8–style training curves:
# “PPO_SP self-play average episode rewards on each layout during training.”

# Under SLURM, use submit dir so paths stay correct (script may run from spool).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR/.."
  mkdir -p "$SLURM_SUBMIT_DIR/logs"
else
  mkdir -p logs
  cd "$(dirname "$0")/.."
fi

# Match 01_bc / 02_ppo_sp: load conda so python (numpy, matplotlib) is available
module load miniforge 2>/dev/null || true
conda activate overcooked 2>/dev/null || true

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(pwd)"
fi

if [[ -z "${RUNS_DIR:-}" ]]; then
  RUNS_DIR="$REPO_ROOT/data/ppo_runs"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/figures}"
mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export REPO_ROOT RUNS_DIR OUT_DIR

python - <<'PY'
from __future__ import annotations

import math
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(os.environ["REPO_ROOT"]).resolve()
RUNS_DIR = Path(os.environ["RUNS_DIR"]).resolve()
OUT_DIR = Path(os.environ["OUT_DIR"]).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Layouts and display names (paper Figure 8 style; order matches 02_ppo_sp.slurm).
LAYOUT_ORDER = ["simple", "unident_s", "random1", "random0", "random3"]
LAYOUT_TITLES = {
    "simple": "Simple",
    "unident_s": "Unident S",
    "random0": "Random 0",
    "random1": "Random 1",
    "random3": "Random 3",
}

# training_info.pkl is written once per PPO update; each update uses this many env steps.
# From paper: num_envs=30, horizon=400.
STEPS_PER_UPDATE = 30 * 400

# Metric to plot: sparse (true) episode reward mean (no shaping).
REWARD_KEY = "ep_sparse_rew_mean"


def load_training_info(pkl_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (steps, values) from training_info.pkl. steps are env timesteps."""
    try:
        with pkl_path.open("rb") as f:
            logs = pickle.load(f)
    except Exception:
        return None
    y = logs.get(REWARD_KEY, [])
    if not y:
        return None
    y = np.asarray(y, dtype=np.float64)
    # Step index 0 = first update => STEPS_PER_UPDATE env steps, etc.
    x = (np.arange(len(y)) + 1) * STEPS_PER_UPDATE
    return x, y


def discover_runs(runs_dir: Path) -> dict[str, list[tuple[Path, np.ndarray, np.ndarray]]]:
    """Group (run_dir, x, y) by layout. Run dirs: ppo_sp_<layout> or ppo_sp_jax_<layout>."""
    grouped = {}
    for layout in LAYOUT_ORDER:
        grouped[layout] = []
        for run_name in [f"ppo_sp_{layout}", f"ppo_sp_jax_{layout}"]:
            run_dir = runs_dir / run_name
            if not run_dir.is_dir():
                continue
            for seed_dir in sorted(run_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                    continue
                pkl_path = seed_dir / "training_info.pkl"
                if not pkl_path.exists():
                    continue
                result = load_training_info(pkl_path)
                if result is None:
                    continue
                x, y = result
                grouped[layout].append((seed_dir, x, y))
        # Dedupe by seed dir (in case both ppo_sp_ and ppo_sp_jax_ exist for same layout)
        seen = set()
        unique = []
        for item in grouped[layout]:
            key = item[0]
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        grouped[layout] = unique
    return grouped


def smooth(y: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = weight * out[i - 1] + (1.0 - weight) * y[i]
    return out


def aggregate(runs: list, smoothing: float = 0.6, points: int = 300):
    """Interpolate all runs to a common grid, smooth, return (grid, mean, stderr, n_runs)."""
    if not runs:
        return None
    xs = [x for _, x, _ in runs]
    ys = [y for _, _, y in runs]
    x_min = max(float(x[0]) for x in xs)
    x_max = min(float(x[-1]) for x in xs)
    if x_max <= x_min:
        x_min = min(float(x[0]) for x in xs)
        x_max = max(float(x[-1]) for x in xs)
    grid = np.linspace(x_min, x_max, points)
    smoothed = []
    for x, y in zip(xs, ys):
        y_interp = np.interp(grid, x, y)
        smoothed.append(smooth(y_interp, smoothing))
    arr = np.stack(smoothed, axis=0)
    mean = arr.mean(axis=0)
    n = arr.shape[0]
    stderr = arr.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(mean)
    return grid, mean, stderr, len(runs)


def main():
    grouped = discover_runs(RUNS_DIR)
    if not any(grouped.values()):
        raise SystemExit(
            f"No PPO self-play training_info.pkl found under {RUNS_DIR}. "
            "Run 02_ppo_sp.slurm first, or set RUNS_DIR to your data/ppo_runs."
        )

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    summary_lines = []
    for i, layout in enumerate(LAYOUT_ORDER):
        ax = axes[i]
        runs = grouped.get(layout, [])
        agg = aggregate(runs)
        if agg is None:
            ax.set_title(f"{LAYOUT_TITLES[layout]}\n(no runs found)")
            ax.axis("off")
            continue

        grid, mean, stderr, n_runs = agg
        ax.plot(grid, mean, linewidth=2)
        ax.fill_between(grid, mean - stderr, mean + stderr, alpha=0.25)
        ax.set_title(f"{LAYOUT_TITLES[layout]} (n={n_runs})")
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Average episode reward")
        ax.grid(alpha=0.3)
        summary_lines.append(f"{layout}: {n_runs} run(s)")

    fig.suptitle("Figure 8 — PPO_SP self-play average episode rewards during training", fontsize=14)

    png_path = OUT_DIR / "figure8_ppo_sp_training_curves.png"
    pdf_path = OUT_DIR / "figure8_ppo_sp_training_curves.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    txt_path = OUT_DIR / "figure8_ppo_sp_training_curves_summary.txt"
    txt_path.write_text(
        "PPO_SP runs by layout:\n" + "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {txt_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
PY

echo "Done: $(date)"
