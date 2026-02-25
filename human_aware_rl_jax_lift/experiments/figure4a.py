"""Recreate Figure 4a grouped bar chart from Carroll et al. (NeurIPS 2019)."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


LAYOUT_ORDER = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

LAYOUT_LABELS = [
    "Cramped Room",
    "Asymmetric Advantages",
    "Coordination Ring",
    "Forced Coordination",
    "Counter Circuit",
]

CONDITION_ORDER = [
    "SP_SP",
    "SP_HProxy",
    "PPOBC_HProxy",
    "BC_HProxy",
    "SP_HProxy_sw",
    "PPOBC_HProxy_sw",
    "BC_HProxy_sw",
]

CONDITION_LABELS = {
    "SP_SP": "SP+SP",
    "SP_HProxy": "SP+HProxy",
    "PPOBC_HProxy": "PPO_BC+HProxy",
    "BC_HProxy": "BC+HProxy",
    "SP_HProxy_sw": "SP+HProxy switched",
    "PPOBC_HProxy_sw": "PPO_BC+HProxy switched",
    "BC_HProxy_sw": "BC+HProxy switched",
}


def _seed_value(seed_map: Dict[Any, float], seed: int) -> float:
    if seed in seed_map:
        return float(seed_map[seed])
    if str(seed) in seed_map:
        return float(seed_map[str(seed)])
    raise KeyError(f"Missing seed={seed} in {list(seed_map.keys())}")


def _compute_mean_se(seed_map: Dict[Any, float], num_seeds: int = 5) -> Tuple[float, float]:
    values = np.asarray([_seed_value(seed_map, s) for s in range(num_seeds)], dtype=np.float64)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(num_seeds))
    return mean, se


def compute_stats(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean and standard error per layout/condition."""
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for layout in LAYOUT_ORDER:
        if layout not in results:
            raise KeyError(f"Missing layout '{layout}' in results")
        stats[layout] = {}
        row = results[layout]
        for cond in CONDITION_ORDER:
            if cond not in row:
                raise KeyError(f"Missing condition '{cond}' for layout '{layout}'")
            mean, se = _compute_mean_se(row[cond], num_seeds=5)
            stats[layout][cond] = {"mean": mean, "se": se}
        if "gold_standard" in row and row["gold_standard"] is not None:
            stats[layout]["gold_standard"] = {"mean": float(row["gold_standard"]), "se": 0.0}
    return stats


def print_summary_table(stats: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Print requested summary table to stdout."""
    cols = ["SP_SP", "SP_HProxy", "PPOBC_HProxy", "BC_HProxy"]
    header = (
        f"{'Layout':<24} | {'SP+SP':>14} | {'SP+HProxy':>14} | "
        f"{'PPO_BC+HProxy':>16} | {'BC+HProxy':>14}"
    )
    print(header)
    print("-" * len(header))
    for layout_key, layout_label in zip(LAYOUT_ORDER, LAYOUT_LABELS):
        entries: List[str] = []
        for c in cols:
            m = stats[layout_key][c]["mean"]
            se = stats[layout_key][c]["se"]
            entries.append(f"{m:.2f} ± {se:.2f}")
        print(
            f"{layout_label:<24} | {entries[0]:>14} | {entries[1]:>14} | "
            f"{entries[2]:>16} | {entries[3]:>14}"
        )


def plot_figure_4a(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    teal = "#4DBBD5"
    orange = "#E64B35"
    gray = "#8E8E8E"

    style = {
        "SP_SP": dict(facecolor="#FFFFFF", edgecolor="black", hatch=None, alpha=1.0),
        "SP_HProxy": dict(facecolor=teal, edgecolor="black", hatch=None, alpha=1.0),
        "PPOBC_HProxy": dict(facecolor=orange, edgecolor="black", hatch=None, alpha=1.0),
        "BC_HProxy": dict(facecolor=gray, edgecolor="black", hatch=None, alpha=1.0),
        "SP_HProxy_sw": dict(facecolor=teal, edgecolor="black", hatch="///", alpha=0.85),
        "PPOBC_HProxy_sw": dict(facecolor=orange, edgecolor="black", hatch="///", alpha=0.85),
        "BC_HProxy_sw": dict(facecolor=gray, edgecolor="black", hatch="///", alpha=0.85),
    }

    bar_width = 0.11
    layout_gap = 0.25
    n_layouts = len(LAYOUT_ORDER)
    n_conds = len(CONDITION_ORDER)
    group_span = n_conds * bar_width
    group_stride = group_span + layout_gap

    centers = np.arange(n_layouts, dtype=np.float64) * group_stride
    offsets = (np.arange(n_conds, dtype=np.float64) - (n_conds - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, cond in enumerate(CONDITION_ORDER):
        xs = centers + offsets[i]
        ys = np.array([stats[layout][cond]["mean"] for layout in LAYOUT_ORDER], dtype=np.float64)
        ses = np.array([stats[layout][cond]["se"] for layout in LAYOUT_ORDER], dtype=np.float64)
        cfg = style[cond]
        ax.bar(
            xs,
            ys,
            width=bar_width,
            facecolor=cfg["facecolor"],
            edgecolor=cfg["edgecolor"],
            hatch=cfg["hatch"],
            alpha=cfg["alpha"],
            linewidth=1.0,
            zorder=3,
        )
        ax.errorbar(
            xs,
            ys,
            yerr=ses,
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
            zorder=4,
        )

    # Optional gold standard per layout: draw horizontal line over each bar cluster.
    for center, layout in zip(centers, LAYOUT_ORDER):
        gs = stats[layout].get("gold_standard")
        if gs is None:
            continue
        x0 = center + offsets[0] - bar_width / 2.0
        x1 = center + offsets[-1] + bar_width / 2.0
        ax.hlines(
            y=gs["mean"],
            xmin=x0,
            xmax=x1,
            colors="red",
            linestyles="--",
            linewidth=1.5,
            zorder=5,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(LAYOUT_LABELS, rotation=15)
    ax.set_ylabel("Reward")
    ax.set_ylim(bottom=0)
    ax.set_title("Agent Performance Paired with Human Proxy")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    legend_handles = [
        Patch(facecolor="#FFFFFF", edgecolor="black", label="SP+SP"),
        Patch(facecolor=teal, edgecolor="black", label="SP+HProxy"),
        Patch(facecolor=orange, edgecolor="black", label="PPO_BC+HProxy"),
        Patch(facecolor=gray, edgecolor="black", label="BC+HProxy"),
        Patch(facecolor=teal, edgecolor="black", hatch="///", label="SP+HProxy switched", alpha=0.85),
        Patch(facecolor=orange, edgecolor="black", hatch="///", label="PPO_BC+HProxy switched", alpha=0.85),
        Patch(facecolor=gray, edgecolor="black", hatch="///", label="BC+HProxy switched", alpha=0.85),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Gold standard"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load results dict from JSON or pickle."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported file format '{suffix}'. Use .json or .pkl/.pickle")


def make_example_results() -> Dict[str, Dict[str, Any]]:
    """Small synthetic example for smoke testing plot generation."""
    rng = np.random.RandomState(0)
    out: Dict[str, Dict[str, Any]] = {}
    for layout in LAYOUT_ORDER:
        out[layout] = {}
        base = rng.uniform(40, 120)
        for cond in CONDITION_ORDER:
            out[layout][cond] = {s: float(base + rng.normal(0, 6)) for s in range(5)}
        out[layout]["gold_standard"] = float(base + 8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate Figure 4a grouped bar chart.")
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Path to JSON or pickle containing results[layout][condition][seed] -> float.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figure_4a.png",
        help="Output image path (default: figure_4a.png).",
    )
    args = parser.parse_args()

    if args.results_path is None:
        print("No --results_path provided; using synthetic example data.")
        results = make_example_results()
    else:
        results = load_results(Path(args.results_path))

    stats = compute_stats(results)
    print_summary_table(stats)
    plot_figure_4a(stats, Path(args.output))
    print(f"\nSaved figure to: {args.output}")


if __name__ == "__main__":
    main()
