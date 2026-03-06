"""Recreate Figure 4a grouped bar chart from Carroll et al. (NeurIPS 2019).

Matches the "Performance with human proxy" baseline plot from the NeurIPS
notebook exactly.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


LAYOUT_ORDER = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

LAYOUT_LABELS = [
    "Cramped Rm.",
    "Asymm. Adv.",
    "Coord. Ring",
    "Forced Coord.",
    "Counter Circ.",
]

# Bar order (left to right within each layout group):
#   SP+SP, SP+HProxy, SP+HProxy_sw, PPO_BC+HProxy, PPO_BC+HProxy_sw, BC+HProxy, BC+HProxy_sw
# Deltas match the baseline notebook: width=0.18, 7 bars per group
# baseline deltas for the humanai plot:
#   -2.9, -1.5, -0.5, 0.5, 1.9, 2.9, 3.9  (scaled by width)
CONDITION_ORDER = [
    "SP_SP",
    "SP_HProxy",
    "PPOBC_HProxy",
    "BC_HProxy",
]

CONDITION_ORDER_SW = [
    "SP_HProxy_sw",
    "PPOBC_HProxy_sw",
    "BC_HProxy_sw",
]

# Maps each switched cond to its parent
SW_PARENT = {
    "SP_HProxy_sw":    "SP_HProxy",
    "PPOBC_HProxy_sw": "PPOBC_HProxy",
    "BC_HProxy_sw":    "BC_HProxy",
}

CONDITION_LABELS = {
    "SP_SP":           "SP+SP",
    "SP_HProxy":       "SP+H$_{Proxy}$",
    "PPOBC_HProxy":    "PPO$_{BC}$+H$_{Proxy}$",
    "BC_HProxy":       "BC+H$_{Proxy}$",
}

# Per-histtype y-axis limit matching the baseline notebook ylim dict.
YLIM: Dict[str, float] = {
    "humanai":     250.0,
    "humanaibase": 250.0,
}
DEFAULT_YLIM = 250.0

SeedAgg = Literal["mean", "best"]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _seed_value(seed_map: Dict[Any, float], seed: int) -> float:
    if seed in seed_map:
        return float(seed_map[seed])
    if str(seed) in seed_map:
        return float(seed_map[str(seed)])
    raise KeyError(f"Missing seed={seed} in {list(seed_map.keys())}")


def _seed_values(seed_map: Dict[Any, float], num_seeds: int = 5) -> np.ndarray:
    return np.asarray([_seed_value(seed_map, s) for s in range(num_seeds)], dtype=np.float64)


def _compute_mean_se(seed_map: Dict[Any, float], num_seeds: int = 5) -> Tuple[float, float]:
    """Mean over seeds + standard error (matches baseline meanbyalgo / stdbyalgo)."""
    values = _seed_values(seed_map, num_seeds=num_seeds)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(num_seeds))
    return mean, se


def _compute_best(seed_map: Dict[Any, float], num_seeds: int = 5) -> Tuple[float, float]:
    values = _seed_values(seed_map, num_seeds=num_seeds)
    best = float(np.max(values))
    return best, 0.0


def compute_stats(
    results: Dict[str, Dict[str, Any]],
    *,
    seed_agg: SeedAgg = "mean",
    num_seeds: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute per-layout / per-condition stats.

    seed_agg:
      - "mean" (default): mean over all seeds + standard error  <- matches baseline
      - "best": max over seeds, se=0
    """
    all_conds = CONDITION_ORDER + CONDITION_ORDER_SW
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for layout in LAYOUT_ORDER:
        if layout not in results:
            raise KeyError(f"Missing layout '{layout}' in results")
        stats[layout] = {}
        row = results[layout]
        for cond in all_conds:
            if cond not in row:
                stats[layout][cond] = {"mean": 0.0, "se": 0.0}
                continue
            if seed_agg == "best":
                mean, se = _compute_best(row[cond], num_seeds=num_seeds)
            else:
                mean, se = _compute_mean_se(row[cond], num_seeds=num_seeds)
            stats[layout][cond] = {"mean": mean, "se": se}

        if "gold_standard" in row and row["gold_standard"] is not None:
            stats[layout]["gold_standard"] = {"mean": float(row["gold_standard"]), "se": 0.0}
    return stats


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    *,
    seed_agg: SeedAgg,
) -> None:
    cols = CONDITION_ORDER
    header_labels = [CONDITION_LABELS[c] for c in cols]
    col_width = 20
    header = f"{'Layout':<24} | " + " | ".join(f"{h:>{col_width}}" for h in header_labels)
    print(header)
    print("-" * len(header))
    for layout_key, layout_label in zip(LAYOUT_ORDER, LAYOUT_LABELS):
        entries: List[str] = []
        for c in cols:
            m = stats[layout_key][c]["mean"]
            se = stats[layout_key][c]["se"]
            entries.append(f"{m:.2f} \u00b1 {se:.2f}" if seed_agg == "mean" else f"{m:.2f}")
        print(f"{layout_label:<24} | " + " | ".join(f"{e:>{col_width}}" for e in entries))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _switch_indices(i: int, j: int, lst: list) -> list:
    """Swap elements at positions i and j (returns a new list)."""
    out = list(lst)
    out[i], out[j] = out[j], out[i]
    return out


def plot_figure_4a(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    histtype: str = "humanai",
) -> None:
    """Produce the grouped bar chart faithful to the NeurIPS baseline.

    Bar layout (7 bars per layout group, width=0.18):
      delta -2.9: SP+SP
      delta -1.5: SP+HProxy      (solid teal)
      delta -0.5: SP+HProxy_sw   (hatched teal)
      delta  0.5: PPO_BC+HProxy  (solid orange)
      delta  1.9: PPO_BC+HProxy_sw (hatched orange)
      delta  2.9: BC+HProxy      (solid gray)
      delta  3.9: BC+HProxy_sw   (hatched gray)

    Gold-standard and PPO_HProxy+HProxy reference lines drawn as red hlines.
    Legend uses single 'Switched start indices' hatch patch.
    switch_indices(0,1) applied to legend handles to match baseline ordering.
    """
    teal   = "#4DBBD5"
    orange = "#E64B35"
    gray   = "#8E8E8E"

    style = {
        "SP_SP":           dict(facecolor="#FFFFFF", edgecolor="gray",  hatch=None,  alpha=1.0,  lw=1.0),
        "SP_HProxy":       dict(facecolor=teal,      edgecolor="gray",  hatch=None,  alpha=1.0,  lw=1.0),
        "PPOBC_HProxy":    dict(facecolor=orange,    edgecolor="gray",  hatch=None,  alpha=1.0,  lw=1.0),
        "BC_HProxy":       dict(facecolor=gray,      edgecolor="gray",  hatch=None,  alpha=1.0,  lw=1.0),
        "SP_HProxy_sw":    dict(facecolor=teal,      edgecolor="gray",  hatch="///", alpha=0.85, lw=1.0),
        "PPOBC_HProxy_sw": dict(facecolor=orange,    edgecolor="gray",  hatch="///", alpha=0.85, lw=1.0),
        "BC_HProxy_sw":    dict(facecolor=gray,      edgecolor="gray",  hatch="///", alpha=0.85, lw=1.0),
    }

    # Baseline parameters
    N     = 5
    width = 0.18
    ind   = np.arange(N)

    # 7-bar deltas matching the baseline notebook exactly
    # order: SP_SP, SP_HProxy, SP_HProxy_sw, PPOBC_HProxy, PPOBC_HProxy_sw, BC_HProxy, BC_HProxy_sw
    DELTAS = {
        "SP_SP":           -2.9,
        "SP_HProxy":       -1.5,
        "SP_HProxy_sw":    -0.5,
        "PPOBC_HProxy":     0.5,
        "PPOBC_HProxy_sw":  1.9,
        "BC_HProxy":        2.9,
        "BC_HProxy_sw":     3.9,
    }

    # hline x-extents span the full 7-bar group width per layout tick
    # group spans from (ind - 2.9*width - width/2) to (ind + 3.9*width + width/2)
    # rounded to match baseline: xmin = ind - 0.4, xmax = ind + 0.4 (relative)
    hline_spans = [
        (-0.62, 0.88),   # cramped_room       (ind=0)
        ( 0.38, 1.88),   # asymmetric_advantages (ind=1)
        ( 1.38, 2.88),   # coordination_ring  (ind=2)
        ( 2.38, 3.88),   # forced_coordination (ind=3)
        ( 3.38, 4.88),   # counter_circuit    (ind=4)
    ]

    plt.rc("legend", fontsize=15)
    plt.rc("axes",   titlesize=25)

    fig, ax0 = plt.subplots(1, figsize=(11, 6))
    ax0.tick_params(axis="x", labelsize=18)
    ax0.tick_params(axis="y", labelsize=18.5)

    # --- Primary solid bars ---
    for cond in CONDITION_ORDER:
        if cond == "PPOBCtest":
            continue
        delta  = DELTAS[cond]
        offset = ind + delta * width
        ys  = np.array([stats[layout][cond]["mean"] for layout in LAYOUT_ORDER])
        ses = np.array([stats[layout][cond]["se"]   for layout in LAYOUT_ORDER])
        cfg = style[cond]
        ax0.bar(
            offset, ys, width,
            label=CONDITION_LABELS[cond],
            yerr=ses,
            facecolor=cfg["facecolor"],
            edgecolor=cfg["edgecolor"],
            hatch=cfg["hatch"],
            alpha=cfg["alpha"],
            linewidth=cfg["lw"],
            zorder=3,
            error_kw=dict(ecolor="black", capsize=2, linewidth=1, zorder=4),
        )

    # --- Switched (hatched) bars at their own separate delta offsets ---
    for cond in CONDITION_ORDER_SW:
        ys  = np.array([stats[layout][cond]["mean"] for layout in LAYOUT_ORDER])
        ses = np.array([stats[layout][cond]["se"]   for layout in LAYOUT_ORDER])
        if np.all(ys == 0):
            continue
        delta  = DELTAS[cond]
        offset = ind + delta * width
        cfg = style[cond]
        ax0.bar(
            offset, ys, width,
            # No label here — legend uses a single shared hatch patch below
            yerr=ses,
            facecolor=cfg["facecolor"],
            edgecolor=cfg["edgecolor"],
            hatch=cfg["hatch"],
            alpha=cfg["alpha"],
            linewidth=cfg["lw"],
            zorder=3,
            error_kw=dict(ecolor="black", capsize=2, linewidth=1, zorder=4),
        )

    # --- Gold-standard hlines (red dashed) ---
    for idx, layout in enumerate(LAYOUT_ORDER):
        gs = stats[layout].get("gold_standard")
        if gs is None:
            continue
        xmin, xmax = hline_spans[idx]
        ax0.hlines(
            y=gs["mean"],
            xmin=xmin,
            xmax=xmax,
            colors="red",
            linestyles="dotted",
            linewidth=2.0,
            zorder=5,
        )

    # --- Axes labels & title ---
    ax0.set_ylabel("Average reward per episode")
    ax0.set_title("Performance with human proxy model")

    # x-ticks centred over the 7-bar group
    # group centre = ind + ((-2.9 + 3.9) / 2) * width = ind + 0.5 * width
    ax0.set_xticks(ind + 0.5 * width)
    ax0.set_xticklabels(LAYOUT_LABELS)
    ax0.tick_params(axis="x", labelsize=18)

    ax0.set_ylim(0, YLIM.get(histtype, DEFAULT_YLIM))

    # --- Legend ---
    # Collect handles/labels from the 4 solid bars
    handles, labels = ax0.get_legend_handles_labels()

    # Add single "Switched start indices" hatch patch
    has_sw = any(
        not np.all(np.array([stats[lay][c]["mean"] for lay in LAYOUT_ORDER]) == 0)
        for c in CONDITION_ORDER_SW
    )
    if has_sw:
        sw_patch = Patch(
            facecolor="white", edgecolor="black",
            hatch="///", alpha=0.5,
            label="Switched start indices",
        )
        handles.append(sw_patch)
        labels.append("Switched start indices")

    # switch_indices(0, 1): swap SP+SP (pos 0) with SP+HProxy (pos 1)
    # so legend reads: SP+HProxy, SP+SP, PPO_BC+HProxy, BC+HProxy, Switched...
    # (matches baseline notebook handle reordering)
    handles = _switch_indices(0, 1, handles)
    labels  = _switch_indices(0, 1, labels)

    ax0.legend(handles=handles, labels=labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported file format '{suffix}'. Use .json or .pkl/.pickle")


def make_example_results() -> Dict[str, Dict[str, Any]]:
    rng = np.random.RandomState(0)
    out: Dict[str, Dict[str, Any]] = {}
    all_conds = CONDITION_ORDER + CONDITION_ORDER_SW
    for layout in LAYOUT_ORDER:
        out[layout] = {}
        base = rng.uniform(40, 120)
        for cond in all_conds:
            out[layout][cond] = {s: float(base + rng.normal(0, 6)) for s in range(5)}
        out[layout]["gold_standard"] = float(base + 8)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate Figure 4a grouped bar chart (faithful to NeurIPS baseline)."
    )
    parser.add_argument(
        "--results_path", type=str, default=None,
        help="Path to JSON or pickle: results[layout][condition][seed] -> float.",
    )
    parser.add_argument(
        "--output", type=str, default="figure_4a.png",
        help="Output image path (default: figure_4a.png).",
    )
    parser.add_argument(
        "--seed_agg", type=str, default="mean", choices=["mean", "best"],
        help="Aggregate seeds as mean+SE (default) or best-seed (max).",
    )
    parser.add_argument(
        "--histtype", type=str, default="humanai",
        help="Histogram type key for ylim lookup (default: humanai).",
    )
    args = parser.parse_args()

    if args.results_path is None:
        print("No --results_path provided; using synthetic example data.")
        results = make_example_results()
    else:
        results = load_results(Path(args.results_path))

    stats = compute_stats(results, seed_agg=args.seed_agg, num_seeds=5)
    print_summary_table(stats, seed_agg=args.seed_agg)
    plot_figure_4a(stats, Path(args.output), histtype=args.histtype)


if __name__ == "__main__":
    main()
