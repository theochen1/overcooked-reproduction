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
#   SP+SP, SP+HProxy, PPOBC+HProxy, BC+HProxy,
#   SP+HProxy_sw, PPOBC+HProxy_sw, BC+HProxy_sw
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

# Per-histtype y-axis limit
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
    """Produce the grouped bar chart.

    Bar order left-to-right within each layout group:
      -3: SP+SP
      -2: SP+HProxy        (solid teal)
      -1: PPOBC+HProxy     (solid orange)
       0: BC+HProxy        (solid gray)
      +1: SP+HProxy_sw     (hatched teal)
      +2: PPOBC+HProxy_sw  (hatched orange)
      +3: BC+HProxy_sw     (hatched gray)

    Gold-standard reference line drawn as red dotted hline.
    Legend uses single 'Switched start indices' hatch patch.
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

    N     = 5
    width = 0.12
    ind   = np.arange(N)

    # All 4 neutrals first, then all 3 switched — each group spans
    # deltas -3..+3 (7 bars), centre = 0.0, outermost at ±3.5*width = ±0.42.
    DELTAS = {
        "SP_SP":           -3.0,
        "SP_HProxy":       -2.0,
        "PPOBC_HProxy":    -1.0,
        "BC_HProxy":        0.0,
        "SP_HProxy_sw":     1.0,
        "PPOBC_HProxy_sw":  2.0,
        "BC_HProxy_sw":     3.0,
    }

    half_group = 3.5 * width  # 0.42
    hline_spans = [
        (ind[i] - half_group, ind[i] + half_group)
        for i in range(N)
    ]

    plt.rc("legend", fontsize=13)
    plt.rc("axes",   titlesize=22)

    fig, ax0 = plt.subplots(1, figsize=(18, 6))
    ax0.tick_params(axis="x", labelsize=16)
    ax0.tick_params(axis="y", labelsize=16)

    # --- Primary solid bars ---
    for cond in CONDITION_ORDER:
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

    # --- Switched (hatched) bars ---
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
            yerr=ses,
            facecolor=cfg["facecolor"],
            edgecolor=cfg["edgecolor"],
            hatch=cfg["hatch"],
            alpha=cfg["alpha"],
            linewidth=cfg["lw"],
            zorder=3,
            error_kw=dict(ecolor="black", capsize=2, linewidth=1, zorder=4),
        )

    # --- Gold-standard hlines (red dotted) ---
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
    ax0.set_ylabel("Average reward per episode", fontsize=16)
    ax0.set_title("Performance with human proxy model")

    # x-ticks centred over the 7-bar group (centre delta = 0.0 -> BC_HProxy at ind+0)
    ax0.set_xticks(ind)
    ax0.set_xticklabels(LAYOUT_LABELS)

    ax0.set_ylim(0, YLIM.get(histtype, DEFAULT_YLIM))
    ax0.set_xlim(-0.6, N - 1 + 0.6)

    # --- Legend ---
    handles, labels = ax0.get_legend_handles_labels()

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
