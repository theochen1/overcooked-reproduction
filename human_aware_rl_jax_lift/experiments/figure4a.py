"""Recreate Figure 4a grouped bar chart from Carroll et al. (NeurIPS 2019).

Matches the "Performance with human proxy" baseline plot from the NeurIPS
notebook exactly:
  - figsize (11, 6)
  - bar width 0.18
  - 4 primary conditions per layout, deltas [-1, 0, 1, 2]
  - mean-over-seeds with standard error (default; --seed_agg best available)
  - hlines spanning calibrated to width=0.18 offsets
  - legend handles reordered with switch_indices(0, 1)
  - PPOBCtest condition skipped
  - y-axis limited to ylim dict value for 'humanai'
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

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
    "Cramped Rm.",
    "Asymm. Adv.",
    "Coord. Ring",
    "Forced Coord.",
    "Counter Circ.",
]

# Primary conditions plotted as bars (PPOBCtest is explicitly skipped).
# Switched-index conditions rendered with hatching.
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

CONDITION_LABELS = {
    "SP_SP": "SP+SP",
    "SP_HProxy": "SP+HProxy",
    "PPOBC_HProxy": "PPO_BC+HProxy",
    "BC_HProxy": "BC+HProxy",
    "SP_HProxy_sw": "SP+HProxy switched",
    "PPOBC_HProxy_sw": "PPO_BC+HProxy switched",
    "BC_HProxy_sw": "BC+HProxy switched",
}

# Per-histtype y-axis limit matching the baseline notebook ylim dict.
YLIM: Dict[str, float] = {
    "humanai": 300.0,
    "humanaibase": 300.0,
}
DEFAULT_YLIM = 300.0

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
                # Switched conditions may be absent; fill with zeros.
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
    col_width = 16
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
# Plotting — faithful to baseline "Performance with human proxy" notebook cell
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

    Visual parameters are taken directly from the NeurIPS notebook baseline:
      figsize = (11, 6)
      N = 5 (layouts)
      width = 0.18
      deltas = [-1, 0, 1, 2]  (primary 4-condition groups)
      hlines xmin/xmax calibrated to width=0.18 offsets
      legend handles reordered via switch_indices(0, 1)
      ylim from YLIM dict
    """
    teal = "#4DBBD5"
    orange = "#E64B35"
    gray = "#8E8E8E"

    style = {
        "SP_SP":          dict(facecolor="#FFFFFF", edgecolor="gray",  hatch=None,   alpha=1.0,  lw=1.0),
        "SP_HProxy":      dict(facecolor=teal,      edgecolor="gray",  hatch=None,   alpha=1.0,  lw=1.0),
        "PPOBC_HProxy":   dict(facecolor=orange,    edgecolor="gray",  hatch=None,   alpha=1.0,  lw=1.0),
        "BC_HProxy":      dict(facecolor=gray,      edgecolor="gray",  hatch=None,   alpha=1.0,  lw=1.0),
        "SP_HProxy_sw":   dict(facecolor=teal,      edgecolor="gray",  hatch="///",  alpha=0.85, lw=1.0),
        "PPOBC_HProxy_sw":dict(facecolor=orange,    edgecolor="gray",  hatch="///",  alpha=0.85, lw=1.0),
        "BC_HProxy_sw":   dict(facecolor=gray,      edgecolor="gray",  hatch="///",  alpha=0.85, lw=1.0),
    }

    # --- Baseline parameters ---
    N = 5                            # number of layouts
    width = 0.18                     # bar width  (matches baseline)
    deltas = [-1, 0, 1, 2]           # offsets for 4 primary conditions
    ind = np.arange(N)               # layout x-positions

    # hline x-extents calibrated to width=0.18 and deltas, matching baseline:
    #   simple    -> ind[0] => x=0; SP_SP offset = -1*0.18 = -0.18 -> xmin=-0.4; BC offset=2*0.18=0.36 -> xmax=0.4
    #   unidents  -> ind[1] => x=1; xmin=0.6, xmax=1.4
    #   ring      -> ind[2] => x=2; xmin=1.6, xmax=2.4
    #   forced    -> ind[3] => x=3; xmin=2.6, xmax=3.4
    #   counter   -> ind[4] => x=4; xmin=3.6, xmax=4.4
    hline_spans = [
        (-0.40, 0.40),   # cramped_room
        ( 0.60, 1.40),   # asymmetric_advantages
        ( 1.60, 2.40),   # coordination_ring
        ( 2.60, 3.45),   # forced_coordination  (baseline uses 3.45)
        ( 3.60, 4.40),   # counter_circuit
    ]

    plt.rc("legend", fontsize=18)
    plt.rc("axes",   titlesize=25)

    fig, ax0 = plt.subplots(1, figsize=(11, 6))   # matches baseline figsize
    ax0.tick_params(axis="x", labelsize=18)
    ax0.tick_params(axis="y", labelsize=18)

    # --- Primary 4-condition bars ---
    for i, cond in enumerate(CONDITION_ORDER):
        # PPOBCtest is not in CONDITION_ORDER, but guard explicitly.
        if cond == "PPOBCtest":
            continue
        delta = deltas[i]
        offset = ind + delta * width
        ys  = np.array([stats[layout][cond]["mean"] for layout in LAYOUT_ORDER])
        ses = np.array([stats[layout][cond]["se"]   for layout in LAYOUT_ORDER])
        cfg = style[cond]

        # SP_SP / reference conditions use no fill colour (colornone -> white + edgecolor gray)
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
            error_kw=dict(ecolor="black", capsize=3, linewidth=1, zorder=4),
        )

    # --- Switched-index bars (hatched) ---
    for cond in CONDITION_ORDER_SW:
        if cond not in stats[LAYOUT_ORDER[0]]:
            continue
        cfg = style[cond]
        ys  = np.array([stats[layout][cond]["mean"] for layout in LAYOUT_ORDER])
        ses = np.array([stats[layout][cond]["se"]   for layout in LAYOUT_ORDER])
        # Switched bars are plotted at offset 0 relative to their parent algo
        # (baseline overlays them; omit if all zeros to avoid clutter)
        if np.all(ys == 0):
            continue
        parent = cond.replace("_sw", "")
        parent_delta = deltas[CONDITION_ORDER.index(parent)] if parent in CONDITION_ORDER else 0
        offset = ind + parent_delta * width
        ax0.bar(
            offset, ys, width,
            label=CONDITION_LABELS[cond] + " (sw)",
            yerr=ses,
            facecolor=cfg["facecolor"],
            edgecolor=cfg["edgecolor"],
            hatch=cfg["hatch"],
            alpha=cfg["alpha"],
            linewidth=cfg["lw"],
            zorder=3,
            error_kw=dict(ecolor="black", capsize=3, linewidth=1, zorder=4),
        )

    # --- Gold-standard hlines ---
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
            linestyles="--",
            linewidth=1.5,
            zorder=5,
        )

    # --- Axes labels & title ---
    ax0.set_ylabel("Average reward per episode")
    ax0.set_title("Performance with human proxy model")

    # x-tick centres at ind + 1.5*width (centred over the 4-bar group, matching baseline)
    ax0.set_xticks(ind + 1.5 * width)
    ax0.set_xticklabels(LAYOUT_LABELS)
    ax0.tick_params(axis="x", labelsize=18)

    ax0.set_ylim(0, YLIM.get(histtype, DEFAULT_YLIM))
    ax0.grid(axis="y", alpha=0.3, zorder=0)

    # --- Legend with switch_indices(0,1) reordering (matches baseline) ---
    handles, labels = ax0.get_legend_handles_labels()
    # Manually append the "Switched start indices" patch if switched bars present
    has_sw = any(
        not np.all(np.array([stats[lay][c]["mean"] for lay in LAYOUT_ORDER]) == 0)
        for c in CONDITION_ORDER_SW
        if c in stats[LAYOUT_ORDER[0]]
    )
    if has_sw:
        patch = Patch(
            facecolor="white", edgecolor="black",
            hatch="///", alpha=0.5,
            label="Switched start indices",
        )
        handles.append(patch)

    # Apply switch_indices(0, 1) as in the baseline notebook
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
