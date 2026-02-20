"""
Plot Figure 4-style evaluation results for HPC models.

Expected input JSON structure:
{
  "cramped_room": {
    "sp_sp": {"mean": ..., "se": ...},
    "sp_hp": {"mean": ..., "se": ...},
    "sp_hp_swapped": {"mean": ..., "se": ...},
    ...
  },
  ...
}
"""

import argparse
import json
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "self_play": "#4A90A4",
    "ppo_bc": "#E8944A",
    "ppo_gail": "#8E6CCF",
    "bc": "#808080",
    "gail": "#5DA5DA",
    "baseline_white": "white",
}

LAYOUT_NAMES = {
    "cramped_room": "Cramped Rm.",
    "asymmetric_advantages": "Asymm. Adv.",
    "coordination_ring": "Coord. Ring",
    "forced_coordination": "Forced Coord.",
    "counter_circuit": "Counter Circ.",
}

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]


def load_results(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def build_series(include_gail: bool) -> List[Tuple[str, str, str, str]]:
    series = [
        ("sp_sp", "SP+SP", COLORS["baseline_white"], None),
        ("sp_hp", "SP+H$_{Proxy}$", COLORS["self_play"], None),
        ("sp_hp_swapped", "SP+H$_{Proxy}$ (swapped)", COLORS["self_play"], "///"),
        ("ppo_bc_hp", "PPO$_{BC}$+H$_{Proxy}$", COLORS["ppo_bc"], None),
        ("ppo_bc_hp_swapped", "PPO$_{BC}$+H$_{Proxy}$ (swapped)", COLORS["ppo_bc"], "///"),
        ("ppo_gail_hp", "PPO$_{GAIL}$+H$_{Proxy}$", COLORS["ppo_gail"], None),
        ("ppo_gail_hp_swapped", "PPO$_{GAIL}$+H$_{Proxy}$ (swapped)", COLORS["ppo_gail"], "///"),
        ("bc_hp", "BC+H$_{Proxy}$", COLORS["bc"], None),
        ("bc_hp_swapped", "BC+H$_{Proxy}$ (swapped)", COLORS["bc"], "///"),
    ]
    if include_gail:
        series.extend(
            [
                ("gail_hp", "GAIL+H$_{Proxy}$", COLORS["gail"], None),
                ("gail_hp_swapped", "GAIL+H$_{Proxy}$ (swapped)", COLORS["gail"], "///"),
            ]
        )
    return series


def plot_results(data: Dict, include_gail: bool, save_path: str = None):
    series = build_series(include_gail)
    num_series = len(series)

    x = np.arange(len(LAYOUTS))
    width = min(0.12, 0.8 / max(num_series, 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (key, label, color, hatch) in enumerate(series):
        means = [data.get(layout, {}).get(key, {}).get("mean", 0) for layout in LAYOUTS]
        ses = [data.get(layout, {}).get(key, {}).get("se", 0) for layout in LAYOUTS]
        offset = (idx - (num_series - 1) / 2) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=ses,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            capsize=2,
            hatch=hatch,
            label=label if "swapped" not in key else None,
        )

    ax.set_ylabel("Average reward per episode", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYOUT_NAMES[l] for l in LAYOUTS], fontsize=10)
    ax.set_ylim(0, 260)
    ax.set_title("Performance with human proxy model", fontsize=12, fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["baseline_white"], edgecolor="black", linewidth=1.5, label="SP+SP"),
        mpatches.Patch(facecolor=COLORS["self_play"], edgecolor="black", label="SP+H$_{Proxy}$"),
        mpatches.Patch(facecolor=COLORS["ppo_bc"], edgecolor="black", label="PPO$_{BC}$+H$_{Proxy}$"),
        mpatches.Patch(facecolor=COLORS["ppo_gail"], edgecolor="black", label="PPO$_{GAIL}$+H$_{Proxy}$"),
        mpatches.Patch(facecolor=COLORS["bc"], edgecolor="black", label="BC+H$_{Proxy}$"),
    ]
    if include_gail:
        legend_elements.append(
            mpatches.Patch(facecolor=COLORS["gail"], edgecolor="black", label="GAIL+H$_{Proxy}$")
        )
    legend_elements.append(
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Switched indices")
    )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot HPC evaluation results")
    parser.add_argument("--input", type=str, required=True, help="Path to evaluation results JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to save the figure")
    parser.add_argument("--include_gail", action="store_true", help="Include GAIL+HP in the plot")
    args = parser.parse_args()

    data = load_results(args.input)
    plot_results(data, include_gail=args.include_gail, save_path=args.output)


if __name__ == "__main__":
    main()
