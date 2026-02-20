"""
Visualization Script for Overcooked AI Paper Results (Figure 4).

This script generates paper-style figures matching Figure 4 from:
"On the Utility of Learning about Humans for Human-AI Coordination"

Figure 4(a): Comparison with agents trained in self-play
Figure 4(b): Comparison with agents trained via PBT

Usage:
    python -m human_aware_rl.visualization.plot_results \\
        --results_file paper_results.json \\
        --output_dir figures/
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# Import matplotlib - required for this module
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.axes import Axes as MplAxes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    MplAxes = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from human_aware_rl.ppo.configs.paper_configs import PAPER_LAYOUTS


# =============================================================================
# Paper Display Names and Colors
# =============================================================================

LAYOUT_DISPLAY_NAMES = {
    "cramped_room": "Cramped Rm.",
    "asymmetric_advantages": "Asymm. Adv.",
    "coordination_ring": "Coord. Ring",
    "forced_coordination": "Forced Coord.",
    "counter_circuit": "Counter Circ.",
}

# Figure 4(a) configs - Self-Play comparison
FIGURE_4A_DISPLAY = {
    "ppo_hp_hp": {"name": r"$\mathrm{PPO}_{H_{Proxy}}$+$H_{Proxy}$", "color": "red", "style": "line"},
    "sp_sp": {"name": "SP+SP", "color": "none", "style": "bar_hollow"},
    "sp_hp": {"name": r"SP+$H_{Proxy}$", "color": "#2d6777", "style": "bar"},  # Teal
    "ppo_bc_hp": {"name": r"$\mathrm{PPO}_{BC}$+$H_{Proxy}$", "color": "#F79646", "style": "bar"},  # Orange
    "bc_hp": {"name": r"BC+$H_{Proxy}$", "color": "#7f7f7f", "style": "bar"},  # Gray
}

# Figure 4(b) configs - PBT comparison
FIGURE_4B_DISPLAY = {
    "ppo_hp_hp": {"name": r"$\mathrm{PPO}_{H_{Proxy}}$+$H_{Proxy}$", "color": "red", "style": "line"},
    "pbt_pbt": {"name": "PBT+PBT", "color": "none", "style": "bar_hollow"},
    "pbt_hp": {"name": r"PBT+$H_{Proxy}$", "color": "#2d6777", "style": "bar"},  # Teal
    "ppo_bc_hp": {"name": r"$\mathrm{PPO}_{BC}$+$H_{Proxy}$", "color": "#F79646", "style": "bar"},  # Orange
    "bc_hp": {"name": r"BC+$H_{Proxy}$", "color": "#7f7f7f", "style": "bar"},  # Gray
}


def set_paper_style():
    """Set matplotlib style to match paper aesthetics."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    # Use serif fonts like the paper
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
    })
    
    # Try LaTeX rendering (may not be available)
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except Exception:
        plt.rcParams['text.usetex'] = False


def plot_figure_4(
    results: Dict[str, Any],
    output_dir: str = "figures",
    fmt: str = "pdf",
    show: bool = False,
):
    """
    Generate Figure 4 with both subplots (a) and (b).
    
    Args:
        results: Dictionary with 'figure_4a' and/or 'figure_4b' results
        output_dir: Directory to save figures
        fmt: Output format (pdf, png, svg)
        show: Whether to display the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    set_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    has_4a = "figure_4a" in results
    has_4b = "figure_4b" in results
    
    if has_4a and has_4b:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        ax_4a, ax_4b = axes
    elif has_4a:
        fig, ax_4a = plt.subplots(1, 1, figsize=(14, 5))
        ax_4b = None
    elif has_4b:
        fig, ax_4b = plt.subplots(1, 1, figsize=(14, 5))
        ax_4a = None
    else:
        print("No results to plot")
        return
    
    # Plot Figure 4(a)
    if has_4a and ax_4a is not None:
        _plot_figure_subplot(
            ax=ax_4a,
            data=results["figure_4a"],
            config=FIGURE_4A_DISPLAY,
            title="Performance with human proxy model",
            subtitle="(a) Comparison with agents trained in self-play.",
            bar_configs=["sp_sp", "sp_hp", "ppo_bc_hp", "bc_hp"],
            line_config="ppo_hp_hp",
        )
    
    # Plot Figure 4(b)
    if has_4b and ax_4b is not None:
        _plot_figure_subplot(
            ax=ax_4b,
            data=results["figure_4b"],
            config=FIGURE_4B_DISPLAY,
            title="Performance with human proxy model",
            subtitle="(b) Comparison with agents trained via PBT.",
            bar_configs=["pbt_pbt", "pbt_hp", "ppo_bc_hp", "bc_hp"],
            line_config="ppo_hp_hp",
        )
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"figure_4.{fmt}")
    plt.savefig(output_path, format=fmt, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Also save individual figures
    if has_4a:
        _plot_single_figure(
            results["figure_4a"],
            FIGURE_4A_DISPLAY,
            "Performance with human proxy model",
            "(a) Comparison with agents trained in self-play.",
            ["sp_sp", "sp_hp", "ppo_bc_hp", "bc_hp"],
            "ppo_hp_hp",
            os.path.join(output_dir, f"figure_4a.{fmt}"),
            fmt,
        )
    
    if has_4b:
        _plot_single_figure(
            results["figure_4b"],
            FIGURE_4B_DISPLAY,
            "Performance with human proxy model",
            "(b) Comparison with agents trained via PBT.",
            ["pbt_pbt", "pbt_hp", "ppo_bc_hp", "bc_hp"],
            "ppo_hp_hp",
            os.path.join(output_dir, f"figure_4b.{fmt}"),
            fmt,
        )


def _plot_single_figure(
    data: Dict,
    config: Dict,
    title: str,
    subtitle: str,
    bar_configs: List[str],
    line_config: str,
    output_path: str,
    fmt: str,
):
    """Plot a single figure (4a or 4b)."""
    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    _plot_figure_subplot(ax, data, config, title, subtitle, bar_configs, line_config)
    plt.tight_layout()
    plt.savefig(output_path, format=fmt, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {output_path}")
    plt.close()


def _plot_figure_subplot(
    ax: Any,  # matplotlib.axes.Axes
    data: Dict[str, Dict],
    config: Dict[str, Dict],
    title: str,
    subtitle: str,
    bar_configs: List[str],
    line_config: str,
):
    """Plot a single subplot for Figure 4."""
    # Get layouts in order
    layouts = [l for l in PAPER_LAYOUTS if l in data]
    n_layouts = len(layouts)
    
    if n_layouts == 0:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        return
    
    # Bar positioning
    n_bar_configs = len(bar_configs)
    bar_width = 0.15
    x = np.arange(n_layouts)
    
    # Calculate offsets for bars
    # We have: hollow bar, 3 solid bars, then hatched versions
    # Total: 4 bars normal + 3 hatched = 7 groups
    
    legend_handles = []
    legend_labels = []
    
    # Plot hollow bar (self-play baseline) - leftmost
    hollow_config = bar_configs[0]  # sp_sp or pbt_pbt
    if hollow_config in config:
        means_0, stds_0 = _get_metrics(data, layouts, hollow_config, "order_0")
        means_1, stds_1 = _get_metrics(data, layouts, hollow_config, "order_1")
        
        offset = -2.5 * bar_width
        
        # Normal order (hollow)
        bars = ax.bar(
            x + offset, means_0, bar_width,
            color='none',
            edgecolor='gray',
            linewidth=1.5,
            linestyle=':',
            yerr=stds_0,
            capsize=2,
            error_kw={'elinewidth': 1, 'capthick': 1},
        )
        legend_handles.append(bars)
        legend_labels.append(config[hollow_config]["name"])
        
        # Switched order (hollow with hatch)
        ax.bar(
            x + offset + 4 * bar_width, means_1, bar_width,
            color='none',
            edgecolor='gray',
            linewidth=1.5,
            linestyle=':',
            hatch='///',
            yerr=stds_1,
            capsize=2,
            error_kw={'elinewidth': 1, 'capthick': 1},
        )
    
    # Plot solid bars
    solid_configs = bar_configs[1:]  # [sp_hp/pbt_hp, ppo_bc_hp, bc_hp]
    for i, cfg in enumerate(solid_configs):
        if cfg not in config or cfg not in data.get(layouts[0], {}):
            continue
            
        means_0, stds_0 = _get_metrics(data, layouts, cfg, "order_0")
        means_1, stds_1 = _get_metrics(data, layouts, cfg, "order_1")
        
        color = config[cfg]["color"]
        offset_0 = (-1.5 + i) * bar_width
        offset_1 = (1.5 + i) * bar_width
        
        # Normal order
        bars = ax.bar(
            x + offset_0, means_0, bar_width,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            yerr=stds_0,
            capsize=2,
            error_kw={'elinewidth': 1, 'capthick': 1},
        )
        legend_handles.append(bars)
        legend_labels.append(config[cfg]["name"])
        
        # Switched order (with hatch)
        ax.bar(
            x + offset_1, means_1, bar_width,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            hatch='///',
            yerr=stds_1,
            capsize=2,
            error_kw={'elinewidth': 1, 'capthick': 1},
        )
    
    # Plot gold standard line (PPO_HP + HP)
    if line_config in config:
        means, _ = _get_metrics(data, layouts, line_config, "order_0")
        
        for i, (layout, mean) in enumerate(zip(layouts, means)):
            if mean > 0:
                ax.hlines(
                    mean,
                    xmin=i - 0.4,
                    xmax=i + 0.4,
                    colors='red',
                    linestyles=':',
                    linewidth=2,
                )
        
        # Add to legend
        line_handle = plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2)
        legend_handles.insert(0, line_handle)
        legend_labels.insert(0, config[line_config]["name"])
    
    # Add switched indices legend entry
    hatch_patch = Patch(facecolor='white', edgecolor='gray', hatch='///', label='Switched indices')
    legend_handles.append(hatch_patch)
    legend_labels.append('Switched indices')
    
    # Configure axes
    ax.set_ylabel("Average reward per episode", fontsize=14)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYOUT_DISPLAY_NAMES.get(l, l) for l in layouts], fontsize=12)
    ax.set_ylim(0, 250)
    
    # Add subtitle below the plot
    ax.text(0.5, -0.15, subtitle, ha='center', va='top', transform=ax.transAxes, fontsize=12)
    
    # Legend
    ax.legend(
        legend_handles, legend_labels,
        loc='upper right',
        fontsize=10,
        framealpha=0.9,
        ncol=2,
    )
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)


def _get_metrics(
    data: Dict[str, Dict],
    layouts: List[str],
    config_name: str,
    order_key: str,
) -> Tuple[List[float], List[float]]:
    """Extract means and standard errors from results."""
    means = []
    stds = []
    
    for layout in layouts:
        if layout not in data or config_name not in data[layout]:
            means.append(0)
            stds.append(0)
            continue
        
        result = data[layout][config_name].get(order_key, {})
        if isinstance(result, dict) and "error" not in result:
            means.append(result.get("mean_reward", 0))
            stds.append(result.get("stderr_reward", 0))
        else:
            means.append(0)
            stds.append(0)
    
    return means, stds


def plot_training_curves(
    training_data: Dict[str, List[float]],
    output_path: str = "training_curves.pdf",
    title: str = "Training Progress",
    xlabel: str = "Training Iterations",
    ylabel: str = "Episode Reward",
    show: bool = False,
):
    """
    Plot training curves.
    
    Args:
        training_data: Dictionary mapping experiment names to reward lists
        output_path: Path to save figure
        title: Figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        show: Whether to display the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2d6777', '#F79646', '#7f7f7f', '#4472C4', '#70AD47']
    
    for i, (name, rewards) in enumerate(training_data.items()):
        iterations = list(range(len(rewards)))
        color = colors[i % len(colors)]
        ax.plot(iterations, rewards, label=name, linewidth=2, color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format=output_path.split(".")[-1], bbox_inches="tight", dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Saved figure to {output_path}")


def plot_layout_comparison(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "layout_comparison.pdf",
    show: bool = False,
):
    """
    Plot simple comparison across layouts for a single agent type.
    
    Args:
        results: Dictionary mapping layouts to results
        output_path: Path to save figure
        show: Whether to display the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    set_paper_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layouts = list(results.keys())
    means = [results[l].get("mean_reward", 0) for l in layouts]
    stds = [results[l].get("stderr_reward", 0) for l in layouts]
    
    x = np.arange(len(layouts))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#2d6777", edgecolor='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels([LAYOUT_DISPLAY_NAMES.get(l, l) for l in layouts])
    ax.set_ylabel("Average Reward")
    ax.set_title("Performance Across Layouts")
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format=output_path.split(".")[-1], bbox_inches="tight", dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-style Figure 4 from evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--results_file",
        type=str,
        default="paper_results.json",
        help="Path to evaluation results JSON file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save figures"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "jpg", "svg"],
        help="Output format for figures"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures instead of just saving"
    )
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting")
        print("Install with: pip install matplotlib")
        return
    
    # Load results
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        print("Run evaluation first:")
        print("  python -m human_aware_rl.evaluation.evaluate_paper --output_file paper_results.json")
        return
    
    with open(args.results_file, "r") as f:
        results = json.load(f)
    
    # Generate Figure 4
    plot_figure_4(
        results=results,
        output_dir=args.output_dir,
        fmt=args.format,
        show=args.show,
    )
    
    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
