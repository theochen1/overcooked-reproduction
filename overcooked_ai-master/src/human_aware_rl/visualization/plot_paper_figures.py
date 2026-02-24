"""Plot helpers for paper reproduction Figures 5-7."""

import os
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def _require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting paper figures")


def _save_multi_format(fig, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    root, ext = os.path.splitext(output_path)
    ext = ext.lower().lstrip(".")
    if ext:
        fig.savefig(output_path, format=ext, bbox_inches="tight", dpi=180)
    else:
        fig.savefig(root + ".pdf", format="pdf", bbox_inches="tight", dpi=180)
        fig.savefig(root + ".png", format="png", bbox_inches="tight", dpi=180)


def plot_figure_5(data: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    """Planning comparison bars (Figure 5 style)."""
    _require_matplotlib()
    layouts = list(data.keys())
    algos = ["CP+CP", "CP+HP", "P_BC+HP", "BC+HP"]
    x = np.arange(len(layouts))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(algos))
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, algo in enumerate(algos):
        means = [data[l].get(algo, {}).get("mean", np.nan) for l in layouts]
        errs = [data[l].get(algo, {}).get("stderr", 0.0) for l in layouts]
        ax.bar(x + offsets[i], means, width=width, label=algo, yerr=errs)
    # Gold-standard horizontal reference if present per layout.
    for idx, layout in enumerate(layouts):
        ref = data[layout].get("P_HProxy+HProxy", {}).get("mean")
        if ref is not None:
            ax.hlines(ref, xmin=idx - 0.45, xmax=idx + 0.45, colors="red", linestyles="dotted")
    ax.set_title("Figure 5: Planning Comparison")
    ax.set_ylabel("Average reward per episode")
    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=20, ha="right")
    ax.legend(loc="best")
    _save_multi_format(fig, output_path)
    plt.close(fig)


def _plot_metric_grid(
    data: Dict[str, Dict[str, Dict[str, float]]],
    metric_mean_key: str,
    metric_err_key: str,
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    _require_matplotlib()
    layouts = list(data.keys())
    model_names: List[str] = sorted(
        {model_name for layout in layouts for model_name in data[layout].keys()}
    )
    x = np.arange(len(layouts))
    width = max(0.08, min(0.22, 0.8 / max(1, len(model_names))))
    offsets = np.linspace(-(len(model_names) - 1) * width / 2, (len(model_names) - 1) * width / 2, len(model_names))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(model_names):
        means = [data[l].get(model_name, {}).get(metric_mean_key, np.nan) for l in layouts]
        errs = [data[l].get(model_name, {}).get(metric_err_key, 0.0) for l in layouts]
        ax.bar(x + offsets[i], means, width=width, yerr=errs, label=model_name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=20, ha="right")
    ax.legend(loc="best", ncol=2, fontsize=9)
    _save_multi_format(fig, output_path)
    plt.close(fig)


def plot_figure_6(data: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    """Off-distribution cross-entropy loss (Figure 6 style)."""
    _plot_metric_grid(
        data=data,
        metric_mean_key="loss_mean",
        metric_err_key="loss_stderr",
        title="Figure 6: Off-distribution Loss",
        ylabel="Cross-entropy loss",
        output_path=output_path,
    )


def plot_figure_7(data: Dict[str, Dict[str, Dict[str, float]]], output_path: str) -> None:
    """Off-distribution action accuracy (Figure 7 style)."""
    _plot_metric_grid(
        data=data,
        metric_mean_key="accuracy_mean",
        metric_err_key="accuracy_stderr",
        title="Figure 7: Off-distribution Accuracy",
        ylabel="Accuracy",
        output_path=output_path,
    )

