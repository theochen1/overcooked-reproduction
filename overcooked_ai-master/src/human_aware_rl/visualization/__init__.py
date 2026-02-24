"""
Visualization utilities for Overcooked AI experiments.
"""

# Only import plotting functions if matplotlib is available
try:
    from human_aware_rl.visualization.plot_results import (
        plot_paper_results,
        plot_training_curves,
        set_paper_style,
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plot_paper_results = None
    plot_training_curves = None
    set_paper_style = None

try:
    from human_aware_rl.visualization.plot_paper_figures import (
        plot_figure_5,
        plot_figure_6,
        plot_figure_7,
    )
except ImportError:
    plot_figure_5 = None
    plot_figure_6 = None
    plot_figure_7 = None

__all__ = [
    "plot_paper_results",
    "plot_training_curves",
    "set_paper_style",
    "plot_figure_5",
    "plot_figure_6",
    "plot_figure_7",
    "PLOTTING_AVAILABLE",
]

