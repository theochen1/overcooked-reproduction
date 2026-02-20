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

__all__ = [
    "plot_paper_results",
    "plot_training_curves",
    "set_paper_style",
    "PLOTTING_AVAILABLE",
]

