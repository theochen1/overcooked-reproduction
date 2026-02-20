"""
Evaluation utilities for Overcooked AI agents.
"""

__all__ = []

# Import from evaluate_all if available
try:
    from human_aware_rl.evaluation.evaluate_all import (
        evaluate_bc_self_play,
        evaluate_ppo_self_play,
        evaluate_bc_with_ppo,
        evaluate_agent_with_human_proxy,
        run_all_evaluations,
    )
    __all__.extend([
        "evaluate_bc_self_play",
        "evaluate_ppo_self_play",
        "evaluate_bc_with_ppo",
        "evaluate_agent_with_human_proxy",
        "run_all_evaluations",
    ])
except ImportError:
    pass

# Import from evaluate_paper if available
try:
    from human_aware_rl.evaluation.evaluate_paper import EVALUATION_CONFIGS
    __all__.append("EVALUATION_CONFIGS")
except ImportError:
    pass

# Import from evaluate_bc_hp if available
try:
    from human_aware_rl.evaluation.evaluate_bc_hp import (
        evaluate_bc_vs_hp,
        evaluate_all_layouts,
    )
    __all__.extend([
        "evaluate_bc_vs_hp",
        "evaluate_all_layouts",
    ])
except ImportError:
    pass

