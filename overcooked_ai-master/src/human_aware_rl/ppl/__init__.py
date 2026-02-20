"""
Probabilistic Programming Language (PPL) based models for Overcooked AI.

This module provides PPL-based alternatives to traditional neural network
approaches for behavior cloning and imitation learning.

Available models:
- BayesianBC: Bayesian neural network for behavior cloning
- RationalAgent: Softmax-rational model with explicit planning
- HierarchicalBC: Hierarchical model with goal inference
- WebPPLBridge: Interface for WebPPL models (for cognitive modeling)

Requirements:
    pip install pyro-ppl numpyro
"""

__all__ = []

# Lazy imports to avoid import errors if pyro isn't installed
try:
    from human_aware_rl.ppl.bayesian_bc import (
        BayesianBCModel,
        train_bayesian_bc,
        BayesianBCAgent,
        BayesianBCConfig,
    )
    __all__.extend([
        "BayesianBCModel",
        "train_bayesian_bc",
        "BayesianBCAgent",
        "BayesianBCConfig",
    ])
except ImportError as e:
    print(f"Warning: Could not import bayesian_bc: {e}")
    print("Install with: pip install pyro-ppl")

try:
    from human_aware_rl.ppl.rational_agent import (
        RationalAgentModel,
        train_rational_agent,
        RationalAgent,
        RationalAgentConfig,
    )
    __all__.extend([
        "RationalAgentModel",
        "train_rational_agent",
        "RationalAgent",
        "RationalAgentConfig",
    ])
except ImportError as e:
    print(f"Warning: Could not import rational_agent: {e}")

try:
    from human_aware_rl.ppl.hierarchical_bc import (
        HierarchicalBCModel,
        train_hierarchical_bc,
        HierarchicalBCAgent,
        HierarchicalBCConfig,
    )
    __all__.extend([
        "HierarchicalBCModel",
        "train_hierarchical_bc",
        "HierarchicalBCAgent",
        "HierarchicalBCConfig",
    ])
except ImportError as e:
    print(f"Warning: Could not import hierarchical_bc: {e}")

try:
    from human_aware_rl.ppl.webppl_bridge import (
        WebPPLModel,
        WebPPLAgent,
        SoftmaxRationalModel,
        GoalInferenceModel,
        create_webppl_agent,
    )
    __all__.extend([
        "WebPPLModel",
        "WebPPLAgent",
        "SoftmaxRationalModel",
        "GoalInferenceModel",
        "create_webppl_agent",
    ])
except ImportError as e:
    pass  # WebPPL bridge doesn't require pyro
