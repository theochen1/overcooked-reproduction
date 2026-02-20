"""
Human-Aware Reinforcement Learning for Overcooked AI.

This module provides tools for training AI agents that coordinate with humans
in the Overcooked environment, as described in:

"On the Utility of Learning about Humans for Human-AI Coordination"
Carroll et al., NeurIPS 2019
https://arxiv.org/abs/1910.05789

Main components:
- imitation: Behavior cloning from human demonstrations (PyTorch)
- jaxmarl: JAX-based multi-agent RL training
- bridge: Utilities for loading and evaluating trained policies
- human: Human data processing utilities

Installation:
    # For behavior cloning only
    pip install ".[bc]"
    
    # For full training stack (recommended)
    pip install ".[harl]"
    
    # With CUDA support
    pip install ".[harl-cuda]"
"""

__version__ = "2.0.0"

# Import main components
from human_aware_rl.data_dir import DATA_DIR

__all__ = [
    "DATA_DIR",
    "__version__",
]

