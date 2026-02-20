"""
Imitation learning module for Overcooked AI.

This module provides imitation learning functionality for training
agents from human demonstration data.

Main components:
- behavior_cloning: PyTorch-based BC training
- bc_agent: Agent wrapper for BC models
- airl: Adversarial Inverse Reinforcement Learning (AIRL) training
- airl_agent: Agent wrapper for AIRL-trained policies
- behavior_cloning_tf2: Legacy TensorFlow implementation (deprecated)
"""

# Import PyTorch BC components (recommended)
try:
    from human_aware_rl.imitation.behavior_cloning import (
        BC_SAVE_DIR,
        BCModel,
        BCLSTMModel,
        build_bc_model,
        get_bc_params,
        load_bc_model,
        load_data,
        save_bc_model,
        train_bc_model,
        evaluate_bc_model,
    )
    from human_aware_rl.imitation.bc_agent import (
        BCAgent,
        BehaviorCloningPolicy,
    )
    PYTORCH_BC_AVAILABLE = True
except ImportError:
    PYTORCH_BC_AVAILABLE = False

# Import AIRL components
try:
    from human_aware_rl.imitation.airl import (
        AIRL_SAVE_DIR,
        AIRLConfig,
        AIRLDiscriminator,
        AIRLPolicy,
        AIRLPolicyLSTM,
        AIRLTrainer,
        load_airl_model,
        save_airl_model,
    )
    from human_aware_rl.imitation.airl_agent import (
        AIRLAgent,
    )
    AIRL_AVAILABLE = True
except ImportError:
    AIRL_AVAILABLE = False

__all__ = [
    # BC
    "BC_SAVE_DIR",
    "BCModel",
    "BCLSTMModel",
    "BCAgent",
    "BehaviorCloningPolicy",
    "build_bc_model",
    "get_bc_params",
    "load_bc_model",
    "load_data",
    "save_bc_model",
    "train_bc_model",
    "evaluate_bc_model",
    "PYTORCH_BC_AVAILABLE",
    # AIRL
    "AIRL_SAVE_DIR",
    "AIRLConfig",
    "AIRLDiscriminator",
    "AIRLPolicy",
    "AIRLPolicyLSTM",
    "AIRLTrainer",
    "AIRLAgent",
    "load_airl_model",
    "save_airl_model",
    "AIRL_AVAILABLE",
]

