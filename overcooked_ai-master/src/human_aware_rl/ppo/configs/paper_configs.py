"""
Paper hyperparameter configurations for PPO training.

These configurations match the hyperparameters used in:
"On the Utility of Learning about Humans for Human-AI Coordination"
by Micah Carroll et al.

Tables 3 and 4 from the paper appendix are encoded here.
"""

from typing import Dict, Any, List, Tuple

# Layout mapping from paper names to environment names
# IMPORTANT: Use legacy layout files which have the correct MDP parameters
# (cook_time=20, num_items_for_soup=3, delivery_reward=20)
PAPER_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring", 
    "forced_coordination",
    "counter_circuit",
]

LAYOUT_TO_ENV = {
    # All layouts use legacy versions with correct paper MDP parameters:
    # cook_time=20, num_items_for_soup=3, delivery_reward=20
    "cramped_room": "cramped_room_legacy",
    "asymmetric_advantages": "asymmetric_advantages_legacy",
    "coordination_ring": "coordination_ring_legacy",
    "forced_coordination": "random0_legacy",
    "counter_circuit": "random3_legacy",
}

# Common parameters across all experiments (from paper)
# Paper Tables 2 & 3: batch_size = num_minibatches * minibatch_size = 6 * 2000 = 12,000
# This implies 30 parallel environments (30 envs * 400 steps = 12,000)
PAPER_COMMON_PARAMS = {
    # Network architecture
    "num_hidden_layers": 3,
    "hidden_dim": 64,
    "num_filters": 25,
    "num_conv_layers": 3,
    "use_lstm": False,
    "cell_size": 256,
    
    # Training batch settings
    # Paper Table 2: num_minibatches=6, minibatch_size=2000 -> batch=12,000
    # 30 envs * 400 steps = 12,000
    "train_batch_size": 12000,
    "num_minibatches": 6,
    "rollout_fragment_length": 400,
    "num_sgd_iter": 8,
    
    # Entropy coefficient
    # Original ppo.py default: ENTROPY = 0.1 (line 101)
    # No experiment script overrides this, so all layouts use 0.1
    # Previously incorrectly "corrected" to 0.01 based on baselines default,
    # but the Overcooked code explicitly sets ENTROPY = 0.1
    "entropy_coeff_start": 0.1,
    "entropy_coeff_end": 0.1,
    "entropy_coeff_horizon": 0,   # No annealing
    "use_entropy_annealing": False,
    
    # Episode settings
    "horizon": 400,
    "old_dynamics": True,  # Paper uses old dynamics (cooking starts automatically)
    
    # Reward shaping (paper uses shaped rewards)
    "use_phi": False,  # Paper trains without potential-based shaping
    "reward_shaping_factor": 1.0,
    
    # Number of parallel workers
    # Paper Table 2: batch=12,000 / 400 steps = 30 envs
    "num_workers": 30,
    
    # Observation encoding
    "use_legacy_encoding": True,  # Paper uses 20-channel legacy encoding
    
    # Evaluation
    "evaluation_interval": 50,
    "evaluation_num_games": 50,
}

# Entropy coefficient configuration
# Original ppo.py sets ENTROPY = 0.1 as the default.
# No experiment script overrides ENTROPY, so all layouts use 0.1.
# The baselines default (0.01 for atari) is NOT used -- the Overcooked code
# explicitly passes ENTROPY=0.1 to the learn() function.
LAYOUT_ENTROPY_CONFIGS = {
    "cramped_room": {
        "entropy_coeff_start": 0.1,
        "entropy_coeff_end": 0.1,
        "entropy_coeff_horizon": 0,
    },
    "asymmetric_advantages": {
        "entropy_coeff_start": 0.1,
        "entropy_coeff_end": 0.1,
        "entropy_coeff_horizon": 0,
    },
    "coordination_ring": {
        "entropy_coeff_start": 0.1,
        "entropy_coeff_end": 0.1,
        "entropy_coeff_horizon": 0,
    },
    "forced_coordination": {
        "entropy_coeff_start": 0.1,
        "entropy_coeff_end": 0.1,
        "entropy_coeff_horizon": 0,
    },
    "counter_circuit": {
        "entropy_coeff_start": 0.1,
        "entropy_coeff_end": 0.1,
        "entropy_coeff_horizon": 0,
    },
}


# PPO Self-Play Hyperparameters (per-layout) -- Paper Table 2 + original scripts
# batch_size = 6 * 2000 = 12,000 (30 envs * 400 steps)
# Total timesteps and VF_COEF from original ppo_sp_experiments.sh (ground truth)
# NOTE: Paper Table 2 says VF=0.5 for cramped_room, but the original script
# actually uses VF_COEF=1. We follow the original code as ground truth.
PAPER_PPO_SP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cramped_room": {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 1.0,                 # Original script: VF_COEF=1 (paper says 0.5)
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 2.5e6,
        "total_timesteps": 6_000_000,    # Original: PPO_RUN_TOT_TIMESTEPS=6e6
        "num_training_iters": 500,       # 6M / 12000 = 500
    },
    "asymmetric_advantages": {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.5,
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 2.5e6,
        "total_timesteps": 7_000_000,    # Original: PPO_RUN_TOT_TIMESTEPS=7e6
        "num_training_iters": 583,       # 7M / 12000 = 583
    },
    "coordination_ring": {
        "learning_rate": 6e-4,
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.5,
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 3.5e6,
        "total_timesteps": 10_000_000,   # Original: PPO_RUN_TOT_TIMESTEPS=1e7
        "num_training_iters": 833,       # 10M / 12000 = 833
    },
    "forced_coordination": {
        "learning_rate": 8e-4,
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.5,
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 2.5e6,
        "total_timesteps": 7_500_000,    # Original: PPO_RUN_TOT_TIMESTEPS=7.5e6
        "num_training_iters": 625,       # 7.5M / 12000 = 625
    },
    "counter_circuit": {
        "learning_rate": 8e-4,
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.5,
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 2.5e6,
        "total_timesteps": 8_000_000,    # Original: PPO_RUN_TOT_TIMESTEPS=8e6
        "num_training_iters": 667,       # 8M / 12000 = 667
    },
}


# Table 4: PBT Hyperparameters (per-layout)
# PBT-specific settings differ from self-play
PAPER_PBT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cramped_room": {
        "learning_rate": 2e-3,
        "reward_shaping_horizon": 3e6,
        "total_env_steps": 8e6,
    },
    "asymmetric_advantages": {
        "learning_rate": 8e-4,
        "reward_shaping_horizon": 5e6,
        "total_env_steps": 1.1e7,
    },
    "coordination_ring": {
        "learning_rate": 8e-4,
        "reward_shaping_horizon": 4e6,
        "total_env_steps": 5e6,
    },
    "forced_coordination": {
        "learning_rate": 3e-3,
        "reward_shaping_horizon": 7e6,
        "total_env_steps": 8e6,
    },
    "counter_circuit": {
        "learning_rate": 1e-3,
        "reward_shaping_horizon": 4e6,
        "total_env_steps": 6e6,
    },
}

# PBT common parameters
PBT_COMMON_PARAMS = {
    "population_size": 8,
    "ppo_iteration_timesteps": 40000,
    "num_minibatches": 10,
    "minibatch_size": 2000,
    
    # Mutation parameters
    "mutation_prob": 0.33,  # 33% chance of mutation
    "mutation_factor_low": 0.75,
    "mutation_factor_high": 1.25,
    
    # Parameters that can be mutated
    "mutable_params": ["learning_rate", "entropy_coeff", "vf_coef", "gae_lambda"],
    
    # Initial ranges for mutable parameters
    "initial_entropy_coeff": 0.5,
    "initial_vf_coef": 0.1,
}


# PPO_BC / PPO_HP configurations (PPO trained with BC or Human Proxy partner)
# Paper Table 3 + total_timesteps from original ppo_bc_experiments.sh
# Key differences from SP: per-layout LR, LR annealing factor, VF coef, reward
# shaping horizon, self-play annealing schedule, num_minibatches.
#
# Batch size: num_minibatches * minibatch_size = 12,000 for all layouts
# This implies 30 parallel environments (30 envs * 400 steps = 12,000)
PAPER_PPO_BC_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cramped_room": {
        "learning_rate": 1e-3,
        "lr_annealing_factor": 3,       # LR decays from 1e-3 to 3.33e-4
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.5,
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 1e6,
        "num_minibatches": 10,          # minibatch_size=1200
        "total_timesteps": 8_000_000,   # Original: PPO_RUN_TOT_TIMESTEPS=8e6
        # SELF_PLAY_HORIZON=[5e5, 3e6]:
        # Original: self_play_randomization starts at 1.0 (100% self-play), anneals to 0 (100% BC)
        # bc_factor = 1 - self_play_randomization: starts 0.0 (self-play), ends 1.0 (BC)
        "bc_schedule": [
            (0, 0.0),       # 0% BC (100% self-play)
            (5e5, 0.0),     # Still 100% self-play
            (3e6, 1.0),     # Fully transitioned to 100% BC
        ],
    },
    "asymmetric_advantages": {
        "learning_rate": 1e-3,
        "lr_annealing_factor": 3,       # LR decays from 1e-3 to 3.33e-4
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.5,
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 6e6,
        "num_minibatches": 12,          # minibatch_size=1000
        "total_timesteps": 10_000_000,  # Original: PPO_RUN_TOT_TIMESTEPS=1e7
        # SELF_PLAY_HORIZON=[1e6, 7e6]:
        # self-play first, then transition to BC
        "bc_schedule": [
            (0, 0.0),       # 0% BC (100% self-play)
            (1e6, 0.0),     # Still 100% self-play
            (7e6, 1.0),     # Fully transitioned to 100% BC
        ],
    },
    "coordination_ring": {
        "learning_rate": 1e-3,
        "lr_annealing_factor": 1.5,     # LR decays from 1e-3 to 6.67e-4
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.5,
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 5e6,
        "num_minibatches": 15,          # minibatch_size=800
        "total_timesteps": 16_000_000,  # Original: PPO_RUN_TOT_TIMESTEPS=1.6e7
        # SELF_PLAY_HORIZON=[2e6, 6e6]:
        # self-play first, then transition to BC
        "bc_schedule": [
            (0, 0.0),       # 0% BC (100% self-play)
            (2e6, 0.0),     # Still 100% self-play
            (6e6, 1.0),     # Fully transitioned to 100% BC
        ],
    },
    "forced_coordination": {
        "learning_rate": 1.5e-3,
        "lr_annealing_factor": 2,       # LR decays from 1.5e-3 to 7.5e-4
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.1,                 # NOT 0.5!
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 4e6,
        "num_minibatches": 15,          # minibatch_size=800
        "total_timesteps": 9_000_000,   # Original: PPO_RUN_TOT_TIMESTEPS=9e6
        # SELF_PLAY_HORIZON=None: self_play_randomization=0 → always BC
        "bc_schedule": [
            (0, 1.0),
            (float('inf'), 1.0),
        ],
    },
    "counter_circuit": {
        "learning_rate": 1.5e-3,
        "lr_annealing_factor": 3,       # LR decays from 1.5e-3 to 5e-4
        "gamma": 0.99,
        "clip_eps": 0.05,
        "max_grad_norm": 0.1,
        "gae_lambda": 0.98,
        "vf_coef": 0.1,                 # NOT 0.5!
        "kl_coeff": 0.2,
        "reward_shaping_horizon": 4e6,
        "num_minibatches": 15,          # minibatch_size=800
        "total_timesteps": 12_000_000,  # Original: PPO_RUN_TOT_TIMESTEPS=1.2e7
        # SELF_PLAY_HORIZON=[1e6, 4e6]:
        # self-play first, then transition to BC
        "bc_schedule": [
            (0, 0.0),       # 0% BC (100% self-play)
            (1e6, 0.0),     # Still 100% self-play
            (4e6, 1.0),     # Fully transitioned to 100% BC
        ],
    },
}

# PPO_BC common params (overrides from PAPER_COMMON_PARAMS)
# Paper Table 3 uses 30 envs (batch_size=12,000 = 30 * 400)
PAPER_PPO_BC_COMMON = {
    "num_workers": 30,       # 30 envs * 400 steps = 12,000 batch (Paper Table 3)
    "train_batch_size": 12000,
}


def get_ppo_sp_config(layout: str, seed: int = 0, **overrides) -> Dict[str, Any]:
    """
    Get PPO self-play configuration for a layout.
    
    Args:
        layout: Layout name (paper name, e.g., 'cramped_room')
        seed: Random seed
        **overrides: Additional parameter overrides
        
    Returns:
        Configuration dictionary
    """
    if layout not in PAPER_PPO_SP_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}. Available: {list(PAPER_PPO_SP_CONFIGS.keys())}")
    
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Start with common params
    config = {
        **PAPER_COMMON_PARAMS,
        **PAPER_PPO_SP_CONFIGS[layout],
        "layout_name": env_layout,
        "seed": seed,
        "experiment_name": f"ppo_sp_{layout}_seed{seed}",
        "bc_schedule": [(0, 0.0), (float('inf'), 0.0)],  # No BC partner
    }
    
    # Apply layout-specific entropy configs (Table 2: fixed at 0.1, no annealing)
    if layout in LAYOUT_ENTROPY_CONFIGS:
        entropy_config = LAYOUT_ENTROPY_CONFIGS[layout]
        config["entropy_coeff_start"] = entropy_config["entropy_coeff_start"]
        config["entropy_coeff_end"] = entropy_config["entropy_coeff_end"]
        config["entropy_coeff_horizon"] = entropy_config["entropy_coeff_horizon"]
        # Disable entropy annealing since it's fixed (Table 2)
        config["use_entropy_annealing"] = False
    
    # Total timesteps: use per-layout value from original scripts
    # (already set in PAPER_PPO_SP_CONFIGS via "total_timesteps" key)
    if "total_timesteps" not in config:
        config["total_timesteps"] = 10_000_000  # Fallback
    
    config.update(overrides)
    return config


def get_pbt_config(layout: str, **overrides) -> Dict[str, Any]:
    """
    Get PBT configuration for a layout.
    
    Args:
        layout: Layout name (paper name)
        **overrides: Additional parameter overrides
        
    Returns:
        Configuration dictionary
    """
    if layout not in PAPER_PBT_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}. Available: {list(PAPER_PBT_CONFIGS.keys())}")
    
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    config = {
        **PAPER_COMMON_PARAMS,
        **PBT_COMMON_PARAMS,
        **PAPER_PBT_CONFIGS[layout],
        "layout_name": env_layout,
        "experiment_name": f"pbt_{layout}",
    }
    
    config.update(overrides)
    return config


def get_ppo_bc_config(layout: str, seed: int = 0, bc_model_dir: str = None, **overrides) -> Dict[str, Any]:
    """
    Get PPO_BC configuration for a layout (Paper Table 3).
    
    PPO_BC has DISTINCT hyperparameters from PPO_SP, including per-layout
    learning rates, LR annealing factors, VF coefficients, reward shaping
    horizons, BC schedules, and minibatch settings.
    
    Args:
        layout: Layout name (paper name)
        seed: Random seed
        bc_model_dir: Path to BC model directory
        **overrides: Additional parameter overrides
        
    Returns:
        Configuration dictionary
    """
    if layout not in PAPER_PPO_BC_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}. Available: {list(PAPER_PPO_BC_CONFIGS.keys())}")
    
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Start with common params, then override with BC-specific common and per-layout
    config = {
        **PAPER_COMMON_PARAMS,
        **PAPER_PPO_BC_COMMON,       # Override num_workers=30, batch=12000
        **PAPER_PPO_BC_CONFIGS[layout],  # Per-layout Table 3 values
        "layout_name": env_layout,
        "seed": seed,
        "experiment_name": f"ppo_bc_{layout}_seed{seed}",
        "bc_model_dir": bc_model_dir,
        # PPO_BC uses LR annealing (factor-based, Paper Table 3)
        "use_lr_annealing": True,
    }
    
    # Apply layout-specific entropy config (same as SP: fixed at 0.1)
    if layout in LAYOUT_ENTROPY_CONFIGS:
        entropy_config = LAYOUT_ENTROPY_CONFIGS[layout]
        config["entropy_coeff_start"] = entropy_config["entropy_coeff_start"]
        config["entropy_coeff_end"] = entropy_config["entropy_coeff_end"]
        config["entropy_coeff_horizon"] = entropy_config["entropy_coeff_horizon"]
        config["use_entropy_annealing"] = False
    
    # Total timesteps: use per-layout value from original scripts
    # (already set in PAPER_PPO_BC_CONFIGS via "total_timesteps" key)
    if "total_timesteps" not in config:
        config["total_timesteps"] = 10_000_000  # Fallback
    
    config.update(overrides)
    return config


# =============================================================================
# PPO_GAIL Configurations
# =============================================================================

# PPO_GAIL_controlled: Uses EXACT same HPs as PPO_BC (Paper Table 3),
# only swapping the BC partner for a GAIL partner.
# This isolates the partner model quality as the sole independent variable.
PAPER_PPO_GAIL_CONFIGS = PAPER_PPO_BC_CONFIGS  # Same per-layout HPs
PAPER_PPO_GAIL_COMMON = PAPER_PPO_BC_COMMON    # Same batch/env settings


def get_ppo_gail_config(layout: str, seed: int = 0, gail_model_dir: str = None,
                        **overrides) -> Dict[str, Any]:
    """
    Get PPO_GAIL_controlled configuration for a layout.

    Uses the EXACT same hyperparameters as PPO_BC (Paper Table 3) so that the
    only experimental variable is the partner model (GAIL instead of BC).

    Args:
        layout: Layout name (paper name)
        seed: Random seed
        gail_model_dir: Path to GAIL model directory
        **overrides: Additional parameter overrides

    Returns:
        Configuration dictionary
    """
    if layout not in PAPER_PPO_GAIL_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}. Available: {list(PAPER_PPO_GAIL_CONFIGS.keys())}")

    env_layout = LAYOUT_TO_ENV.get(layout, layout)

    # Identical to get_ppo_bc_config except experiment_name and model_dir key
    config = {
        **PAPER_COMMON_PARAMS,
        **PAPER_PPO_GAIL_COMMON,
        **PAPER_PPO_GAIL_CONFIGS[layout],
        "layout_name": env_layout,
        "seed": seed,
        "experiment_name": f"ppo_gail_{layout}_seed{seed}",
        "gail_model_dir": gail_model_dir,
        "use_lr_annealing": True,
    }

    # Entropy: fixed at 0.1, same as SP and BC
    if layout in LAYOUT_ENTROPY_CONFIGS:
        entropy_config = LAYOUT_ENTROPY_CONFIGS[layout]
        config["entropy_coeff_start"] = entropy_config["entropy_coeff_start"]
        config["entropy_coeff_end"] = entropy_config["entropy_coeff_end"]
        config["entropy_coeff_horizon"] = entropy_config["entropy_coeff_horizon"]
        config["use_entropy_annealing"] = False

    # GAIL controlled uses same total_timesteps as PPO_BC (per-layout from original scripts)
    if "total_timesteps" not in config:
        config["total_timesteps"] = 10_000_000  # Fallback

    config.update(overrides)
    return config


# PPO_GAIL_optimized: Bayesian-optimized HPs, but with controlled training
# budget (10M steps) and proper reward shaping horizon (from Table 3).
# Used as an ablation to show the ceiling of GAIL with tuned HPs.
BAYESIAN_OPTIMIZED_GAIL_PARAMS = {
    "learning_rate": 1.63e-4,
    "gamma": 0.964,
    "gae_lambda": 0.6,
    "clip_eps": 0.132,
    "vf_coef": 0.00995,
    "max_grad_norm": 0.247,
    "kl_coeff": 0.197,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.1,
    "entropy_coeff_horizon": 3e5,
    "use_entropy_annealing": True,
    "num_minibatches": 10,
}


def get_ppo_gail_optimized_config(layout: str, seed: int = 0,
                                  gail_model_dir: str = None,
                                  **overrides) -> Dict[str, Any]:
    """
    Get PPO_GAIL_optimized configuration (ablation: Bayesian-optimized HPs).

    Uses Bayesian-optimized hyperparameters but matches PPO_BC on:
    - Total timesteps (10M)
    - Reward shaping horizon (from Paper Table 3, per layout)
    - Partner schedule (from Paper Table 3, per layout)

    This isolates the HP-tuning effect from the training-budget effect.

    Args:
        layout: Layout name
        seed: Random seed
        gail_model_dir: Path to GAIL model directory
        **overrides: Additional parameter overrides
    """
    if layout not in PAPER_PPO_BC_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}")

    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    bc_layout_cfg = PAPER_PPO_BC_CONFIGS[layout]

    config = {
        **PAPER_COMMON_PARAMS,
        **BAYESIAN_OPTIMIZED_GAIL_PARAMS,
        "layout_name": env_layout,
        "seed": seed,
        "experiment_name": f"ppo_gail_opt_{layout}_seed{seed}",
        "gail_model_dir": gail_model_dir,
        # Structural controls: match PPO_BC so only HPs differ
        "num_workers": 32,  # Bayesian config used 32 envs
        "reward_shaping_horizon": bc_layout_cfg["reward_shaping_horizon"],
        "bc_schedule": bc_layout_cfg["bc_schedule"],
        "total_timesteps": 10_000_000,
        "use_lr_annealing": False,  # Bayesian config used constant LR
    }

    config.update(overrides)
    return config


# PPO_SP_optimized: Bayesian-optimized HPs with self-play (no partner).
# Ablation to separate the HP-tuning effect from the GAIL-partner effect.
def get_ppo_sp_optimized_config(layout: str, seed: int = 0,
                                **overrides) -> Dict[str, Any]:
    """
    Get PPO_SP with Bayesian-optimized HPs (ablation).

    Tests whether the Bayesian-optimized HPs alone (without any partner model)
    explain the performance improvement. If SP_optimized matches GAIL_optimized,
    the HPs are the driver. If SP_optimized << GAIL_optimized, GAIL is the key.

    Args:
        layout: Layout name
        seed: Random seed
        **overrides: Additional parameter overrides
    """
    if layout not in PAPER_PPO_SP_CONFIGS:
        raise ValueError(f"Unknown layout: {layout}")

    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    sp_layout_cfg = PAPER_PPO_SP_CONFIGS[layout]

    config = {
        **PAPER_COMMON_PARAMS,
        **BAYESIAN_OPTIMIZED_GAIL_PARAMS,
        "layout_name": env_layout,
        "seed": seed,
        "experiment_name": f"ppo_sp_opt_{layout}_seed{seed}",
        "num_workers": 32,
        "reward_shaping_horizon": sp_layout_cfg["reward_shaping_horizon"],
        "bc_schedule": [(0, 0.0), (float('inf'), 0.0)],  # No partner
        "total_timesteps": 10_000_000,
        "use_lr_annealing": False,
    }

    config.update(overrides)
    return config


def print_config_summary():
    """Print a summary of all paper configurations."""
    print("="*80)
    print("Paper PPO Self-Play Configurations")
    print("="*80)
    
    headers = ["Layout", "LR", "Gamma", "Clip", "GradClip", "Lambda", "VF", "RewHorizon", "Iters"]
    row_format = "{:<20}" + "{:<10}" * (len(headers) - 1)
    
    print(row_format.format(*headers))
    print("-"*80)
    
    for layout in PAPER_LAYOUTS:
        cfg = PAPER_PPO_SP_CONFIGS[layout]
        print(row_format.format(
            layout,
            f"{cfg['learning_rate']:.2e}",
            f"{cfg['gamma']:.3f}",
            f"{cfg['clip_eps']:.3f}",
            f"{cfg['max_grad_norm']:.3f}",
            f"{cfg['gae_lambda']:.1f}",
            f"{cfg['vf_coef']:.2e}",
            f"{cfg['reward_shaping_horizon']:.0e}",
            str(cfg['num_training_iters']),
        ))
    
    print("\n")
    print("="*80)
    print("Paper PBT Configurations")
    print("="*80)
    
    headers = ["Layout", "LR", "RewHorizon", "TotalSteps"]
    row_format = "{:<20}" + "{:<15}" * (len(headers) - 1)
    
    print(row_format.format(*headers))
    print("-"*80)
    
    for layout in PAPER_LAYOUTS:
        cfg = PAPER_PBT_CONFIGS[layout]
        print(row_format.format(
            layout,
            f"{cfg['learning_rate']:.2e}",
            f"{cfg['reward_shaping_horizon']:.0e}",
            f"{cfg['total_env_steps']:.0e}",
        ))


    print("\n")
    print("="*80)
    print("Paper PPO_BC Configurations (Table 3)")
    print("="*80)
    
    headers = ["Layout", "LR", "LR_Factor", "VF", "RewHorizon", "MiniBatches", "BC Schedule"]
    row_format = "{:<20}" + "{:<10}" * 5 + "{:<30}"
    
    print(row_format.format(*headers))
    print("-"*110)
    
    for layout in PAPER_LAYOUTS:
        cfg = PAPER_PPO_BC_CONFIGS[layout]
        schedule = cfg['bc_schedule']
        # Format schedule: show start/end of annealing
        if schedule[-1][1] > 0:
            sched_str = "N/A (100% BC always)"
        else:
            start = [s for s in schedule if s[1] == 1.0][-1][0]
            end = [s for s in schedule if s[1] == 0.0][0][0]
            sched_str = f"[{start:.0e}, {end:.0e}]"
        
        print(row_format.format(
            layout,
            f"{cfg['learning_rate']:.1e}",
            f"{cfg['lr_annealing_factor']}",
            f"{cfg['vf_coef']}",
            f"{cfg['reward_shaping_horizon']:.0e}",
            str(cfg['num_minibatches']),
            sched_str,
        ))


    print("\n")
    print("="*80)
    print("PPO_GAIL_controlled Configurations (same as PPO_BC Table 3, partner=GAIL)")
    print("="*80)
    print("  -> Uses identical HPs to PPO_BC. Only the partner model differs (GAIL vs BC).")
    for layout in PAPER_LAYOUTS:
        cfg = get_ppo_gail_config(layout, seed=0, gail_model_dir="<placeholder>")
        print(f"  {layout}: LR={cfg['learning_rate']:.1e}, VF={cfg.get('vf_coef', 0.5)}, "
              f"RewHorizon={cfg['reward_shaping_horizon']:.0e}, "
              f"Minibatches={cfg.get('num_minibatches', 6)}, "
              f"LR_Factor={cfg.get('lr_annealing_factor', 'N/A')}")

    print("\n")
    print("="*80)
    print("PPO_GAIL_optimized Configurations (Bayesian HPs, controlled budget)")
    print("="*80)
    print(f"  Bayesian base HPs: LR={BAYESIAN_OPTIMIZED_GAIL_PARAMS['learning_rate']:.2e}, "
          f"VF={BAYESIAN_OPTIMIZED_GAIL_PARAMS['vf_coef']:.5f}, "
          f"Ent=[{BAYESIAN_OPTIMIZED_GAIL_PARAMS['entropy_coeff_start']}->"
          f"{BAYESIAN_OPTIMIZED_GAIL_PARAMS['entropy_coeff_end']}]")
    for layout in PAPER_LAYOUTS:
        cfg = get_ppo_gail_optimized_config(layout, seed=0, gail_model_dir="<placeholder>")
        print(f"  {layout}: RewHorizon={cfg['reward_shaping_horizon']:.0e}, "
              f"Total={cfg['total_timesteps']/1e6:.0f}M")

    print("\n")
    print("="*80)
    print("PPO_SP_optimized Configurations (Bayesian HPs, self-play, ablation)")
    print("="*80)
    for layout in PAPER_LAYOUTS:
        cfg = get_ppo_sp_optimized_config(layout, seed=0)
        print(f"  {layout}: RewHorizon={cfg['reward_shaping_horizon']:.0e}, "
              f"Total={cfg['total_timesteps']/1e6:.0f}M, "
              f"bc_schedule={cfg['bc_schedule']}")


if __name__ == "__main__":
    print_config_summary()

