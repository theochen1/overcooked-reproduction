"""
Shared utilities for model loading and evaluation across different runs.

Provides:
- Agent wrappers for BC, GAIL, and PPO models
- Model loading helpers
- Path resolution by run number
"""

import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action

# Layouts
LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

LAYOUT_TO_ENV = {
    # CRITICAL: Must use legacy layout names to match training environments.
    # Legacy layouts have cook_time=20, num_items_for_soup=3, delivery_reward=20
    # baked into the layout file. Non-legacy names lack these, causing MDP
    # parameter mismatches between training and evaluation.
    "cramped_room": "cramped_room_legacy",
    "asymmetric_advantages": "asymmetric_advantages_legacy",
    "coordination_ring": "coordination_ring_legacy",
    "forced_coordination": "random0_legacy",
    "counter_circuit": "random3_legacy",
}

SEEDS = [0, 10, 20, 30, 40]


class BCAgentWrapper(Agent):
    """Wrapper for BC model."""

    def __init__(self, model, featurize_fn, stochastic: bool = True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic

    def action(self, state):
        obs = self.featurize_fn(state)[self.agent_index]
        obs_flat = obs.flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1)

            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = probs.argmax(dim=-1).item()

        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": probs.numpy()}


class GAILAgentWrapper(Agent):
    """Wrapper for GAIL model."""

    def __init__(self, model, featurize_fn, stochastic: bool = True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic

    def action(self, state):
        obs = self.featurize_fn(state)[self.agent_index]
        obs_flat = obs.flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        with torch.no_grad():
            # GAIL policy returns (logits, value)
            logits, _ = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1)

            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = probs.argmax(dim=-1).item()

        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": probs.numpy()}


class PPOAgentWrapper(Agent):
    """Wrapper for JAX PPO model - uses jaxmarl env for consistent observation encoding."""

    def __init__(self, params, config, layout_name: str, stochastic: bool = True):
        super().__init__()
        self.params = params
        self.config = config
        self.layout_name = layout_name
        self.stochastic = stochastic
        self.jax_env = None  # Lazy init

        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp

        # Get the actual params dict
        self.p = params["params"] if "params" in params else params

        # Infer model structure from params
        self._analyze_params()

    def _analyze_params(self):
        """Analyze params to understand model structure."""
        self.conv_layers = sorted([k for k in self.p.keys() if k.startswith("Conv_")])
        self.dense_layers = sorted([k for k in self.p.keys() if k.startswith("Dense_")])
        self.has_conv = len(self.conv_layers) > 0

    def _init_jax_env(self):
        """Initialize JAX environment for observation encoding."""
        from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig

        # Map layout names
        env_name = LAYOUT_TO_ENV.get(self.layout_name, self.layout_name)
        config = OvercookedJaxEnvConfig(
            layout_name=env_name,
            horizon=400,
            use_lossless_encoding=True,  # Same as training
            old_dynamics=True,
        )
        self.jax_env = OvercookedJaxEnv(config)

    def _forward_conv(self, x, layer_name):
        """Apply a conv layer manually."""
        import jax.numpy as jnp
        from jax import lax

        kernel = self.p[layer_name]["kernel"]
        bias = self.p[layer_name]["bias"]

        # Determine padding based on layer index
        layer_idx = int(layer_name.split("_")[1])
        if layer_idx < len(self.conv_layers) - 1:
            padding = "SAME"
        else:
            padding = "VALID"

        # Apply convolution
        x = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        x = x + bias
        return x

    def _forward_dense(self, x, layer_name):
        """Apply a dense layer."""
        import jax.numpy as jnp

        kernel = self.p[layer_name]["kernel"]
        bias = self.p[layer_name]["bias"]
        return jnp.dot(x, kernel) + bias

    def _leaky_relu(self, x, alpha: float = 0.2):
        """Leaky ReLU activation.
        
        CRITICAL: alpha must match training. The original TF code uses
        tf.nn.leaky_relu which defaults to alpha=0.2, and the JAX training
        code uses nn.leaky_relu(x, negative_slope=0.2).
        """
        import jax.numpy as jnp

        return jnp.where(x > 0, x, alpha * x)

    def action(self, state):
        import jax
        import jax.numpy as jnp
        from jax import random

        # Initialize JAX env on first call
        if self.jax_env is None:
            self._init_jax_env()

        # Get observation using the legacy encoding (20 channels, same as training)
        obs = self.jax_env.base_env.lossless_state_encoding_mdp_legacy(state)[self.agent_index]
        obs = np.array(obs, dtype=np.float32)

        # Process through conv layers if present
        if self.has_conv and len(obs.shape) >= 2:
            if len(obs.shape) == 2:
                obs = obs[..., None]  # Add channel: (H, W) -> (H, W, 1)
            x = jnp.array(obs)[None, ...]  # Add batch: (H, W, C) -> (1, H, W, C)

            # Apply conv layers with saved params
            for layer_name in self.conv_layers:
                x = self._forward_conv(x, layer_name)
                x = self._leaky_relu(x)

            # Flatten
            x = x.reshape(-1)
        else:
            x = jnp.array(obs.flatten())

        # Dense layers (skip last 2 which are actor/critic heads)
        hidden_layers = self.dense_layers[:-2]
        for layer_name in hidden_layers:
            x = self._forward_dense(x, layer_name)
            x = self._leaky_relu(x)

        # Actor head (second to last dense layer)
        actor_layer = self.dense_layers[-2]
        logits = self._forward_dense(x, actor_layer)

        probs = jax.nn.softmax(logits)

        if self.stochastic:
            key = random.PRNGKey(np.random.randint(0, 2**31))
            action_idx = random.categorical(key, logits).item()
        else:
            action_idx = jnp.argmax(logits).item()

        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": np.array(probs)}


def get_model_paths(run_number: int, base_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get model directory paths for a given run number.

    Args:
        run_number: Run number (e.g., 3, 4)
        base_dir: Optional base directory. If None, uses default structure.

    Returns:
        Dictionary mapping model type to directory path.
    """
    if base_dir:
        # Custom base directory structure
        return {
            "bc_train": os.path.join(base_dir, "bc_runs", "train"),
            "bc_test": os.path.join(base_dir, "bc_runs", "test"),
            "gail": os.path.join(base_dir, "gail_runs"),
            "ppo_sp": os.path.join(base_dir, "ppo_sp"),
            "ppo_bc": os.path.join(base_dir, "ppo_bc"),
            "ppo_gail": os.path.join(base_dir, "ppo_gail"),
        }

    # Default structure based on discovered locations
    eval_dir = os.path.dirname(os.path.dirname(__file__))
    return {
        "bc_train": os.path.join(eval_dir, f"bc_runs_run{run_number}", "train"),
        "bc_test": os.path.join(eval_dir, f"bc_runs_run{run_number}", "test"),
        "gail": os.path.join(eval_dir, f"gail_runs_run{run_number}"),
        "ppo_sp": os.path.join(os.path.dirname(eval_dir), "results", f"ppo_sp_run{run_number}"),
        "ppo_bc": os.path.join(os.path.dirname(eval_dir), "results", f"ppo_bc_run{run_number}"),
        "ppo_gail": os.path.join(os.path.dirname(eval_dir), "results", f"ppo_gail_run{run_number}"),
    }


def load_bc_model(model_dir: str, verbose: bool = False):
    """Load BC model."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model as _load

    return _load(model_dir, verbose=verbose)


def load_gail_model(model_dir: str, state_dim: int, action_dim: int = 6):
    """Load GAIL model."""
    from human_aware_rl.imitation.gail import GAILPolicy

    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No GAIL model at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    # Create policy with correct dimensions
    policy = GAILPolicy(state_dim=state_dim, action_dim=action_dim)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    return policy


def load_ppo_model(model_dir: str, verbose: bool = False) -> Tuple[Dict, Dict]:
    """
    Load PPO model from checkpoint directory.

    Args:
        model_dir: Directory containing checkpoint_* subdirectories
        verbose: Print debug information

    Returns:
        Tuple of (params, config)
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint_")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint in {model_dir}")

    latest = sorted(checkpoint_dirs)[-1]
    checkpoint_path = os.path.join(model_dir, latest)

    params_path = os.path.join(checkpoint_path, "params.pkl")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No params.pkl in {checkpoint_path}")

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    config_path = os.path.join(checkpoint_path, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    else:
        config = {}

    if verbose:
        p = params["params"] if "params" in params else params
        print(f"    Params keys: {list(p.keys())}")
        if "Dense_0" in p:
            print(f"    Dense_0 kernel shape: {p['Dense_0']['kernel'].shape}")
        print(f"    Config: {config}")

    return params, config


def load_ppo_model_from_files(model_dir: str, verbose: bool = False) -> Tuple[Dict, Dict]:
    """
    Load PPO model from direct files (for PPO_GAIL format where params.pkl is directly in seed directory).
    """
    params_path = os.path.join(model_dir, "params.pkl")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No params.pkl in {model_dir}")

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    config_path = os.path.join(model_dir, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    else:
        config = {}

    if verbose:
        p = params["params"] if "params" in params else params
        print(f"    Params keys: {list(p.keys())}")

    return params, config
