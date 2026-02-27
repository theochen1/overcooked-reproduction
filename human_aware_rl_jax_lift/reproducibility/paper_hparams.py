"""Paper-locked hyperparameters keyed by algorithm and layout.

PPO SP (and shared defaults) are loaded from paper_config.yaml when present;
fallback remains the in-module _PPO_SP for tests/standalone runs.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PAPER_LAYOUTS = ("simple", "unident_s", "random1", "random0", "random3")

# Package root (human_aware_rl_jax_lift); paper_config.yaml lives there.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PAPER_CONFIG_PATH = _PACKAGE_ROOT / "paper_config.yaml"


_PPO_SP = {
    # Table 2 (PPOSP)
    "simple": {
        "learning_rate": 1e-3,
        "vf_coef": 0.5,
        "rew_shaping_horizon": int(2.5e6),
    },
    "unident_s": {
        "learning_rate": 1e-3,
        "vf_coef": 0.5,
        "rew_shaping_horizon": int(2.5e6),
    },
    "random1": {
        "learning_rate": 6e-4,
        "vf_coef": 0.5,
        "rew_shaping_horizon": int(3.5e6),
    },
    "random0": {
        "learning_rate": 8e-4,
        "vf_coef": 0.5,
        "rew_shaping_horizon": int(2.5e6),
    },
    "random3": {
        "learning_rate": 8e-4,
        "vf_coef": 0.5,
        "rew_shaping_horizon": int(2.5e6),
    },
}

_PPO_BC = {
    # Table 3 (PPOBC)
    "simple": {
        "learning_rate": 1e-3,
        "lr_annealing": 3.0,
        "vf_coef": 0.5,
        "num_minibatches": 10,
        "rew_shaping_horizon": int(1e6),
        "self_play_horizon": (int(5e5), int(3e6)),
    },
    "unident_s": {
        "learning_rate": 1e-3,
        "lr_annealing": 3.0,
        "vf_coef": 0.5,
        "num_minibatches": 12,
        "rew_shaping_horizon": int(6e6),
        "self_play_horizon": (int(1e6), int(7e6)),
    },
    "random1": {
        "learning_rate": 1e-3,
        "lr_annealing": 1.5,
        "vf_coef": 0.5,
        "num_minibatches": 15,
        "rew_shaping_horizon": int(5e6),
        "self_play_horizon": (int(2e6), int(6e6)),
    },
    "random0": {
        "learning_rate": 1.5e-3,
        "lr_annealing": 2.0,
        "vf_coef": 0.1,
        "num_minibatches": 15,
        "rew_shaping_horizon": int(4e6),
        "self_play_horizon": None,
    },
    "random3": {
        "learning_rate": 1.5e-3,
        "lr_annealing": 3.0,
        "vf_coef": 0.1,
        "num_minibatches": 15,
        "rew_shaping_horizon": int(4e6),
        "self_play_horizon": (int(1e6), int(4e6)),
    },
}

_PBT = {
    # Appendix D
    "num_envs": 50,
    "population_size": 3,
    "mutation_probability": 0.33,
    "mutation_factors": [0.75, 1.25],
    "iter_per_selection": 9,
    "num_selection_games": 6,
}

_BC = {
    # Appendix A
    "hidden_dim": 64,
    "num_hidden_layers": 2,
    "val_split": 0.15,
    "stuck_time": 3,
}


# YAML key names in paper_config.yaml -> PPOConfig field names
_YAML_TO_PPO: Dict[str, str] = {
    "num_envs": "num_envs",
    "horizon": "horizon",
    "num_minibatches": "num_minibatches",
    "num_epochs": "num_epochs",
    "learning_rate": "learning_rate",
    "gamma": "gamma",
    "gae_lambda": "gae_lambda",
    "clip_eps": "clip_eps",
    "entropy_coef": "ent_coef",
    "value_coef": "vf_coef",
    "max_grad_norm": "max_grad_norm",
    "rew_shaping_horizon": "rew_shaping_horizon",
    "total_timesteps": "total_timesteps",
    "randomize_agent_idx": "randomize_agent_idx",
}


def _get_ppo_sp_from_yaml(layout: str) -> Optional[Dict[str, Any]]:
    """Load PPO SP hparams from paper_config.yaml (ppo_defaults + ppo_sp_layout_overrides).
    Returns a dict keyed by PPOConfig field names so train_ppo_sp can pass them directly.
    Returns None if file or layout missing.
    """
    if not _PAPER_CONFIG_PATH.is_file():
        return None
    raw = yaml.safe_load(_PAPER_CONFIG_PATH.read_text()) or {}
    defaults = raw.get("ppo_defaults") or {}
    layout_overrides = (raw.get("ppo_sp_layout_overrides") or {}).get(layout)
    if layout_overrides is None:
        return None
    merged = {**defaults, **layout_overrides}
    out: Dict[str, Any] = {}
    for yaml_key, ppo_key in _YAML_TO_PPO.items():
        if yaml_key not in merged:
            continue
        val = merged[yaml_key]
        if ppo_key == "randomize_agent_idx":
            out[ppo_key] = bool(val)
        elif ppo_key in ("num_envs", "horizon", "num_minibatches", "num_epochs", "rew_shaping_horizon", "total_timesteps"):
            out[ppo_key] = int(val)
        else:
            out[ppo_key] = float(val)
    return out


def get_hparams(alg: str, layout: Optional[str] = None) -> dict:
    """Get paper-locked hyperparameters by algorithm and optional layout."""
    if alg == "ppo_sp":
        if layout not in _PPO_SP:
            raise KeyError(f"Unsupported layout '{layout}' for alg '{alg}'")
        from_yaml = _get_ppo_sp_from_yaml(layout)
        if from_yaml is not None:
            return deepcopy(from_yaml)
        return deepcopy(_PPO_SP[layout])
    if alg == "ppo_bc":
        if layout not in _PPO_BC:
            raise KeyError(f"Unsupported layout '{layout}' for alg '{alg}'")
        result = deepcopy(_PPO_BC[layout])
        # Merge paper_config.yaml: ppo_defaults (ent_coef, max_grad_norm, etc.) + ppo_bc_layout_overrides
        if _PAPER_CONFIG_PATH.is_file():
            raw = yaml.safe_load(_PAPER_CONFIG_PATH.read_text()) or {}
            defaults = raw.get("ppo_defaults") or {}
            bc_overrides = (raw.get("ppo_bc_layout_overrides") or {}).get(layout) or {}
            # Paper PPO defaults (same as PPO_SP)
            if "entropy_coef" in defaults:
                result["ent_coef"] = float(defaults["entropy_coef"])
            if "max_grad_norm" in defaults:
                result["max_grad_norm"] = float(defaults["max_grad_norm"])
            # Layout overrides (may override defaults)
            for yaml_key, ppo_key in _YAML_TO_PPO.items():
                if yaml_key not in bc_overrides:
                    continue
                val = bc_overrides[yaml_key]
                if ppo_key == "randomize_agent_idx":
                    result[ppo_key] = bool(val)
                elif ppo_key in ("num_envs", "horizon", "num_minibatches", "num_epochs", "rew_shaping_horizon", "total_timesteps"):
                    result[ppo_key] = int(val)
                else:
                    result[ppo_key] = float(val)
        return result
    if alg == "pbt":
        return deepcopy(_PBT)
    if alg == "bc":
        return deepcopy(_BC)
    raise KeyError(f"Unsupported alg '{alg}'")
