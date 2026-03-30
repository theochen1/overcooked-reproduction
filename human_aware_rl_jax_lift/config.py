"""Paper-locked hyperparameters and deterministic seeding.

All hyperparameters are loaded from paper_config.yaml (shipped with the package).
"""

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import numpy as np
import yaml

PAPER_LAYOUTS = ("simple", "unident_s", "random1", "random0", "random3")

# Package root (human_aware_rl_jax_lift); paper_config.yaml lives there.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PAPER_CONFIG_PATH = _PACKAGE_ROOT / "paper_config.yaml"


def _load_yaml() -> dict:
    """Load paper_config.yaml or raise a clear error."""
    if not _PAPER_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"paper_config.yaml not found at {_PAPER_CONFIG_PATH}. "
            "This file ships with the package and should not be removed."
        )
    return yaml.safe_load(_PAPER_CONFIG_PATH.read_text()) or {}


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


def _coerce_ppo_value(ppo_key: str, val: Any) -> Any:
    """Cast a YAML value to the type expected by PPOConfig."""
    if ppo_key == "randomize_agent_idx":
        return bool(val)
    if ppo_key in ("num_envs", "horizon", "num_minibatches", "num_epochs",
                    "rew_shaping_horizon", "total_timesteps"):
        return int(val)
    return float(val)


def _get_ppo_sp_hparams(layout: str) -> Dict[str, Any]:
    """Load PPO SP hparams from paper_config.yaml (ppo_defaults + ppo_sp_layout_overrides)."""
    raw = _load_yaml()
    defaults = raw.get("ppo_defaults") or {}
    layout_overrides = (raw.get("ppo_sp_layout_overrides") or {}).get(layout)
    if layout_overrides is None:
        raise KeyError(f"Unsupported layout '{layout}' for alg 'ppo_sp'")
    merged = {**defaults, **layout_overrides}
    out: Dict[str, Any] = {}
    for yaml_key, ppo_key in _YAML_TO_PPO.items():
        if yaml_key in merged:
            out[ppo_key] = _coerce_ppo_value(ppo_key, merged[yaml_key])
    return out


def _get_ppo_bc_hparams(layout: str) -> Dict[str, Any]:
    """Load PPO BC hparams from paper_config.yaml (ppo_defaults + ppo_bc_layout_overrides)."""
    raw = _load_yaml()
    defaults = raw.get("ppo_defaults") or {}
    bc_overrides = (raw.get("ppo_bc_layout_overrides") or {}).get(layout)
    if bc_overrides is None:
        raise KeyError(f"Unsupported layout '{layout}' for alg 'ppo_bc'")
    result: Dict[str, Any] = {}
    # Apply ppo_defaults first
    if "entropy_coef" in defaults:
        result["ent_coef"] = float(defaults["entropy_coef"])
    if "max_grad_norm" in defaults:
        result["max_grad_norm"] = float(defaults["max_grad_norm"])
    # Apply layout overrides
    for yaml_key, ppo_key in _YAML_TO_PPO.items():
        if yaml_key in bc_overrides:
            result[ppo_key] = _coerce_ppo_value(ppo_key, bc_overrides[yaml_key])
    # Handle fields not in _YAML_TO_PPO
    if "lr_annealing" in bc_overrides:
        result["lr_annealing"] = float(bc_overrides["lr_annealing"])
    if "self_play_horizon" in bc_overrides:
        sph = bc_overrides["self_play_horizon"]
        result["self_play_horizon"] = tuple(int(x) for x in sph) if sph is not None else None
    return result


def _get_pbt_hparams() -> Dict[str, Any]:
    """Load PBT hparams from paper_config.yaml (pbt_defaults)."""
    raw = _load_yaml()
    pbt = raw.get("pbt_defaults") or {}
    return {
        "num_envs": int(pbt.get("num_envs", 50)),
        "population_size": int(pbt.get("population_size", 3)),
        "mutation_probability": float(pbt.get("mutation_probability", 0.33)),
        "mutation_factors": list(pbt.get("mutation_factors", [0.75, 1.25])),
        "iter_per_selection": int(pbt.get("iter_per_selection", 9)),
        "num_selection_games": int(pbt.get("num_selection_games", 6)),
    }


def _get_bc_hparams() -> Dict[str, Any]:
    """Load BC hparams from paper_config.yaml (bc section)."""
    raw = _load_yaml()
    bc = raw.get("bc") or {}
    return {
        "hidden_dim": 64,
        "num_hidden_layers": 2,
        "val_split": 0.15,
        "stuck_time": 3,
        "learning_rate": float(bc.get("learning_rate", 1e-3)),
        "num_epochs": int(bc.get("num_epochs", 100)),
        "batch_size": int(bc.get("batch_size", 64)),
        "adam_eps": float(bc.get("adam_eps", 1e-8)),
    }


def get_tf_compat() -> bool:
    """Read the tf_compat flag from paper_config.yaml.

    Returns True (replicate TF quirks) if the key is missing or the file is absent.
    """
    if not _PAPER_CONFIG_PATH.is_file():
        return True
    raw = yaml.safe_load(_PAPER_CONFIG_PATH.read_text()) or {}
    return bool(raw.get("tf_compat", True))


def get_hparams(alg: str, layout: Optional[str] = None) -> dict:
    """Get paper-locked hyperparameters by algorithm and optional layout."""
    if alg == "ppo_sp":
        return deepcopy(_get_ppo_sp_hparams(layout))
    if alg == "ppo_bc":
        return deepcopy(_get_ppo_bc_hparams(layout))
    if alg == "pbt":
        return deepcopy(_get_pbt_hparams())
    if alg == "bc":
        return deepcopy(_get_bc_hparams())
    raise KeyError(f"Unsupported alg '{alg}'")


def set_global_seed(seed: int):
    """Seed Python, NumPy, and return a JAX PRNGKey."""
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)
