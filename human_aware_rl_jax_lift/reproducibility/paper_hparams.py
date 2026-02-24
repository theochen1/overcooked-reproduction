"""Paper-locked hyperparameters keyed by algorithm and layout."""

from copy import deepcopy
from typing import Optional


PAPER_LAYOUTS = ("simple", "unident_s", "random1", "random0", "random3")


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


def get_hparams(alg: str, layout: Optional[str] = None) -> dict:
    """Get paper-locked hyperparameters by algorithm and optional layout."""
    if alg == "ppo_sp":
        if layout not in _PPO_SP:
            raise KeyError(f"Unsupported layout '{layout}' for alg '{alg}'")
        return deepcopy(_PPO_SP[layout])
    if alg == "ppo_bc":
        if layout not in _PPO_BC:
            raise KeyError(f"Unsupported layout '{layout}' for alg '{alg}'")
        return deepcopy(_PPO_BC[layout])
    if alg == "pbt":
        return deepcopy(_PBT)
    if alg == "bc":
        return deepcopy(_BC)
    raise KeyError(f"Unsupported alg '{alg}'")
