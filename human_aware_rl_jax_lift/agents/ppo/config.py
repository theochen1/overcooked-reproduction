"""PPO configuration matching legacy defaults."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Training
    total_timesteps: int = int(5e6)
    num_envs: int = 30
    horizon: int = 400
    num_minibatches: int = 6
    num_epochs: int = 8
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_eps: float = 0.05
    ent_coef: float = 0.1
    vf_coef: float = 0.5
    max_grad_norm: float = 0.1
    lr_annealing: float = 1.0
    rew_shaping_horizon: int = 0
    self_play_horizon: tuple[int, int] | None = None
    trajectory_self_play: bool = True
    save_best_thresh: float = 50.0
    other_agent_type: str = "sp"
    layout_name: str = "simple"

    # Network
    num_filters: int = 25
    hidden_dim: int = 32
    num_actions: int = 6
