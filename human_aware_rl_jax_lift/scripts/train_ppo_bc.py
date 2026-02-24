"""CLI for PPO runs against BC partners."""

import argparse
from pathlib import Path

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.training.checkpoints import load_best_bc_model_paths
from human_aware_rl_jax_lift.training.ppo_run import ppo_run


DEFAULT_SELF_PLAY_HORIZON = {
    "simple": (int(5e5), int(3e6)),
    "unident_s": (int(1e6), int(7e6)),
    "random1": (int(2e6), int(6e6)),
    "random0": None,
    "random3": (int(1e6), int(4e6)),
}

BC_LAYOUT_DEFAULTS = {
    # Paper Table 3 (PPOBC)
    "simple": {"learning_rate": 1e-3, "lr_annealing": 3.0, "vf_coef": 0.5, "rew_shaping_horizon": int(1e6)},
    "unident_s": {"learning_rate": 1e-3, "lr_annealing": 3.0, "vf_coef": 0.5, "rew_shaping_horizon": int(6e6)},
    "random1": {"learning_rate": 1e-3, "lr_annealing": 1.5, "vf_coef": 0.5, "rew_shaping_horizon": int(5e6)},
    "random0": {"learning_rate": 1.5e-3, "lr_annealing": 2.0, "vf_coef": 0.1, "rew_shaping_horizon": int(4e6)},
    "random3": {"learning_rate": 1.5e-3, "lr_annealing": 3.0, "vf_coef": 0.1, "rew_shaping_horizon": int(4e6)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO models against BC partners.")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--bc_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, default="data/ppo_runs")
    parser.add_argument("--bc_paths_file", type=str, default="data/bc_runs/best_bc_model_paths.pkl")
    parser.add_argument("--total_timesteps", type=int, default=int(5e6))
    parser.add_argument("--self_play_horizon", type=int, nargs=2, default=None)
    args = parser.parse_args()

    bc_paths = load_best_bc_model_paths(Path(args.bc_paths_file))
    other_agent_type = f"bc_{args.bc_split}"
    self_play_horizon = (
        tuple(args.self_play_horizon)
        if args.self_play_horizon is not None
        else DEFAULT_SELF_PLAY_HORIZON.get(args.layout, None)
    )
    overrides = BC_LAYOUT_DEFAULTS.get(args.layout, {})
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        layout_name=args.layout,
        other_agent_type=other_agent_type,
        learning_rate=float(overrides.get("learning_rate", 1e-3)),
        lr_annealing=float(overrides.get("lr_annealing", 1.0)),
        vf_coef=float(overrides.get("vf_coef", 0.5)),
        rew_shaping_horizon=int(overrides.get("rew_shaping_horizon", 0)),
        self_play_horizon=self_play_horizon,
    )
    run_name = f"ppo_bc_{args.bc_split}_{args.layout}"
    summaries = ppo_run(
        layout_name=args.layout,
        seeds=list(args.seeds),
        config=cfg,
        other_agent_type=other_agent_type,
        save_dir=args.save_dir,
        ex_name=run_name,
        best_bc_model_paths=bc_paths,
    )
    print({"run_name": run_name, "num_seeds": len(summaries), "summaries": summaries})


if __name__ == "__main__":
    main()
