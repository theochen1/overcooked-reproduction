"""CLI for PPO runs against BC partners."""

import argparse
from pathlib import Path

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.reproducibility.paper_hparams import get_hparams
from human_aware_rl_jax_lift.training.checkpoints import load_best_bc_model_paths
from human_aware_rl_jax_lift.training.ppo_run import ppo_run

BC_TOTAL_TIMESTEPS = {
    "simple": int(8e6),
    "unident_s": int(1e7),
    "random1": int(1.6e7),
    "random0": int(9e6),
    "random3": int(1.2e7),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO models against BC partners.")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--bc_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, default="data/ppo_runs")
    parser.add_argument("--bc_paths_file", type=str, default="data/bc_runs/best_bc_model_paths.pkl")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--self_play_horizon", type=int, nargs=2, default=None)
    args = parser.parse_args()

    bc_paths = load_best_bc_model_paths(Path(args.bc_paths_file))
    other_agent_type = f"bc_{args.bc_split}"
    overrides = get_hparams("ppo_bc", args.layout)
    self_play_horizon = (
        tuple(args.self_play_horizon)
        if args.self_play_horizon is not None
        else overrides["self_play_horizon"]
    )
    cfg = PPOConfig(
        total_timesteps=int(BC_TOTAL_TIMESTEPS[args.layout] if args.total_timesteps is None else args.total_timesteps),
        layout_name=args.layout,
        other_agent_type=other_agent_type,
        learning_rate=float(overrides["learning_rate"]),
        lr_annealing=float(overrides["lr_annealing"]),
        vf_coef=float(overrides["vf_coef"]),
        num_minibatches=int(overrides["num_minibatches"]),
        rew_shaping_horizon=int(overrides["rew_shaping_horizon"]),
        self_play_horizon=self_play_horizon,
        randomize_agent_idx=bool(overrides.get("randomize_agent_idx", False)),
        ent_coef=float(overrides.get("ent_coef", 0.01)),
        max_grad_norm=float(overrides.get("max_grad_norm", 0.1)),
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
