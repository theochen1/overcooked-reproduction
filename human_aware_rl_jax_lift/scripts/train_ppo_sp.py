"""CLI for PPO self-play runs."""

import argparse

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.training.ppo_run import ppo_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO self-play models.")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, default="data/ppo_runs")
    parser.add_argument("--total_timesteps", type=int, default=int(5e6))
    args = parser.parse_args()

    cfg = PPOConfig(total_timesteps=args.total_timesteps, layout_name=args.layout, other_agent_type="sp")
    run_name = f"ppo_sp_{args.layout}"
    summaries = ppo_run(
        layout_name=args.layout,
        seeds=list(args.seeds),
        config=cfg,
        other_agent_type="sp",
        save_dir=args.save_dir,
        ex_name=run_name,
    )
    print({"run_name": run_name, "num_seeds": len(summaries), "summaries": summaries})


if __name__ == "__main__":
    main()
