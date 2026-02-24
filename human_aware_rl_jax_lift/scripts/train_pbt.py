"""CLI for PBT training runs."""

import argparse

from human_aware_rl_jax_lift.agents.pbt.config import PBTConfig
from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.reproducibility.paper_hparams import get_hparams
from human_aware_rl_jax_lift.training.checkpoints import load_best_bc_model_paths
from human_aware_rl_jax_lift.training.pbt_run import pbt_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PBT runs in human_aware_rl_jax_lift.")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--total_steps_per_agent", type=int, default=None)
    parser.add_argument("--other_agent_type", type=str, default="sp", choices=["sp", "bc_train", "bc_test"])
    parser.add_argument("--save_dir", type=str, default="data/pbt_runs")
    parser.add_argument("--bc_paths_file", type=str, default="data/bc_runs/best_bc_model_paths.pkl")
    args = parser.parse_args()

    pbt_h = get_hparams("pbt")
    ppo_cfg = PPOConfig(layout_name=args.layout, other_agent_type=args.other_agent_type, num_envs=int(pbt_h["num_envs"]))
    pbt_cfg = PBTConfig(
        population_size=int(pbt_h["population_size"]),
        resample_prob=float(pbt_h["mutation_probability"]),
        mutation_factors=list(pbt_h["mutation_factors"]),
        iter_per_selection=int(pbt_h["iter_per_selection"]),
        num_selection_games=int(pbt_h["num_selection_games"]),
    )
    bc_paths = None
    if args.other_agent_type != "sp":
        bc_paths = load_best_bc_model_paths(args.bc_paths_file)
    out = pbt_run(
        layout_name=args.layout,
        seeds=list(args.seeds),
        ppo_config=ppo_cfg,
        pbt_config=pbt_cfg,
        total_steps_per_agent=int(8e6 if args.total_steps_per_agent is None else args.total_steps_per_agent),
        other_agent_type=args.other_agent_type,
        best_bc_model_paths=bc_paths,
        save_dir=args.save_dir,
        ex_name=f"pbt_{args.layout}",
    )
    print(out)


if __name__ == "__main__":
    main()
