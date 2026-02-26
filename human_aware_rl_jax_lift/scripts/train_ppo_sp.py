"""CLI for PPO self-play runs.

Supports both the legacy Python rollout path (ppo_run) and the fully-JAX
GPU-optimised path (ppo_run_jax).

Usage:
  python -u scripts/train_ppo_sp.py --layout unident_s --seeds 2229 7649 --jax
"""

import argparse

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.reproducibility.paper_hparams import get_hparams
from human_aware_rl_jax_lift.training.ppo_run import ppo_run
from human_aware_rl_jax_lift.training.ppo_run_jax import ppo_run_jax


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO self-play models.")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, default="data/ppo_runs")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable detailed PPO diagnostics logging (gradient norms/adv stats/loss components).",
    )
    parser.add_argument(
        "--jax",
        action="store_true",
        help="Use the fully-JAX rollout runner (vmap+lax.scan). Recommended on GPU.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Override max gradient norm for clipping (default: use config value).",
    )
    parser.add_argument(
        "--randomize_agent_idx",
        action="store_true",
        help="Randomly assign training agent to P0 or P1 each episode (TF parity: off).",
    )
    parser.add_argument(
        "--bootstrap_with_zero_obs",
        action="store_true",
        help="Use V(zeros) as rollout bootstrap value (mimics TF runner bootstrap path).",
    )
    parser.add_argument(
        "--global_adv_norm",
        action="store_true",
        help="Normalize advantages once over full batch before minibatching.",
    )
    args = parser.parse_args()

    overrides = get_hparams("ppo_sp", args.layout)
    cfg_kwargs = dict(
        total_timesteps=int(6e6 if args.total_timesteps is None else args.total_timesteps),
        layout_name=args.layout,
        other_agent_type="sp",
        learning_rate=float(overrides["learning_rate"]),
        vf_coef=float(overrides["vf_coef"]),
        rew_shaping_horizon=int(overrides["rew_shaping_horizon"]),
    )
    if args.max_grad_norm is not None:
        cfg_kwargs["max_grad_norm"] = args.max_grad_norm
    if args.randomize_agent_idx:
        cfg_kwargs["randomize_agent_idx"] = True
    if args.bootstrap_with_zero_obs:
        cfg_kwargs["bootstrap_with_zero_obs"] = True
    if args.global_adv_norm:
        cfg_kwargs["global_adv_norm"] = True
    cfg = PPOConfig(**cfg_kwargs)

    if args.jax:
        run_name = f"ppo_sp_jax_{args.layout}"
        summaries = ppo_run_jax(
            layout_name=args.layout,
            seeds=list(args.seeds),
            config=cfg,
            other_agent_type="sp",
            save_dir=args.save_dir,
            ex_name=run_name,
            diagnostics=bool(args.diagnostics),
        )
    else:
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
