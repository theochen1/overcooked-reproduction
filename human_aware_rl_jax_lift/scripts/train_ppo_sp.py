"""CLI for PPO self-play runs.

Usage:
  python -u scripts/train_ppo_sp.py --layout unident_s --seeds 2229 7649
"""

import argparse
from dataclasses import fields

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.reproducibility.paper_hparams import get_hparams

# Keys that PPOConfig accepts; we only pass these from overrides so YAML can drive all of them.
_PPO_CONFIG_KEYS = {f.name for f in fields(PPOConfig)}
from human_aware_rl_jax_lift.training.ppo_run import ppo_run


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
        "--max_grad_norm",
        type=float,
        default=None,
        help="Override max gradient norm for clipping (default: use config value).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate (default: paper hparams).",
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=None,
        help="Override value loss coefficient (default: paper hparams).",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=None,
        help="Override entropy coefficient (default: config default).",
    )
    parser.add_argument(
        "--clip_eps",
        type=float,
        default=None,
        help="Override PPO clip epsilon (default: config value).",
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
    parser.add_argument(
        "--adv_norm_fp64",
        action="store_true",
        help="Compute advantage normalization mean/std in float64 for parity probing.",
    )
    args = parser.parse_args()

    overrides = get_hparams("ppo_sp", args.layout)
    # total_timesteps: CLI override > paper_config layout override > 6e6 default (Fig. 8)
    total_timesteps = (
        args.total_timesteps
        if args.total_timesteps is not None
        else overrides.get("total_timesteps", int(6e6))
    )
    # Base: timesteps and identity; then all YAML/config overrides that PPOConfig accepts
    cfg_kwargs = dict(
        total_timesteps=int(total_timesteps),
        layout_name=args.layout,
        other_agent_type="sp",
    )
    for key, value in overrides.items():
        if key in _PPO_CONFIG_KEYS:
            cfg_kwargs[key] = value
    # CLI overrides win over config (so run_lr_vf_grid_simple.slurm etc. behave as before)
    if args.max_grad_norm is not None:
        cfg_kwargs["max_grad_norm"] = args.max_grad_norm
    if args.learning_rate is not None:
        cfg_kwargs["learning_rate"] = args.learning_rate
    if args.vf_coef is not None:
        cfg_kwargs["vf_coef"] = args.vf_coef
    if args.ent_coef is not None:
        cfg_kwargs["ent_coef"] = args.ent_coef
    if args.clip_eps is not None:
        cfg_kwargs["clip_eps"] = args.clip_eps
    if args.randomize_agent_idx:
        cfg_kwargs["randomize_agent_idx"] = True
    if args.bootstrap_with_zero_obs:
        cfg_kwargs["bootstrap_with_zero_obs"] = True
    if args.global_adv_norm:
        cfg_kwargs["global_adv_norm"] = True
    if args.adv_norm_fp64:
        cfg_kwargs["adv_norm_fp64"] = True
    cfg = PPOConfig(**cfg_kwargs)

    run_name = f"ppo_sp_{args.layout}"
    summaries = ppo_run(
        layout_name=args.layout,
        seeds=list(args.seeds),
        config=cfg,
        other_agent_type="sp",
        save_dir=args.save_dir,
        ex_name=run_name,
        diagnostics=bool(args.diagnostics),
    )

    print({"run_name": run_name, "num_seeds": len(summaries), "summaries": summaries})


if __name__ == "__main__":
    main()
