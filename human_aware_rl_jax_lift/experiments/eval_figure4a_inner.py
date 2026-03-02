"""Figure 4A evaluation (inner loop).

This script evaluates a trained PPO policy against BC (train split) and a
held-out human proxy (BC test split / HProxy) in the paper's Figure 4A setup.

Patch note (2026-03):
- Add --ckpt flag to force using either the best-checkpoint or final checkpoint
  for PPO runs. This helps diagnose cases where "best" was selected during
  training under self-play mixing but evaluation assumes a fixed partner.
"""

import argparse
import pickle
from pathlib import Path
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.training.vec_env import batched_step, encode_obs, make_batched_state
from human_aware_rl_jax_lift.agents.bc.policy import make_bc_policy
from human_aware_rl_jax_lift.agents.ppo.policy import make_ppo_policy


def _load_bc_params(best_paths_file: Path, split: str, layout_name: str):
    with best_paths_file.open("rb") as f:
        best_paths = pickle.load(f)
    bc_path = Path(best_paths[split][layout_name])
    if bc_path.is_dir():
        bc_path = bc_path / "model.pkl"
    with bc_path.open("rb") as f:
        payload = pickle.load(f)
    return payload.get("params", payload) if isinstance(payload, dict) else payload


def _load_ppo_params(
    ppo_runs_dir: Path,
    run_name: str,
    seed: int,
    *,
    ckpt: str = "best",
):
    """Load PPO checkpoint params.

    ckpt:
      - "best": prefer seed*/best/params.pkl, fall back to seed*/ppo_agent/params.pkl
      - "final": prefer seed*/ppo_agent/params.pkl, fall back to seed*/best/params.pkl

    The fallback makes the flag robust to older run dirs that may only have one
    of the two.
    """
    seed_dir = ppo_runs_dir / run_name / f"seed{seed}"
    best_dir = seed_dir / "best"
    final_dir = seed_dir / "ppo_agent"

    def _try(dir_: Path):
        p = dir_ / "params.pkl"
        if not p.exists():
            return None
        with p.open("rb") as f:
            payload = pickle.load(f)
        return payload.get("params", payload) if isinstance(payload, dict) else payload

    if ckpt not in ("best", "final"):
        raise ValueError(f"Invalid ckpt='{ckpt}'. Expected 'best' or 'final'.")

    if ckpt == "best":
        out = _try(best_dir)
        if out is not None:
            return out
        out = _try(final_dir)
        if out is not None:
            return out
    else:
        out = _try(final_dir)
        if out is not None:
            return out
        out = _try(best_dir)
        if out is not None:
            return out

    raise FileNotFoundError(
        f"Could not find PPO params for {run_name}/seed{seed} with ckpt='{ckpt}'. "
        f"Looked for {best_dir/'params.pkl'} and {final_dir/'params.pkl'}."
    )


def load_first_existing(
    ppo_runs_dir: Path,
    run_names: Tuple[str, ...],
    seed: int,
    *,
    ckpt: str,
):
    for name in run_names:
        try:
            return _load_ppo_params(ppo_runs_dir, name, seed, ckpt=ckpt)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"None of the PPO run names exist for seed={seed}: {run_names}"
    )


def _eval_pair(
    terrain,
    ppo_policy_fn: Callable,
    partner_policy_fn: Callable,
    *,
    rng: jax.Array,
    num_envs: int,
    horizon: int,
    n_episodes: int,
    player_order_actions: bool,
):
    """Evaluate PPO (training agent) paired with a partner policy.

    Returns mean sparse reward over n_episodes.
    """
    rng, env_rng = jax.random.split(rng)
    bstate = make_batched_state(terrain, num_envs, env_rng, randomize_agent_idx=False)
    obs0, obs1 = encode_obs(terrain, bstate)

    ep_returns = []
    ep_done = np.zeros((num_envs,), dtype=np.int32)
    ep_ret = np.zeros((num_envs,), dtype=np.float32)

    rng, step_rng = jax.random.split(rng)
    keys = jax.random.split(step_rng, horizon)

    for t in range(horizon):
        k = keys[t]
        k0, k1, kres = jax.random.split(k, 3)

        # PPO acts as training agent (obs0), partner acts on obs1.
        a0 = ppo_policy_fn(obs0, k0)
        a1 = partner_policy_fn(obs1, k1)

        reset_keys = jax.random.split(kres, num_envs)

        bstate, obs0, obs1, rewards, dones, sparse_r = batched_step(
            terrain,
            bstate,
            a0,
            a1,
            reset_keys,
            shaping_factor=jnp.asarray(0.0, dtype=jnp.float32),
            horizon=horizon,
            player_order_actions=player_order_actions,
            randomize_agent_idx=False,
        )

        sparse_r_np = np.asarray(sparse_r)
        done_np = np.asarray(dones).astype(np.int32)

        ep_ret += sparse_r_np
        just_done = (done_np == 1) & (ep_done == 0)
        if np.any(just_done):
            ep_returns.extend(ep_ret[just_done].tolist())
            ep_done[just_done] = 1
        if len(ep_returns) >= n_episodes:
            break

    if not ep_returns:
        return 0.0
    return float(np.mean(ep_returns[:n_episodes]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, default=30)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--ppo_runs_dir", type=str, default="data/ppo_runs")
    parser.add_argument("--best_bc_paths", type=str, default="data/bc_runs/best_bc_model_paths.pkl")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which PPO checkpoint to evaluate: 'best' (default) or 'final' (ppo_agent).",
    )
    args = parser.parse_args()

    layout = args.layout
    seed = int(args.seed)
    horizon = 400

    terrain = parse_layout(layout)

    best_paths_file = Path(args.best_bc_paths)
    bc_train_params = _load_bc_params(best_paths_file, "train", layout)
    hproxy_params = _load_bc_params(best_paths_file, "test", layout)

    # PPO checkpoints (you may have multiple run-name conventions)
    ppo_runs_dir = Path(args.ppo_runs_dir)
    ppo_bc_test_params = load_first_existing(
        ppo_runs_dir,
        (f"ppo_bc_test_{layout}", f"ppo_bc_{layout}", f"ppo_bc_test_{layout}_v0"),
        seed,
        ckpt=args.ckpt,
    )

    ppo_policy = make_ppo_policy(ppo_bc_test_params)
    bc_train_policy = make_bc_policy(bc_train_params)
    hproxy_policy = make_bc_policy(hproxy_params)

    rng = jax.random.PRNGKey(0)

    # NOTE: player_order_actions controls whether (a0,a1) are interpreted as (p0,p1)
    # or (agent_idx, other). For this evaluator we want agent_idx ordering.
    player_order_actions = False

    # Gold standard: average over PPO as P0 and PPO as P1 (role swap)
    v0 = _eval_pair(
        terrain,
        ppo_policy,
        hproxy_policy,
        rng=rng,
        num_envs=args.num_envs,
        horizon=horizon,
        n_episodes=args.n_episodes,
        player_order_actions=player_order_actions,
    )
    v1 = _eval_pair(
        terrain,
        hproxy_policy,
        ppo_policy,
        rng=rng,
        num_envs=args.num_envs,
        horizon=horizon,
        n_episodes=args.n_episodes,
        player_order_actions=player_order_actions,
    )

    # Additional baselines
    v_bc = _eval_pair(
        terrain,
        ppo_policy,
        bc_train_policy,
        rng=rng,
        num_envs=args.num_envs,
        horizon=horizon,
        n_episodes=args.n_episodes,
        player_order_actions=player_order_actions,
    )

    out = {
        "layout": layout,
        "seed": seed,
        "ckpt": args.ckpt,
        "gold_standard_mean": 0.5 * (v0 + v1),
        "gold_standard_v0": v0,
        "gold_standard_v1": v1,
        "ppo_plus_bc_train": v_bc,
    }
    print(out)


if __name__ == "__main__":
    main()
