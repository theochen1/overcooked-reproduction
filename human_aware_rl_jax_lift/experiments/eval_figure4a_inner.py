"""Figure 4A evaluation (inner loop).

This script evaluates a trained PPO policy paired with:
- BC(train) (paper's "BC" partner)
- BC(test) (paper's held-out human proxy / HProxy)

Compatibility goals
-------------------
- Works when executed from repo root (imports as human_aware_rl_jax_lift.*)
- Works with the existing SLURM wrapper human_aware_rl_jax_lift/slurm/05_eval_figure4a.slurm
  which runs from inside the human_aware_rl_jax_lift/ directory.

Patch notes (2026-03):
- --seed is now optional; if omitted, we auto-discover seed*/ directories.
- Accept legacy flag aliases: --num_games, --bc_paths_file, --out_dir.
- Add optional JSON output to --out_dir.
- Define a local `shaping` alias (0.0) for robustness against older edits.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# Ensure repo root is on sys.path even when launched from human_aware_rl_jax_lift/
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.agents.ppo.model import ActorCriticCNN
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.training.partners import BCPartner
from human_aware_rl_jax_lift.training.vec_env import batched_step, encode_obs, make_batched_state


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


def _first_existing_run_dir(ppo_runs_dir: Path, run_names: Tuple[str, ...]) -> Path:
    for name in run_names:
        d = ppo_runs_dir / name
        if d.exists() and d.is_dir():
            return d
    raise FileNotFoundError(f"None of these PPO run dirs exist under {ppo_runs_dir}: {run_names}")


def _discover_seeds(run_dir: Path) -> List[int]:
    seeds: List[int] = []
    for p in run_dir.glob("seed*"):
        if not p.is_dir():
            continue
        s = p.name.replace("seed", "")
        if s.isdigit():
            seeds.append(int(s))
    seeds.sort()
    if not seeds:
        raise FileNotFoundError(f"No seed*/ directories found under {run_dir}")
    return seeds


def make_ppo_policy(params, *, stochastic: bool = True) -> Callable:
    model = ActorCriticCNN(num_actions=6, num_filters=25, hidden_dim=32)
    apply_fn = jax.jit(model.apply)

    def act(obs_batch, rng: jax.Array):
        logits, _ = apply_fn(params, jnp.asarray(obs_batch, dtype=jnp.float32))
        if stochastic:
            a = jax.random.categorical(rng, logits, axis=-1)
        else:
            a = jnp.argmax(logits, axis=-1)
        return np.asarray(a, dtype=np.int32)

    return act


def _eval_pair(
    terrain,
    ppo_policy_fn: Callable,
    partner_params: dict,
    *,
    rng: jax.Array,
    num_envs: int,
    horizon: int,
    n_episodes: int,
    ppo_agent_idx: int,
):
    rng, env_rng = jax.random.split(rng)
    bstate = make_batched_state(terrain, num_envs, env_rng, randomize_agent_idx=False)
    bstate = bstate.replace(agent_idx=jnp.full((num_envs,), int(ppo_agent_idx), dtype=jnp.int32))
    obs0, obs1 = encode_obs(terrain, bstate)

    partner = BCPartner(params=partner_params, terrain=terrain, stochastic=True)

    ep_returns = []
    ep_ret = np.zeros((num_envs,), dtype=np.float32)

    n_blocks = int(np.ceil(float(n_episodes) / float(num_envs)))
    max_steps = horizon * max(1, n_blocks)

    player_order_actions = False

    # Sparse-only evaluation: shaping is always 0.0
    shaping = jnp.asarray(0.0, dtype=jnp.float32)

    for _ in range(max_steps):
        rng, k0, k1, kres = jax.random.split(rng, 4)

        a0 = ppo_policy_fn(obs0, k0)
        a1 = partner.act(
            np.asarray(obs1),
            k1,
            states=bstate.states,
            agent_idx=np.asarray(bstate.agent_idx),
        )

        reset_keys = jax.random.split(kres, num_envs)
        bstate, obs0, obs1, _rewards, dones, sparse_r = batched_step(
            terrain,
            bstate,
            a0,
            a1,
            reset_keys,
            shaping_factor=shaping,
            horizon=horizon,
            player_order_actions=player_order_actions,
            randomize_agent_idx=False,
        )

        ep_ret += np.asarray(sparse_r, dtype=np.float32)
        done_mask = np.asarray(dones, dtype=np.float32) > 0.5
        if np.any(done_mask):
            ep_returns.extend(ep_ret[done_mask].tolist())
            ep_ret[done_mask] = 0.0
            if len(ep_returns) >= n_episodes:
                break

    if not ep_returns:
        return 0.0
    return float(np.mean(ep_returns[:n_episodes]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional. If omitted, auto-discovers all seed*/ runs under the PPO run dir.",
    )

    parser.add_argument("--num_envs", type=int, default=30)

    parser.add_argument(
        "--n_episodes",
        "--num_games",
        dest="n_episodes",
        type=int,
        default=200,
        help="Number of episodes (games) to evaluate per condition.",
    )

    parser.add_argument("--ppo_runs_dir", type=str, default="data/ppo_runs")

    parser.add_argument(
        "--best_bc_paths",
        "--bc_paths_file",
        dest="best_bc_paths",
        type=str,
        default="data/bc_runs/best_bc_model_paths.pkl",
        help="Pickle file mapping layout -> BC model path for train/test splits.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional. If set, write a JSON file of per-seed results into this directory.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which PPO checkpoint to evaluate: 'best' (default) or 'final' (ppo_agent).",
    )
    args = parser.parse_args()

    layout = args.layout
    horizon = 400

    terrain = parse_layout(layout)

    best_paths_file = Path(args.best_bc_paths)
    bc_train_params = _load_bc_params(best_paths_file, "train", layout)
    hproxy_params = _load_bc_params(best_paths_file, "test", layout)

    ppo_runs_dir = Path(args.ppo_runs_dir)
    run_names = (f"ppo_bc_test_{layout}", f"ppo_bc_{layout}", f"ppo_bc_test_{layout}_v0")

    if args.seed is None:
        run_dir = _first_existing_run_dir(ppo_runs_dir, run_names)
        seeds = _discover_seeds(run_dir)
    else:
        seeds = [int(args.seed)]

    results = []
    for seed in seeds:
        ppo_params = None
        used_name = None
        for name in run_names:
            try:
                ppo_params = _load_ppo_params(ppo_runs_dir, name, seed, ckpt=args.ckpt)
                used_name = name
                break
            except FileNotFoundError:
                continue
        if ppo_params is None:
            raise FileNotFoundError(
                f"Could not load PPO checkpoint for layout={layout} seed={seed} in {ppo_runs_dir}"
            )

        ppo_policy = make_ppo_policy(ppo_params, stochastic=True)
        rng = jax.random.PRNGKey(0)

        v0 = _eval_pair(
            terrain,
            ppo_policy,
            hproxy_params,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
            ppo_agent_idx=0,
        )
        v1 = _eval_pair(
            terrain,
            ppo_policy,
            hproxy_params,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
            ppo_agent_idx=1,
        )

        v_bc = _eval_pair(
            terrain,
            ppo_policy,
            bc_train_params,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
            ppo_agent_idx=0,
        )

        out = {
            "layout": layout,
            "seed": seed,
            "ckpt": args.ckpt,
            "ppo_run_name": used_name,
            "gold_standard_mean": 0.5 * (v0 + v1),
            "gold_standard_v0": v0,
            "gold_standard_v1": v1,
            "ppo_plus_bc_train": v_bc,
        }
        results.append(out)
        print(out)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fig4a_inner_{layout}_{args.ckpt}.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
