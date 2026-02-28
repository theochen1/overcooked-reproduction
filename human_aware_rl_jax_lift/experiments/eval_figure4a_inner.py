#!/usr/bin/env python
"""JAX-native Figure 4a evaluation (paper-faithful, Slurm-friendly).

Outputs
-------
Writes: {out_dir}/results_{layout}.json
Format: {paper_layout_key: {condition: {seed_idx: mean_reward}, ..., "gold_standard": float}}

Protocol (Carroll et al., NeurIPS 2019)
--------------------------------------
- horizon = 400 timesteps
- 100 rollouts per (seed, condition)
- 5 seeds per training condition (standard error computed in plotting)
- HProxy is a held-out BC model (bc_test)
- BC is the accessible imperfect model (bc_train)
- gold standard: PPO trained with HProxy itself (ppo_bc_test)
- hashed bars: swapped starting positions (swap which policy controls P0 vs P1)

Notes
-----
This script intentionally avoids Docker / TensorFlow and evaluates directly from
human_aware_rl_jax_lift checkpoints:
- PPO checkpoints: seed{seed}/best/params.pkl (fallback seed{seed}/ppo_agent/params.pkl)
- BC checkpoints: model.pkl + bc_metadata.pkl
- Best BC paths map: data/bc_runs/best_bc_model_paths.pkl

"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

from human_aware_rl_jax_lift.training.checkpoints import (
    load_best_bc_model_paths,
    load_bc_checkpoint,
    load_ppo_checkpoint,
)

# Inference-time models
from human_aware_rl_jax_lift.agents.ppo.model import ActorCriticCNN
from human_aware_rl_jax_lift.agents.bc.model import BCPolicy

# JAX env/rollout utilities
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.training.vec_env import make_batched_state, encode_obs, batched_step
from human_aware_rl_jax_lift.encoding.bc_features import featurize_state_64


HORIZON = 400

# Short layout name -> paper key used by figure4a.py
LAYOUT_KEY = {
    "simple": "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random1": "coordination_ring",
    "random0": "forced_coordination",
    "random3": "counter_circuit",
}

# Seed tables (match paper reproduction / TF baselines)
PPO_SP_SEEDS: List[int] = [2229, 386, 7225, 7649, 9807]
PPO_BC_TEST_SEEDS: List[int] = [184, 2888, 4467, 7360, 7424]   # PPO trained with HProxy (gold standard)
PPO_BC_TRAIN_SEEDS: List[int] = [1887, 516, 5578, 5987, 9456]  # PPO trained with BC_train


def _maybe_logits(output):
    if isinstance(output, tuple) and len(output) >= 1:
        return output[0]
    return output


def _load_bc_params(best_paths: dict, layout: str, split: str) -> dict:
    bc_dir = Path(best_paths[split][layout])
    bc_dir = bc_dir if bc_dir.is_dir() else bc_dir.parent
    params, _meta = load_bc_checkpoint(bc_dir)
    return params


def _load_ppo_params(ppo_runs_dir: Path, run_name: str, seed: int) -> dict:
    seed_dir = ppo_runs_dir / run_name / f"seed{seed}"
    best_dir = seed_dir / "best"
    final_dir = seed_dir / "ppo_agent"

    if (best_dir / "params.pkl").exists():
        return load_ppo_checkpoint(best_dir)
    if (final_dir / "params.pkl").exists():
        return load_ppo_checkpoint(final_dir)
    raise FileNotFoundError(
        f"Could not find PPO params.pkl for run={run_name} seed={seed}. "
        f"Tried: {best_dir}/params.pkl and {final_dir}/params.pkl"
    )


def _bc_logits(params: dict, terrain, states, which_player: int) -> jnp.ndarray:
    f0, f1 = jax.vmap(lambda s: featurize_state_64(terrain, s))(states)
    feats = f0 if which_player == 0 else f1
    logits = BCPolicy().apply(params, feats)
    return _maybe_logits(logits)


def _ppo_logits(params: dict, obs: jnp.ndarray) -> jnp.ndarray:
    out = ActorCriticCNN().apply(params, obs)
    return _maybe_logits(out)


def _eval_pair(
    terrain,
    a0_kind: str,
    a0_params: dict,
    a1_kind: str,
    a1_params: dict,
    *,
    num_games: int,
    rng_key: jax.Array,
) -> float:
    """Evaluate mean cumulative sparse reward over HORIZON steps."""

    rng_key, reset_rng = jax.random.split(rng_key)
    bstate = make_batched_state(terrain, num_games, reset_rng, randomize_agent_idx=False)
    obs0, obs1 = encode_obs(terrain, bstate)

    shaping = jnp.array(0.0, dtype=jnp.float32)

    def step_fn(carry, _t):
        bstate, obs0, obs1, rng = carry
        rng, k0, k1, kresets = jax.random.split(rng, 4)
        reset_keys = jax.random.split(kresets, num_games)

        if a0_kind == "ppo":
            logits0 = _ppo_logits(a0_params, obs0)
        else:
            logits0 = _bc_logits(a0_params, terrain, bstate.states, which_player=0)

        if a1_kind == "ppo":
            logits1 = _ppo_logits(a1_params, obs1)
        else:
            logits1 = _bc_logits(a1_params, terrain, bstate.states, which_player=1)

        act0 = jax.random.categorical(k0, logits0)
        act1 = jax.random.categorical(k1, logits1)

        bstate, obs0, obs1, _rewards, _dones, sparse_r = batched_step(
            terrain,
            bstate,
            act0.astype(jnp.int32),
            act1.astype(jnp.int32),
            reset_keys,
            shaping,
            HORIZON,
            player_order_actions=True,
            randomize_agent_idx=False,
        )
        return (bstate, obs0, obs1, rng), sparse_r

    (_bstate, _obs0, _obs1, _rng), sparse_traj = jax.lax.scan(
        step_fn, (bstate, obs0, obs1, rng_key), xs=None, length=HORIZON
    )

    returns = jnp.sum(sparse_traj, axis=0)
    return float(jnp.mean(returns))


def _try_run_names(prefixes: List[str], layout: str) -> List[str]:
    return [p.format(layout=layout) for p in prefixes]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Figure 4a for one layout (JAX-native).")
    ap.add_argument("--layout", required=True, choices=sorted(LAYOUT_KEY.keys()))
    ap.add_argument("--num_games", type=int, default=100)
    ap.add_argument("--ppo_runs_dir", type=str, default="data/ppo_runs")
    ap.add_argument("--bc_paths_file", type=str, default="data/bc_runs/best_bc_model_paths.pkl")
    ap.add_argument("--out_dir", type=str, default="eval_results")
    args = ap.parse_args()

    layout = args.layout
    fig_key = LAYOUT_KEY[layout]

    ppo_runs_dir = Path(args.ppo_runs_dir)
    best_bc_paths = load_best_bc_model_paths(Path(args.bc_paths_file))

    bc_train = _load_bc_params(best_bc_paths, layout, "train")
    hproxy = _load_bc_params(best_bc_paths, layout, "test")

    terrain = parse_layout(layout)

    out: Dict[str, object] = {
        "SP_SP": {},
        "SP_HProxy": {},
        "PPOBC_HProxy": {},
        "BC_HProxy": {},
        "SP_HProxy_sw": {},
        "PPOBC_HProxy_sw": {},
        "BC_HProxy_sw": {},
        "gold_standard": None,
    }

    sp_run_candidates = _try_run_names([
        "ppo_sp_jax_{layout}",
        "ppo_sp_{layout}",
    ], layout)

    bc_train_run_candidates = _try_run_names([
        "ppo_bc_train_jax_{layout}",
        "ppo_bc_train_{layout}",
    ], layout)

    bc_test_run_candidates = _try_run_names([
        "ppo_bc_test_jax_{layout}",
        "ppo_bc_test_{layout}",
    ], layout)

    def load_first_existing(run_names: List[str], seed: int) -> Tuple[str, dict]:
        last_err = None
        for rn in run_names:
            try:
                return rn, _load_ppo_params(ppo_runs_dir, rn, seed)
            except FileNotFoundError as e:
                last_err = e
                continue
        raise last_err if last_err is not None else FileNotFoundError("No run names provided")

    # PPO-SP
    for i, seed in enumerate(PPO_SP_SEEDS):
        rn, params = load_first_existing(sp_run_candidates, seed)
        rng = jax.random.PRNGKey(int(seed))
        out["SP_SP"][i] = _eval_pair(terrain, "ppo", params, "ppo", params, num_games=args.num_games, rng_key=rng)
        out["SP_HProxy"][i] = _eval_pair(terrain, "ppo", params, "bc", hproxy, num_games=args.num_games, rng_key=rng)
        out["SP_HProxy_sw"][i] = _eval_pair(terrain, "bc", hproxy, "ppo", params, num_games=args.num_games, rng_key=rng)
        print(f"[{layout}] SP run={rn} seed={seed} -> done")

    # PPO-BC-train (trained with BC_train; evaluated w/ HProxy)
    for i, seed in enumerate(PPO_BC_TRAIN_SEEDS):
        rn, params = load_first_existing(bc_train_run_candidates, seed)
        rng = jax.random.PRNGKey(int(seed))
        out["PPOBC_HProxy"][i] = _eval_pair(terrain, "ppo", params, "bc", hproxy, num_games=args.num_games, rng_key=rng)
        out["PPOBC_HProxy_sw"][i] = _eval_pair(terrain, "bc", hproxy, "ppo", params, num_games=args.num_games, rng_key=rng)
        print(f"[{layout}] PPOBC-train run={rn} seed={seed} -> done")

    # BC baseline (BC_train vs HProxy) — run once and replicate
    rng = jax.random.PRNGKey(0)
    bc_mean = _eval_pair(terrain, "bc", bc_train, "bc", hproxy, num_games=args.num_games, rng_key=rng)
    bc_mean_sw = _eval_pair(terrain, "bc", hproxy, "bc", bc_train, num_games=args.num_games, rng_key=rng)
    for i in range(5):
        out["BC_HProxy"][i] = bc_mean
        out["BC_HProxy_sw"][i] = bc_mean_sw

    # Gold standard: PPO trained with HProxy itself (bc_test)
    gold_vals: List[float] = []
    for seed in PPO_BC_TEST_SEEDS:
        rn, params = load_first_existing(bc_test_run_candidates, seed)
        rng = jax.random.PRNGKey(int(seed))
        v0 = _eval_pair(terrain, "ppo", params, "bc", hproxy, num_games=args.num_games, rng_key=rng)
        v1 = _eval_pair(terrain, "bc", hproxy, "ppo", params, num_games=args.num_games, rng_key=rng)
        gold_vals.append(0.5 * (v0 + v1))
        print(f"[{layout}] gold run={rn} seed={seed} -> done")
    out["gold_standard"] = float(sum(gold_vals) / len(gold_vals))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"results_{layout}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({fig_key: out}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
