"""Figure 4A evaluation (inner loop).

This script writes results in the legacy contract expected by:
- human_aware_rl_jax_lift/experiments/prepare_results.py
- human_aware_rl_jax_lift/experiments/figure4a.py

It is designed to run under the SLURM wrapper:
  human_aware_rl_jax_lift/slurm/05_eval_figure4a.slurm

Output contract
--------------
Writes a single JSON object to:
  {out_dir}/results_{layout}.json

with schema:
  results[layout_key][condition][seed_idx] -> float
  results[layout_key]["gold_standard"] -> float

where:
- layout is the short SLURM layout name (simple, unident_s, random0, random1, random3)
- layout_key matches figure4a.py's LAYOUT_ORDER
- condition keys match figure4a.py's CONDITION_ORDER
- seed_idx are contiguous integers starting at 0 (0..4 for the paper)

Checkpoint semantics
-------------------
Default behavior mirrors legacy: prefer "best" if available, else fallback to "final".

Notes
-----
- Evaluation uses sparse reward only (shaping = 0.0) and horizon = 400.
- Policies are evaluated stochastically.
- Different PPO runs may have different raw seed folder names (e.g., seed184 vs seed386).
  To avoid spurious failures, we discover seeds per *run group* (SP vs PPOBC vs gold)
  and filter to seeds that actually contain a checkpoint file.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path even when launched from human_aware_rl_jax_lift/
_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[1]  # .../human_aware_rl_jax_lift
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


# Dir convention (SLURM layout names) -> figure4a.py layout keys
_LAYOUT_KEY = {
    "simple": "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random1": "coordination_ring",
    "random0": "forced_coordination",
    "random3": "counter_circuit",
}


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

    # Legacy semantics: try best if available, else final.
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


def _seed_has_any_ckpt(seed_dir: Path) -> bool:
    return (seed_dir / "best" / "params.pkl").exists() or (seed_dir / "ppo_agent" / "params.pkl").exists()


def _discover_seeds(run_dir: Path) -> List[int]:
    seeds: List[int] = []
    for p in run_dir.glob("seed*"):
        if not p.is_dir():
            continue
        s = p.name.replace("seed", "")
        if not s.isdigit():
            continue
        if not _seed_has_any_ckpt(p):
            continue
        seeds.append(int(s))

    seeds.sort()
    if not seeds:
        raise FileNotFoundError(
            f"No seed*/ directories with a checkpoint found under {run_dir}. "
            f"Expected either best/params.pkl or ppo_agent/params.pkl inside each seed dir."
        )
    return seeds


def _seeds_for_group(
    *,
    ppo_runs_dir: Path,
    run_names: Tuple[str, ...],
    num_seeds: int,
    seed_override: Optional[int],
    group_name: str,
) -> List[int]:
    if seed_override is not None:
        return [int(seed_override)]

    run_dir = _first_existing_run_dir(ppo_runs_dir, run_names)
    discovered = _discover_seeds(run_dir)
    if len(discovered) < int(num_seeds):
        raise FileNotFoundError(
            f"Not enough usable seeds for group '{group_name}'. Found {len(discovered)} under {run_dir}, "
            f"need {num_seeds}. Usable seeds are: {discovered}"
        )
    return discovered[: int(num_seeds)]


def make_ppo_act(params, *, stochastic: bool = True) -> Callable:
    model = ActorCriticCNN(num_actions=6, num_filters=25, hidden_dim=32)
    apply_fn = jax.jit(model.apply)

    def act(obs_batch, rng: jax.Array, **_kwargs):
        logits, _ = apply_fn(params, jnp.asarray(obs_batch, dtype=jnp.float32))
        if stochastic:
            a = jax.random.categorical(rng, logits, axis=-1)
        else:
            a = jnp.argmax(logits, axis=-1)
        return np.asarray(a, dtype=np.int32)

    return act


def make_bc_act(params, terrain, *, stochastic: bool = True) -> Callable:
    partner = BCPartner(params=params, terrain=terrain, stochastic=stochastic)

    def act(obs_batch, rng: jax.Array, *, states, agent_idx):
        return partner.act(
            np.asarray(obs_batch),
            rng,
            states=states,
            agent_idx=agent_idx,
        )

    return act


def _eval_joint(
    terrain,
    act0: Callable,
    act1: Callable,
    *,
    rng: jax.Array,
    num_envs: int,
    horizon: int,
    n_episodes: int,
) -> float:
    rng, env_rng = jax.random.split(rng)
    bstate = make_batched_state(terrain, num_envs, env_rng, randomize_agent_idx=False)
    # Keep player assignment fixed so obs0/obs1 correspond to physical player0/player1.
    bstate = bstate.replace(agent_idx=jnp.zeros((num_envs,), dtype=jnp.int32))
    obs0, obs1 = encode_obs(terrain, bstate)

    ep_returns: List[float] = []
    ep_ret = np.zeros((num_envs,), dtype=np.float32)

    n_blocks = int(np.ceil(float(n_episodes) / float(num_envs)))
    max_steps = horizon * max(1, n_blocks)

    shaping = jnp.asarray(0.0, dtype=jnp.float32)  # sparse-only eval
    player_order_actions = False

    for _ in range(max_steps):
        rng, k0, k1, kres = jax.random.split(rng, 4)

        a0 = act0(obs0, k0, states=bstate.states, agent_idx=np.asarray(bstate.agent_idx))
        a1 = act1(obs1, k1, states=bstate.states, agent_idx=np.asarray(bstate.agent_idx))

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


def _load_first_available(
    ppo_runs_dir: Path,
    run_names: Tuple[str, ...],
    *,
    seed: int,
    ckpt: str,
) -> Tuple[str, dict]:
    last_err: Optional[Exception] = None
    for name in run_names:
        try:
            return name, _load_ppo_params(ppo_runs_dir, name, seed, ckpt=ckpt)
        except FileNotFoundError as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"Could not load PPO checkpoint for seed={seed} from any of: {run_names}. "
        f"Last error: {last_err}"
    )


def main() -> None:
    default_data_dir = _PKG_ROOT / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional. If set, evaluates only this raw seed for PPO conditions.",
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

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="How many seeds to include when auto-discovering (default: 5).",
    )

    parser.add_argument("--ppo_runs_dir", type=str, default=str(default_data_dir / "ppo_runs"))

    parser.add_argument(
        "--best_bc_paths",
        "--bc_paths_file",
        dest="best_bc_paths",
        type=str,
        default=str(default_data_dir / "bc_runs" / "best_bc_model_paths.pkl"),
        help="Pickle file mapping layout -> BC model path for train/test splits.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set, write results_{layout}.json into this directory.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which PPO checkpoint to evaluate: 'best' (default; fallback to final) or 'final'.",
    )
    args = parser.parse_args()

    layout = args.layout
    if layout not in _LAYOUT_KEY:
        raise KeyError(f"Unknown layout '{layout}'. Expected one of: {sorted(_LAYOUT_KEY.keys())}")

    layout_key = _LAYOUT_KEY[layout]
    horizon = 400
    num_slots = 1 if args.seed is not None else int(args.num_seeds)

    terrain = parse_layout(layout)

    best_paths_file = Path(args.best_bc_paths)
    bc_train_params = _load_bc_params(best_paths_file, "train", layout)
    hproxy_params = _load_bc_params(best_paths_file, "test", layout)

    bc_train_act = make_bc_act(bc_train_params, terrain, stochastic=True)
    hproxy_act = make_bc_act(hproxy_params, terrain, stochastic=True)

    ppo_runs_dir = Path(args.ppo_runs_dir)

    # Prefer the explicit legacy names first, but keep fallbacks for older dirs.
    ppo_sp_runs = (f"ppo_sp_{layout}",)
    ppo_bc_runs = (f"ppo_bc_train_{layout}", f"ppo_bc_{layout}")
    ppo_gs_runs = (f"ppo_bc_test_{layout}", f"ppo_bc_test_{layout}_v0")

    sp_seeds = _seeds_for_group(
        ppo_runs_dir=ppo_runs_dir,
        run_names=ppo_sp_runs,
        num_seeds=num_slots,
        seed_override=args.seed,
        group_name="SP",
    )
    bc_seeds = _seeds_for_group(
        ppo_runs_dir=ppo_runs_dir,
        run_names=ppo_bc_runs,
        num_seeds=num_slots,
        seed_override=args.seed,
        group_name="PPOBC",
    )
    gs_seeds = _seeds_for_group(
        ppo_runs_dir=ppo_runs_dir,
        run_names=ppo_gs_runs,
        num_seeds=num_slots,
        seed_override=args.seed,
        group_name="GoldStandard",
    )

    row: Dict[str, dict] = {
        "SP_SP": {},
        "SP_HProxy": {},
        "PPOBC_HProxy": {},
        "BC_HProxy": {},
        "SP_HProxy_sw": {},
        "PPOBC_HProxy_sw": {},
        "BC_HProxy_sw": {},
        "gold_standard": None,
    }

    # BC+HProxy does not depend on PPO seeds; compute once and replicate.
    rng = jax.random.PRNGKey(0)
    v_bc_hp = _eval_joint(
        terrain,
        bc_train_act,
        hproxy_act,
        rng=rng,
        num_envs=args.num_envs,
        horizon=horizon,
        n_episodes=args.n_episodes,
    )
    v_bc_hp_sw = _eval_joint(
        terrain,
        hproxy_act,
        bc_train_act,
        rng=rng,
        num_envs=args.num_envs,
        horizon=horizon,
        n_episodes=args.n_episodes,
    )
    for seed_idx in range(num_slots):
        row["BC_HProxy"][seed_idx] = v_bc_hp
        row["BC_HProxy_sw"][seed_idx] = v_bc_hp_sw

    # SP-based conditions
    for seed_idx, seed in enumerate(sp_seeds):
        sp_name, sp_params = _load_first_available(
            ppo_runs_dir, ppo_sp_runs, seed=seed, ckpt=args.ckpt
        )
        sp_act = make_ppo_act(sp_params, stochastic=True)

        rng = jax.random.PRNGKey(0)
        v_sp_sp = _eval_joint(
            terrain,
            sp_act,
            sp_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        )
        v_sp_hp = _eval_joint(
            terrain,
            sp_act,
            hproxy_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        )
        v_sp_hp_sw = _eval_joint(
            terrain,
            hproxy_act,
            sp_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        )

        row["SP_SP"][seed_idx] = v_sp_sp
        row["SP_HProxy"][seed_idx] = v_sp_hp
        row["SP_HProxy_sw"][seed_idx] = v_sp_hp_sw

        print(
            {
                "layout": layout,
                "layout_key": layout_key,
                "seed": seed,
                "seed_idx": seed_idx,
                "ckpt": args.ckpt,
                "ppo_sp_run_name": sp_name,
                "SP_SP": v_sp_sp,
                "SP_HProxy": v_sp_hp,
                "SP_HProxy_sw": v_sp_hp_sw,
            }
        )

    # PPO_BC-based conditions
    for seed_idx, seed in enumerate(bc_seeds):
        bc_name, bc_params = _load_first_available(
            ppo_runs_dir, ppo_bc_runs, seed=seed, ckpt=args.ckpt
        )
        ppo_bc_act = make_ppo_act(bc_params, stochastic=True)

        rng = jax.random.PRNGKey(0)
        v_ppobc_hp = _eval_joint(
            terrain,
            ppo_bc_act,
            hproxy_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        )
        v_ppobc_hp_sw = _eval_joint(
            terrain,
            hproxy_act,
            ppo_bc_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        )

        row["PPOBC_HProxy"][seed_idx] = v_ppobc_hp
        row["PPOBC_HProxy_sw"][seed_idx] = v_ppobc_hp_sw

        print(
            {
                "layout": layout,
                "layout_key": layout_key,
                "seed": seed,
                "seed_idx": seed_idx,
                "ckpt": args.ckpt,
                "ppo_bc_run_name": bc_name,
                "PPOBC_HProxy": v_ppobc_hp,
                "PPOBC_HProxy_sw": v_ppobc_hp_sw,
            }
        )

    # Gold standard: PPO trained with HProxy, paired with HProxy.
    gold_vals: List[float] = []
    for seed_idx, seed in enumerate(gs_seeds):
        gs_name, gs_params = _load_first_available(
            ppo_runs_dir, ppo_gs_runs, seed=seed, ckpt=args.ckpt
        )
        gs_act = make_ppo_act(gs_params, stochastic=True)

        rng = jax.random.PRNGKey(0)
        v_gs_0 = _eval_joint(
            terrain,
            gs_act,
            hproxy_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        )
        v_gs_1 = _eval_joint(
            terrain,
            hproxy_act,
            gs_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        )
        gold_vals.append(0.5 * (v_gs_0 + v_gs_1))

        print(
            {
                "layout": layout,
                "layout_key": layout_key,
                "seed": seed,
                "seed_idx": seed_idx,
                "ckpt": args.ckpt,
                "ppo_gold_run_name": gs_name,
                "gold_standard_seed": gold_vals[-1],
            }
        )

    row["gold_standard"] = float(np.mean(np.asarray(gold_vals, dtype=np.float64))) if gold_vals else None

    out_obj = {layout_key: row}

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"results_{layout}.json"
        out_path.write_text(json.dumps(out_obj, indent=2) + "\n")
        print(f"✓ Wrote: {out_path}")


if __name__ == "__main__":
    main()
