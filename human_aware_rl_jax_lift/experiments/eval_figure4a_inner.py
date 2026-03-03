"""Figure 4A evaluation (inner loop).

This script evaluates trained policies and writes two artifacts:

1) Notebook-faithful raw results (matches the Figure 4A notebook pipeline)
   Written to:
     {out_dir}/results_raw_{layout}.json

   Schema:
     raw[layout_key][experiment_key][seed_idx] -> float

   where experiment_key uses the notebook naming, e.g.:
     - PPO_SP+PPO_SP
     - PPO_SP+BC_test_0 / PPO_SP+BC_test_1
     - PPO_BC_train+BC_test_0 / PPO_BC_train+BC_test_1
     - PPO_BC_test+BC_test_0 / PPO_BC_test+BC_test_1
     - BC_train+BC_test_0 / BC_train+BC_test_1

2) Legacy adapter results (kept for figure4a.py / prepare_results.py)
   Written to:
     {out_dir}/results_{layout}.json

   Schema:
     legacy[layout_key][condition][seed_idx] -> float
     legacy[layout_key]["gold_standard"] -> float

Key notebook-faithful behaviors
------------------------------
- Uses separate, hardcoded seed families (as in the notebook):
    ppo_sp_seeds = [2229, 7649, 7225, 9807, 386]
    ppo_bc_seeds["bc_train"] = [9456, 1887, 5578, 5987, 516]
    ppo_bc_seeds["bc_test"]  = [2888, 7424, 7360, 4467, 184]

- Checkpoint semantics (as in the notebook):
    SP PPO           -> best (fallback to final if best missing)
    PPO_BC train/test-> final (fallback to best if final missing)

- Preserves the two gold-standard reference lines separately:
    PPO_BC_test+BC_test_0 and PPO_BC_test+BC_test_1

- Defaults to 40 evaluation episodes ("num_rounds" in the notebook).

Evaluation notes
----------------
- Evaluation uses sparse reward only (shaping = 0.0) and horizon = 400.
- Policies are evaluated stochastically.
- BC_test_0 vs BC_test_1 correspond to swapping player order in evaluation.

Debugging / progress logs
-------------------------
This script emits timestamped progress logs (JSON-ish dicts) to help diagnose slow
imports, XLA compilation, checkpoint loading, or rollout evaluation stalls.
"""

import argparse
import json
import pickle
import sys
import time
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

# Notebook seed families (do not merge across PPO families)
_PPO_SP_SEEDS = [2229, 7649, 7225, 9807, 386]
_PPO_BC_SEEDS = {
    "bc_train": [9456, 1887, 5578, 5987, 516],
    "bc_test": [2888, 7424, 7360, 4467, 184],
}


def _ts(t0: float) -> str:
    return f"+{time.time() - t0:.1f}s"


def _log(t0: float, msg: str, **kvs) -> None:
    payload = {"t": _ts(t0), "msg": msg}
    payload.update(kvs)
    print(payload)
    sys.stdout.flush()


def _timeit(t0: float, label: str, fn: Callable[[], float], **kvs) -> float:
    _log(t0, f"begin:{label}", **kvs)
    t1 = time.time()
    out = fn()
    _log(t0, f"end:{label}", seconds=round(time.time() - t1, 3), value=float(out), **kvs)
    return out


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
    ckpt: str,
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

    # Notebook semantics:
    # - SP uses ckpt='best' -> try best then final
    # - PPO_BC uses ckpt='final' -> try final then best
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


def _ensure_seed_list(seeds: List[int], n: int, *, name: str) -> List[int]:
    if len(seeds) < n:
        raise ValueError(f"Seed list '{name}' has length {len(seeds)} but need {n}: {seeds}")
    return seeds[:n]


def _to_legacy(layout_key: str, raw_row: Dict[str, Dict[int, float]]) -> Dict[str, dict]:
    legacy_row: Dict[str, dict] = {
        "SP_SP": {},
        "SP_HProxy": {},
        "SP_HProxy_sw": {},
        "PPOBC_HProxy": {},
        "PPOBC_HProxy_sw": {},
        "BC_HProxy": {},
        "BC_HProxy_sw": {},
        "gold_standard": None,
    }

    # Map raw keys to legacy keys
    for seed_idx, v in raw_row["PPO_SP+PPO_SP"].items():
        legacy_row["SP_SP"][seed_idx] = v
    for seed_idx, v in raw_row["PPO_SP+BC_test_0"].items():
        legacy_row["SP_HProxy"][seed_idx] = v
    for seed_idx, v in raw_row["PPO_SP+BC_test_1"].items():
        legacy_row["SP_HProxy_sw"][seed_idx] = v

    for seed_idx, v in raw_row["PPO_BC_train+BC_test_0"].items():
        legacy_row["PPOBC_HProxy"][seed_idx] = v
    for seed_idx, v in raw_row["PPO_BC_train+BC_test_1"].items():
        legacy_row["PPOBC_HProxy_sw"][seed_idx] = v

    for seed_idx, v in raw_row["BC_train+BC_test_0"].items():
        legacy_row["BC_HProxy"][seed_idx] = v
    for seed_idx, v in raw_row["BC_train+BC_test_1"].items():
        legacy_row["BC_HProxy_sw"][seed_idx] = v

    # Legacy expects a single scalar gold_standard. Preserve backwards compatibility by
    # averaging the two start-index conditions, then averaging across seeds.
    gold_seed_vals: List[float] = []
    for seed_idx in raw_row["PPO_BC_test+BC_test_0"].keys():
        v0 = raw_row["PPO_BC_test+BC_test_0"][seed_idx]
        v1 = raw_row["PPO_BC_test+BC_test_1"][seed_idx]
        gold_seed_vals.append(0.5 * (v0 + v1))
    legacy_row["gold_standard"] = float(np.mean(np.asarray(gold_seed_vals, dtype=np.float64))) if gold_seed_vals else None

    return {layout_key: legacy_row}


def main() -> None:
    t0 = time.time()
    _log(t0, "start", argv=" ".join(sys.argv))

    default_data_dir = _PKG_ROOT / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)

    parser.add_argument("--num_envs", type=int, default=30)

    parser.add_argument(
        "--n_episodes",
        "--num_rounds",
        "--num_games",
        dest="n_episodes",
        type=int,
        default=40,
        help="Number of episodes (games) to evaluate per condition (default: 40, matches notebook).",
    )

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="How many seeds from each notebook seed family to evaluate (default: 5).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional debug override: if set, evaluates only this raw seed for all PPO families.",
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
        help="If set, write results_{layout}.json and results_raw_{layout}.json into this directory.",
    )

    parser.add_argument(
        "--sp_ckpt",
        type=str,
        default="best",
        choices=["best", "final"],
        help="SP PPO checkpoint preference (default: best, matches notebook).",
    )

    parser.add_argument(
        "--bc_ckpt",
        type=str,
        default="final",
        choices=["best", "final"],
        help="PPO_BC train/test checkpoint preference (default: final, matches notebook).",
    )

    args = parser.parse_args()

    _log(
        t0,
        "parsed_args",
        layout=args.layout,
        n_episodes=int(args.n_episodes),
        num_envs=int(args.num_envs),
        num_seeds=int(args.num_seeds),
        seed_override=args.seed,
        sp_ckpt=args.sp_ckpt,
        bc_ckpt=args.bc_ckpt,
        ppo_runs_dir=args.ppo_runs_dir,
        bc_paths_file=args.best_bc_paths,
        out_dir=args.out_dir,
        jax_devices=[str(d) for d in jax.devices()],
    )

    layout = args.layout
    if layout not in _LAYOUT_KEY:
        raise KeyError(f"Unknown layout '{layout}'. Expected one of: {sorted(_LAYOUT_KEY.keys())}")

    layout_key = _LAYOUT_KEY[layout]
    horizon = 400
    n_seeds = int(args.num_seeds)

    _log(t0, "parse_layout", layout=layout)
    terrain = parse_layout(layout)
    _log(t0, "parsed_layout", layout=layout)

    best_paths_file = Path(args.best_bc_paths)
    _log(t0, "load_bc_params", split="train", file=str(best_paths_file))
    bc_train_params = _load_bc_params(best_paths_file, "train", layout)
    _log(t0, "load_bc_params", split="test", file=str(best_paths_file))
    bc_test_params = _load_bc_params(best_paths_file, "test", layout)

    bc_train_act = make_bc_act(bc_train_params, terrain, stochastic=True)
    bc_test_act = make_bc_act(bc_test_params, terrain, stochastic=True)

    ppo_runs_dir = Path(args.ppo_runs_dir)

    # Run directory names
    ppo_sp_runs = (f"ppo_sp_{layout}",)
    ppo_bc_train_runs = (f"ppo_bc_train_{layout}", f"ppo_bc_{layout}")
    ppo_bc_test_runs = (f"ppo_bc_test_{layout}", f"ppo_bc_test_{layout}_v0")

    if args.seed is not None:
        sp_seeds = [int(args.seed)]
        bc_train_seeds = [int(args.seed)]
        bc_test_seeds = [int(args.seed)]
    else:
        sp_seeds = _ensure_seed_list(_PPO_SP_SEEDS, n_seeds, name="PPO_SP")
        bc_train_seeds = _ensure_seed_list(_PPO_BC_SEEDS["bc_train"], n_seeds, name="PPO_BC_train")
        bc_test_seeds = _ensure_seed_list(_PPO_BC_SEEDS["bc_test"], n_seeds, name="PPO_BC_test")

    _log(t0, "seed_families", sp_seeds=sp_seeds, bc_train_seeds=bc_train_seeds, bc_test_seeds=bc_test_seeds)

    raw_row: Dict[str, Dict[int, float]] = {
        "PPO_SP+PPO_SP": {},
        "PPO_SP+BC_test_0": {},
        "PPO_SP+BC_test_1": {},
        "PPO_BC_train+BC_test_0": {},
        "PPO_BC_train+BC_test_1": {},
        "PPO_BC_test+BC_test_0": {},
        "PPO_BC_test+BC_test_1": {},
        "BC_train+BC_test_0": {},
        "BC_train+BC_test_1": {},
    }

    # BC baseline does not depend on PPO seeds; compute once and replicate.
    rng = jax.random.PRNGKey(0)
    v_bc0 = _timeit(
        t0,
        "eval_bc_baseline_0",
        lambda: _eval_joint(
            terrain,
            bc_train_act,
            bc_test_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        ),
        num_envs=int(args.num_envs),
        n_episodes=int(args.n_episodes),
    )

    rng = jax.random.PRNGKey(0)
    v_bc1 = _timeit(
        t0,
        "eval_bc_baseline_1",
        lambda: _eval_joint(
            terrain,
            bc_test_act,
            bc_train_act,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            n_episodes=args.n_episodes,
        ),
        num_envs=int(args.num_envs),
        n_episodes=int(args.n_episodes),
    )

    for seed_idx in range(len(sp_seeds) if args.seed is not None else n_seeds):
        raw_row["BC_train+BC_test_0"][seed_idx] = v_bc0
        raw_row["BC_train+BC_test_1"][seed_idx] = v_bc1

    # SP PPO
    for seed_idx, seed in enumerate(sp_seeds):
        _log(t0, "load_sp_ckpt", seed=int(seed), seed_idx=int(seed_idx), ckpt=args.sp_ckpt)
        sp_name, sp_params = _load_first_available(ppo_runs_dir, ppo_sp_runs, seed=seed, ckpt=args.sp_ckpt)
        _log(t0, "loaded_sp_ckpt", seed=int(seed), seed_idx=int(seed_idx), run_name=sp_name)
        sp_act = make_ppo_act(sp_params, stochastic=True)

        rng = jax.random.PRNGKey(0)
        raw_row["PPO_SP+PPO_SP"][seed_idx] = _timeit(
            t0,
            "eval:PPO_SP+PPO_SP",
            lambda: _eval_joint(
                terrain,
                sp_act,
                sp_act,
                rng=rng,
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        rng = jax.random.PRNGKey(0)
        raw_row["PPO_SP+BC_test_0"][seed_idx] = _timeit(
            t0,
            "eval:PPO_SP+BC_test_0",
            lambda: _eval_joint(
                terrain,
                sp_act,
                bc_test_act,
                rng=rng,
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        rng = jax.random.PRNGKey(0)
        raw_row["PPO_SP+BC_test_1"][seed_idx] = _timeit(
            t0,
            "eval:PPO_SP+BC_test_1",
            lambda: _eval_joint(
                terrain,
                bc_test_act,
                sp_act,
                rng=rng,
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        print(
            {
                "layout": layout,
                "layout_key": layout_key,
                "seed": seed,
                "seed_idx": seed_idx,
                "ppo_sp_run_name": sp_name,
                "sp_ckpt": args.sp_ckpt,
                "PPO_SP+PPO_SP": raw_row["PPO_SP+PPO_SP"][seed_idx],
                "PPO_SP+BC_test_0": raw_row["PPO_SP+BC_test_0"][seed_idx],
                "PPO_SP+BC_test_1": raw_row["PPO_SP+BC_test_1"][seed_idx],
            }
        )
        sys.stdout.flush()

    # PPO_BC_train
    for seed_idx, seed in enumerate(bc_train_seeds):
        _log(t0, "load_ppo_bc_train_ckpt", seed=int(seed), seed_idx=int(seed_idx), ckpt=args.bc_ckpt)
        bc_name, bc_params = _load_first_available(
            ppo_runs_dir, ppo_bc_train_runs, seed=seed, ckpt=args.bc_ckpt
        )
        _log(t0, "loaded_ppo_bc_train_ckpt", seed=int(seed), seed_idx=int(seed_idx), run_name=bc_name)
        bc_act = make_ppo_act(bc_params, stochastic=True)

        rng = jax.random.PRNGKey(0)
        raw_row["PPO_BC_train+BC_test_0"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_train+BC_test_0",
            lambda: _eval_joint(
                terrain,
                bc_act,
                bc_test_act,
                rng=rng,
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        rng = jax.random.PRNGKey(0)
        raw_row["PPO_BC_train+BC_test_1"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_train+BC_test_1",
            lambda: _eval_joint(
                terrain,
                bc_test_act,
                bc_act,
                rng=rng,
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        print(
            {
                "layout": layout,
                "layout_key": layout_key,
                "seed": seed,
                "seed_idx": seed_idx,
                "ppo_bc_train_run_name": bc_name,
                "bc_ckpt": args.bc_ckpt,
                "PPO_BC_train+BC_test_0": raw_row["PPO_BC_train+BC_test_0"][seed_idx],
                "PPO_BC_train+BC_test_1": raw_row["PPO_BC_train+BC_test_1"][seed_idx],
            }
        )
        sys.stdout.flush()

    # PPO_BC_test (gold reference lines) — keep _0 and _1 separate
    for seed_idx, seed in enumerate(bc_test_seeds):
        _log(t0, "load_ppo_bc_test_ckpt", seed=int(seed), seed_idx=int(seed_idx), ckpt=args.bc_ckpt)
        gs_name, gs_params = _load_first_available(
            ppo_runs_dir, ppo_bc_test_runs, seed=seed, ckpt=args.bc_ckpt
        )
        _log(t0, "loaded_ppo_bc_test_ckpt", seed=int(seed), seed_idx=int(seed_idx), run_name=gs_name)
        gs_act = make_ppo_act(gs_params, stochastic=True)

        rng = jax.random.PRNGKey(0)
        raw_row["PPO_BC_test+BC_test_0"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_test+BC_test_0",
            lambda: _eval_joint(
                terrain,
                gs_act,
                bc_test_act,
                rng=rng,
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        rng = jax.random.PRNGKey(0)
        raw_row["PPO_BC_test+BC_test_1"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_test+BC_test_1",
            lambda: _eval_joint(
                terrain,
                bc_test_act,
                gs_act,
                rng=rng,
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        print(
            {
                "layout": layout,
                "layout_key": layout_key,
                "seed": seed,
                "seed_idx": seed_idx,
                "ppo_bc_test_run_name": gs_name,
                "bc_ckpt": args.bc_ckpt,
                "PPO_BC_test+BC_test_0": raw_row["PPO_BC_test+BC_test_0"][seed_idx],
                "PPO_BC_test+BC_test_1": raw_row["PPO_BC_test+BC_test_1"][seed_idx],
            }
        )
        sys.stdout.flush()

    out_raw = {layout_key: raw_row}
    out_legacy = _to_legacy(layout_key, raw_row)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / f"results_raw_{layout}.json"
        raw_path.write_text(json.dumps(out_raw, indent=2) + "\n")
        _log(t0, "wrote_raw", path=str(raw_path))

        legacy_path = out_dir / f"results_{layout}.json"
        legacy_path.write_text(json.dumps(out_legacy, indent=2) + "\n")
        _log(t0, "wrote_legacy", path=str(legacy_path))

    _log(t0, "done")


if __name__ == "__main__":
    main()
