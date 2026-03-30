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

- Checkpoint semantics:
    SP PPO           -> best (fallback to final if best missing)
    PPO_BC train/test-> best_after_sp_anneal (fallback to final if missing)
                        This picks the best checkpoint recorded only after
                        self-play has been fully annealed to zero, so the
                        selection is not confounded by self-play mixing.

- Preserves the two gold-standard reference lines separately:
    PPO_BC_test+BC_test_0 and PPO_BC_test+BC_test_1

- Defaults to 40 evaluation episodes ("num_rounds" in the notebook).

Evaluation notes
----------------
- Evaluation uses sparse reward only (shaping = 0.0) and horizon = 400.
- Policies are evaluated stochastically.
- BC_test_0 vs BC_test_1 correspond to swapping player order in evaluation.
- Each (condition, seed_idx) pair uses a unique RNG key so that evaluation
  rollouts are independent across seeds, yielding meaningful standard errors.

Debugging / progress logs
-------------------------
This script emits timestamped progress logs (JSON-ish dicts) to help diagnose slow
imports, XLA compilation, checkpoint loading, rollout evaluation, or stalls.

Performance note
----------------
The BC policy action selection is vectorized + jitted for fast evaluation.

Unstuck heuristic
-----------------
If a BC agent hasn't moved for `stuck_time` steps (same position AND orientation),
recently-taken actions are masked and probabilities renormalized. This prevents
BC agents from getting permanently stuck in corners or against walls.
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
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.agents.bc.model import BCPolicy
from human_aware_rl_jax_lift.agents.ppo.model import ActorCriticCNN
from human_aware_rl_jax_lift.encoding.bc_features import build_dist_table, featurize_state_64, get_mlp_for_layout
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.config import get_tf_compat
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

# RNG offsets per condition — keep them well-separated so keys never collide
# across (condition, seed_idx) pairs.
_COND_RNG_OFFSET = {
    "PPO_SP+PPO_SP":          0,
    "PPO_SP+BC_test_0":       1000,
    "PPO_SP+BC_test_1":       2000,
    "PPO_BC_train+BC_test_0": 3000,
    "PPO_BC_train+BC_test_1": 4000,
    "PPO_BC_test+BC_test_0":  5000,
    "PPO_BC_test+BC_test_1":  6000,
    "BC_train+BC_test_0":     7000,
    "BC_train+BC_test_1":     8000,
}


def _cond_rng(cond_name: str, seed_idx: int) -> jax.Array:
    """Return a unique, reproducible RNG key for a given (condition, seed_idx) pair.

    Using a unique key per (condition, seed_idx) ensures that evaluation
    rollouts are independent across seeds, yielding non-zero standard errors.
    """
    offset = _COND_RNG_OFFSET[cond_name]
    return jax.random.PRNGKey(offset + seed_idx)


def _ts(t0: float) -> str:
    return f"+{time.time() - t0:.1f}s"


def _log(t0: float, msg: str, **kvs) -> None:
    suffix = "".join(f"  {k}={v}" for k, v in kvs.items())
    print(f"[{_ts(t0)}] {msg}{suffix}")
    sys.stdout.flush()


def _timeit(t0: float, label: str, fn: Callable[[], float], **kvs) -> float:
    suffix = "".join(f"  {k}={v}" for k, v in kvs.items())
    print(f"[{_ts(t0)}] >> {label}{suffix}")
    sys.stdout.flush()
    t1 = time.time()
    out = fn()
    elapsed = round(time.time() - t1, 1)
    print(f"[{_ts(t0)}] << {label}  score={out:.2f}  ({elapsed}s){suffix}")
    sys.stdout.flush()
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
    post_sp_dir = seed_dir / "best_after_sp_anneal"

    def _try(dir_: Path):
        p = dir_ / "params.pkl"
        if not p.exists():
            return None
        with p.open("rb") as f:
            payload = pickle.load(f)
        return payload.get("params", payload) if isinstance(payload, dict) else payload

    if ckpt not in ("best", "final", "best_after_sp_anneal"):
        raise ValueError(f"Invalid ckpt='{ckpt}'. Expected 'best', 'final', or 'best_after_sp_anneal'.")

    if ckpt == "best":
        # SP semantics: prefer best, fall back to final
        out = _try(best_dir)
        if out is not None:
            return out
        out = _try(final_dir)
        if out is not None:
            return out

    elif ckpt == "best_after_sp_anneal":
        # PPO_BC semantics: prefer post-SP-anneal best, fall back to final
        out = _try(post_sp_dir)
        if out is not None:
            return out
        out = _try(final_dir)
        if out is not None:
            return out

    else:  # ckpt == "final"
        out = _try(final_dir)
        if out is not None:
            return out
        out = _try(best_dir)
        if out is not None:
            return out

    raise FileNotFoundError(
        f"Could not find PPO params for {run_name}/seed{seed} with ckpt='{ckpt}'. "
        f"Looked for {post_sp_dir/'params.pkl'}, {best_dir/'params.pkl'}, "
        f"and {final_dir/'params.pkl'}."
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


ACTION_STAY = 4  # 0=N,1=S,2=E,3=W,4=STAY,5=INTERACT


def make_bc_act(params, terrain, layout_name: str, *, stochastic: bool = True, no_waits: bool = False, tf_compat: bool = False) -> Callable:
    """Vectorized BC policy action function.

    The `agent_idx` argument indicates which physical player is the *training
    agent*, and this policy acts as the *other* agent.

    Includes the "unstuck" heuristic: if the partner's position AND orientation
    haven't changed for `stuck_time` steps, recently-taken actions are masked
    and probabilities renormalized.

    When no_waits=True, STAY (action 4) is masked and probabilities renormalized.
    """

    model = BCPolicy()
    apply_fn = jax.jit(model.apply)

    # Load MediumLevelPlanner once, then precompute the full distance table.
    # build_dist_table is a one-time O(walkable_cells × 4 × target_positions) BFS
    # at startup; the resulting JAX array enables pure-JAX O(1) lookups at eval time,
    # making featurize_state_64 fully vmappable and GPU-resident (no CPU syncs).
    mlp = get_mlp_for_layout(layout_name)
    dist_table = build_dist_table(mlp, terrain)

    stuck_time = 3
    hist_len = stuck_time + 1

    # Stateful (host-side) per-env history buffers.
    _pos_hist: Optional[np.ndarray] = None  # (N, hist_len, 2)
    _or_hist: Optional[np.ndarray] = None   # (N, hist_len)   — orientation
    _act_hist: Optional[np.ndarray] = None  # (N, hist_len)
    _filled: Optional[np.ndarray] = None  # (N,)

    # Compile once: jit(vmap(featurize)) traces featurize_state_64 on the first call
    # and caches the XLA computation for all subsequent steps (no per-step re-tracing).
    _featurize_batch = jax.jit(
        jax.vmap(lambda s: featurize_state_64(terrain, s, dist_table=dist_table, tf_compat=tf_compat))
    )

    def _features_other(states, agent_idx_np: np.ndarray) -> jnp.ndarray:
        f0, f1 = _featurize_batch(states)
        return jnp.where(jnp.asarray(agent_idx_np)[:, None] == 0, f1, f0)

    @jax.jit
    def _unstuck_adjust_batch(
        probs: jnp.ndarray, stuck_mask: jnp.ndarray, blocked_actions: jnp.ndarray
    ) -> jnp.ndarray:
        # probs: (N, A), stuck_mask: (N,) bool, blocked_actions: (N, stuck_time) int32
        # Block only last stuck_time=3 actions, not all 4 in history.
        # Skip adjustment if all movement dirs (0-3) would be blocked.
        n, _a = probs.shape
        mask = jnp.ones_like(probs)
        mask = mask.at[jnp.arange(n)[:, None], blocked_actions].set(0.0)
        out = probs * mask
        norm = out.sum(axis=-1, keepdims=True)
        movement_probs = out[:, :4].sum(axis=-1, keepdims=True)
        skip = movement_probs <= 0.0
        out_norm = jnp.where(norm > 0, out / norm, probs)
        result = jnp.where(skip, probs, out_norm)
        return jnp.where(stuck_mask[:, None], result, probs)

    @jax.jit
    def _apply_model(
        feats: jnp.ndarray,
        rng: jax.Array,
        stuck_mask: jnp.ndarray,
        blocked_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        logits = apply_fn(params, feats)
        probs = jnn.softmax(logits, axis=-1)
        if no_waits:
            probs = probs.at[:, ACTION_STAY].set(0.0)
            norm = probs.sum(axis=-1, keepdims=True)
            probs = jnp.where(norm > 0, probs / norm, probs)
        probs = _unstuck_adjust_batch(probs, stuck_mask, blocked_actions)
        if stochastic:
            rngs = jax.random.split(rng, probs.shape[0])
            logp = jnp.log(jnp.clip(probs, 1e-20, 1.0))
            a = jax.vmap(lambda lg, r: jax.random.categorical(r, lg, axis=-1))(logp, rngs)
        else:
            a = jnp.argmax(probs, axis=-1)
        return a.astype(jnp.int32)

    def act(_obs_batch, rng: jax.Array, *, states, agent_idx):
        nonlocal _pos_hist, _or_hist, _act_hist, _filled

        agent_idx_np = np.asarray(agent_idx, dtype=np.int32)
        n = int(agent_idx_np.shape[0])

        if _pos_hist is None or _pos_hist.shape[0] != n:
            _pos_hist = np.zeros((n, hist_len, 2), dtype=np.int32)
            _or_hist  = np.zeros((n, hist_len),    dtype=np.int32)
            _act_hist = np.zeros((n, hist_len), dtype=np.int32)
            _filled = np.zeros((n,), dtype=np.int32)

        # Reset history on environment resets (timestep == 0 in our state container).
        timestep = np.asarray(states.timestep, dtype=np.int32)  # (N,)
        reset_mask = timestep == 0
        if np.any(reset_mask):
            _pos_hist[reset_mask] = 0
            _or_hist[reset_mask]  = 0
            _act_hist[reset_mask] = 0
            _filled[reset_mask] = 0

        # Select the "other" physical player's position and orientation.
        # "other" here means the player this BC policy is controlling
        # (mirrors BCPartner semantics where agent_idx is the *training* agent).
        other_idx = 1 - agent_idx_np
        player_pos = np.asarray(states.player_pos, dtype=np.int32)  # (N, 2, 2)
        player_or  = np.asarray(states.player_or,  dtype=np.int32)  # (N, 2)
        pos_other = player_pos[np.arange(n), other_idx]  # (N, 2)
        or_other  = player_or[np.arange(n),  other_idx]  # (N,)

        # Update pos/or history with the CURRENT observation before computing stuck.
        # Including the current state ensures the stuck check fires at the correct step.
        _pos_hist[:, :-1, :] = _pos_hist[:, 1:, :]
        _or_hist[:, :-1]     = _or_hist[:, 1:]
        _pos_hist[:, -1, :] = pos_other
        _or_hist[:, -1]     = or_other
        _filled = np.minimum(_filled + 1, hist_len + 1)

        # Stuck condition: same position AND same orientation for hist_len steps.
        # Gate with _filled > hist_len (requires hist_len+1 pushes = hist_len prior steps
        # + current step) to match TF's None-in-history guard.
        same_pos = np.all(_pos_hist == _pos_hist[:, :1, :], axis=(1, 2))
        same_or  = np.all(_or_hist  == _or_hist[:, :1],     axis=1)
        stuck_mask_np = (_filled > hist_len) & same_pos & same_or

        # Block only last stuck_time actions.
        last_actions = _act_hist[:, -stuck_time:]  # (N, 3)
        feats = _features_other(states, agent_idx_np)
        a = _apply_model(
            feats,
            rng,
            jnp.asarray(stuck_mask_np, dtype=jnp.bool_),
            jnp.asarray(last_actions, dtype=jnp.int32),
        )
        a_np = np.asarray(a, dtype=np.int32)

        # Update action history with chosen action (pos/or already updated above).
        _act_hist[:, :-1] = _act_hist[:, 1:]
        _act_hist[:, -1] = a_np

        return a_np

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
    tf_compat: bool = True,
) -> float:
    rng, env_rng = jax.random.split(rng)
    bstate = make_batched_state(terrain, num_envs, env_rng, randomize_agent_idx=False)

    # Keep env interpretation fixed: training_actions = physical player0, other_actions = physical player1.
    bstate = bstate.replace(agent_idx=jnp.zeros((num_envs,), dtype=jnp.int32))

    obs0, obs1 = encode_obs(terrain, bstate, tf_compat=tf_compat)

    # Partner-policy helpers interpret agent_idx as "who is the training agent"
    # and act as the other agent. For evaluation, we need explicit physical player0/player1 actions.
    # So we pass different agent_idx views to act0/act1:
    # - To get an action for physical player0 (other agent), pretend training agent is player1 (agent_idx=1).
    # - To get an action for physical player1 (other agent), pretend training agent is player0 (agent_idx=0).
    agent_idx_for_p0 = np.ones((num_envs,), dtype=np.int32)
    agent_idx_for_p1 = np.zeros((num_envs,), dtype=np.int32)

    ep_returns: List[float] = []
    ep_ret = np.zeros((num_envs,), dtype=np.float32)

    n_blocks = int(np.ceil(float(n_episodes) / float(num_envs)))
    max_steps = horizon * max(1, n_blocks)

    shaping = jnp.asarray(0.0, dtype=jnp.float32)  # sparse-only eval
    player_order_actions = False

    for _step in range(max_steps):
        rng, k0, k1, kres = jax.random.split(rng, 4)

        a0 = act0(obs0, k0, states=bstate.states, agent_idx=agent_idx_for_p0)
        a1 = act1(obs1, k1, states=bstate.states, agent_idx=agent_idx_for_p1)

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
            tf_compat=tf_compat,
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
    _log(t0, f"eval_figure4a_inner  argv={' '.join(sys.argv[1:])}")

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
        choices=["best", "final", "best_after_sp_anneal"],
        help="SP PPO checkpoint preference (default: best, matches notebook).",
    )

    parser.add_argument(
        "--tf_compat",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable tf_compat mode. Default: read from paper_config.yaml tf_compat flag.",
    )

    parser.add_argument(
        "--bc_ckpt",
        type=str,
        default="best_after_sp_anneal",
        choices=["best", "final", "best_after_sp_anneal"],
        help=(
            "PPO_BC train/test checkpoint preference "
            "(default: best_after_sp_anneal — picks the best checkpoint recorded only "
            "after self-play has been fully annealed to zero, avoiding confounding from "
            "self-play mixing at selection time; falls back to final if missing)."
        ),
    )

    args = parser.parse_args()

    # Resolve tf_compat: CLI override > paper_config.yaml > default True
    if args.tf_compat is None:
        args.tf_compat = get_tf_compat()

    _log(
        t0,
        f"layout={args.layout}  episodes={args.n_episodes}  envs={args.num_envs}  seeds={args.num_seeds}",
        sp_ckpt=args.sp_ckpt,
        bc_ckpt=args.bc_ckpt,
        device=str(jax.devices()[0]),
    )

    layout = args.layout
    if layout not in _LAYOUT_KEY:
        raise KeyError(f"Unknown layout '{layout}'. Expected one of: {sorted(_LAYOUT_KEY.keys())}")

    layout_key = _LAYOUT_KEY[layout]
    horizon = 400
    n_seeds = int(args.num_seeds)

    terrain = parse_layout(layout)
    _log(t0, f"Parsed layout: {layout}")

    best_paths_file = Path(args.best_bc_paths)
    bc_train_params = _load_bc_params(best_paths_file, "train", layout)
    bc_test_params = _load_bc_params(best_paths_file, "test", layout)
    _log(t0, f"BC params loaded (train + test)  file={best_paths_file.name}")

    bc_train_act = make_bc_act(bc_train_params, terrain, layout, stochastic=True, tf_compat=args.tf_compat)
    bc_test_act = make_bc_act(bc_test_params, terrain, layout, stochastic=True, tf_compat=args.tf_compat)

    ppo_runs_dir = Path(args.ppo_runs_dir)

    # Run directory names
    ppo_sp_runs = (f"ppo_sp_{layout}",)
    ppo_bc_train_runs = (f"ppo_bc_train_{layout}", f"ppo_bc_{layout}")
    ppo_bc_test_runs = (f"ppo_bc_test_{layout}", f"ppo_bc_test_{layout}_v0")

    sp_seeds = _ensure_seed_list(_PPO_SP_SEEDS, n_seeds, name="PPO_SP")
    bc_train_seeds = _ensure_seed_list(_PPO_BC_SEEDS["bc_train"], n_seeds, name="PPO_BC_train")
    bc_test_seeds = _ensure_seed_list(_PPO_BC_SEEDS["bc_test"], n_seeds, name="PPO_BC_test")

    _log(t0, f"Seeds  SP={sp_seeds}  BC_train={bc_train_seeds}  BC_test={bc_test_seeds}")

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

    # BC baseline does not depend on PPO seeds, but we still use per-seed_idx RNG
    # keys so that the replicated values reflect independent episode samples.
    for seed_idx in range(n_seeds):
        v_bc0 = _timeit(
            t0,
            "eval_bc_baseline_0",
            lambda: _eval_joint(
                terrain,
                bc_train_act,
                bc_test_act,
                rng=_cond_rng("BC_train+BC_test_0", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            num_envs=int(args.num_envs),
            n_episodes=int(args.n_episodes),
            seed_idx=seed_idx,
        )
        raw_row["BC_train+BC_test_0"][seed_idx] = v_bc0

        v_bc1 = _timeit(
            t0,
            "eval_bc_baseline_1",
            lambda: _eval_joint(
                terrain,
                bc_test_act,
                bc_train_act,
                rng=_cond_rng("BC_train+BC_test_1", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            num_envs=int(args.num_envs),
            n_episodes=int(args.n_episodes),
            seed_idx=seed_idx,
        )
        raw_row["BC_train+BC_test_1"][seed_idx] = v_bc1

    # SP PPO
    for seed_idx, seed in enumerate(sp_seeds):
        sp_name, sp_params = _load_first_available(ppo_runs_dir, ppo_sp_runs, seed=seed, ckpt=args.sp_ckpt)
        _log(t0, f"SP ckpt loaded  seed={seed_idx+1}/{n_seeds}  seed={seed}  run={sp_name}  ckpt={args.sp_ckpt}")
        sp_act = make_ppo_act(sp_params, stochastic=True)

        raw_row["PPO_SP+PPO_SP"][seed_idx] = _timeit(
            t0,
            "eval:PPO_SP+PPO_SP",
            lambda: _eval_joint(
                terrain,
                sp_act,
                sp_act,
                rng=_cond_rng("PPO_SP+PPO_SP", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        raw_row["PPO_SP+BC_test_0"][seed_idx] = _timeit(
            t0,
            "eval:PPO_SP+BC_test_0",
            lambda: _eval_joint(
                terrain,
                sp_act,
                bc_test_act,
                rng=_cond_rng("PPO_SP+BC_test_0", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        raw_row["PPO_SP+BC_test_1"][seed_idx] = _timeit(
            t0,
            "eval:PPO_SP+BC_test_1",
            lambda: _eval_joint(
                terrain,
                bc_test_act,
                sp_act,
                rng=_cond_rng("PPO_SP+BC_test_1", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        _log(
            t0,
            f"SP {seed_idx+1}/{n_seeds}  seed={seed}  run={sp_name}"
            f"  SP+SP={raw_row['PPO_SP+PPO_SP'][seed_idx]:.2f}"
            f"  SP+BC0={raw_row['PPO_SP+BC_test_0'][seed_idx]:.2f}"
            f"  SP+BC1={raw_row['PPO_SP+BC_test_1'][seed_idx]:.2f}",
        )

    # PPO_BC_train
    for seed_idx, seed in enumerate(bc_train_seeds):
        bc_name, bc_params = _load_first_available(
            ppo_runs_dir, ppo_bc_train_runs, seed=seed, ckpt=args.bc_ckpt
        )
        _log(t0, f"PPO_BC_train ckpt loaded  seed={seed_idx+1}/{n_seeds}  seed={seed}  run={bc_name}  ckpt={args.bc_ckpt}")
        bc_act = make_ppo_act(bc_params, stochastic=True)

        raw_row["PPO_BC_train+BC_test_0"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_train+BC_test_0",
            lambda: _eval_joint(
                terrain,
                bc_act,
                bc_test_act,
                rng=_cond_rng("PPO_BC_train+BC_test_0", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        raw_row["PPO_BC_train+BC_test_1"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_train+BC_test_1",
            lambda: _eval_joint(
                terrain,
                bc_test_act,
                bc_act,
                rng=_cond_rng("PPO_BC_train+BC_test_1", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        _log(
            t0,
            f"PPO_BC_train {seed_idx+1}/{n_seeds}  seed={seed}  run={bc_name}"
            f"  BC_train+BC0={raw_row['PPO_BC_train+BC_test_0'][seed_idx]:.2f}"
            f"  BC_train+BC1={raw_row['PPO_BC_train+BC_test_1'][seed_idx]:.2f}",
        )

    # PPO_BC_test (gold reference lines) — keep _0 and _1 separate
    for seed_idx, seed in enumerate(bc_test_seeds):
        gs_name, gs_params = _load_first_available(
            ppo_runs_dir, ppo_bc_test_runs, seed=seed, ckpt=args.bc_ckpt
        )
        _log(t0, f"PPO_BC_test ckpt loaded  seed={seed_idx+1}/{n_seeds}  seed={seed}  run={gs_name}  ckpt={args.bc_ckpt}")
        gs_act = make_ppo_act(gs_params, stochastic=True)

        raw_row["PPO_BC_test+BC_test_0"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_test+BC_test_0",
            lambda: _eval_joint(
                terrain,
                gs_act,
                bc_test_act,
                rng=_cond_rng("PPO_BC_test+BC_test_0", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        raw_row["PPO_BC_test+BC_test_1"][seed_idx] = _timeit(
            t0,
            "eval:PPO_BC_test+BC_test_1",
            lambda: _eval_joint(
                terrain,
                bc_test_act,
                gs_act,
                rng=_cond_rng("PPO_BC_test+BC_test_1", seed_idx),
                num_envs=args.num_envs,
                horizon=horizon,
                n_episodes=args.n_episodes,
                tf_compat=args.tf_compat,
            ),
            seed=int(seed),
            seed_idx=int(seed_idx),
        )

        _log(
            t0,
            f"PPO_BC_test (gold) {seed_idx+1}/{n_seeds}  seed={seed}  run={gs_name}"
            f"  BC_test+BC0={raw_row['PPO_BC_test+BC_test_0'][seed_idx]:.2f}"
            f"  BC_test+BC1={raw_row['PPO_BC_test+BC_test_1'][seed_idx]:.2f}",
        )

    out_raw = {layout_key: raw_row}
    out_legacy = _to_legacy(layout_key, raw_row)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / f"results_raw_{layout}.json"
        raw_path.write_text(json.dumps(out_raw, indent=2) + "\n")
        legacy_path = out_dir / f"results_{layout}.json"
        legacy_path.write_text(json.dumps(out_legacy, indent=2) + "\n")
        _log(t0, f"Wrote  {raw_path.name}  {legacy_path.name}")

    _log(t0, "Done.")


if __name__ == "__main__":
    main()
