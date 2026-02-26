"""PPO training loop using the fully-JAX runner (vmap + lax.scan).

Drop-in replacement for ppo_run.py for self-play (``other_agent_type='sp'``).
BC-partner training still falls back to ppo_run.py because BCPartner uses a
Python loop internally (stuck detection) that cannot be scanned.

Key differences from ppo_run.py
---------------------------------
1. Uses ``make_batched_state`` + ``make_rollout_fn`` from runner_jax.py.
2. Rollout is a single ``jax.lax.scan`` call — no Python step loop.
3. Metric ``float()`` conversion deferred to once per update (not per minibatch).
4. ``shaping_factor`` passed as JAX scalar to ``batched_step`` — no dict rebuild.
"""

import pickle
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.agents.ppo.train import compute_gae, create_train_state, ppo_update_step
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.reward_shaping import annealed_shaping_factor
from human_aware_rl_jax_lift.reproducibility.seed import set_global_seed

from .checkpoints import save_ppo_checkpoint, save_training_info
from .runner_jax import make_rollout_fn
from .vec_env_jax import encode_obs, make_batched_state


# ---------------------------------------------------------------------------
# Logging helpers  (identical to ppo_run.py for output parity)
# ---------------------------------------------------------------------------

def _log_table(kvs: Dict[str, object]) -> None:
    if not kvs:
        return
    key_w = max(len(k) for k in kvs)
    val_strs = {
        k: (f"{v:.8g}" if isinstance(v, float) else str(v))
        for k, v in kvs.items()
    }
    val_w = max(len(s) for s in val_strs.values())
    sep = "-" * (key_w + val_w + 7)
    print(sep)
    for k, vs in val_strs.items():
        print(f"| {k:<{key_w}} | {vs:>{val_w}} |")
    print(sep)
    sys.stdout.flush()


def _ts(t0: float) -> str:
    return f"+{time.time() - t0:.1f}s"


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return float("nan") if var_y < 1e-8 else float(1.0 - np.var(y_true - y_pred) / var_y)


def _current_lr(train_state, fallback: float) -> float:
    try:
        return float(train_state.opt_state.hyperparams["learning_rate"])
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Self-play factor schedule (identical to ppo_run.py)
# ---------------------------------------------------------------------------

def compute_self_play_factor(timestep: int, self_play_horizon: Optional[tuple]) -> float:
    if self_play_horizon is None:
        return 0.0
    thresh, timeline = self_play_horizon
    t = float(timestep)
    if thresh != 0 and timeline - (timeline / float(thresh)) * t > 1:
        return 1.0
    if timeline == thresh:
        return 0.0
    return max(-1.0 * (t - float(thresh)) / float(timeline - thresh) + 1.0, 0.0)


# ---------------------------------------------------------------------------
# Flatten helper (time-major -> flat batch)
# ---------------------------------------------------------------------------

def _flatten(x: jnp.ndarray) -> jnp.ndarray:
    """Merge leading [T, N] dims into [T*N, ...] staying on device."""
    t, n = x.shape[:2]
    return x.reshape((t * n,) + x.shape[2:])


def _update_delta_norms(params_before, params_after) -> dict:
    delta = jax.tree_util.tree_map(lambda new, old: new - old, params_after, params_before)
    p = delta["params"]
    conv = jnp.sqrt(sum(
        jnp.sum(jnp.square(p[f"Conv_{i}"]["kernel"])) +
        jnp.sum(jnp.square(p[f"Conv_{i}"]["bias"]))
        for i in range(3)
    ))
    dense = jnp.sqrt(sum(
        jnp.sum(jnp.square(p[f"Dense_{i}"]["kernel"])) +
        jnp.sum(jnp.square(p[f"Dense_{i}"]["bias"]))
        for i in range(3)
    ))
    policy_head = jnp.sqrt(
        jnp.sum(jnp.square(p["Dense_3"]["kernel"])) +
        jnp.sum(jnp.square(p["Dense_3"]["bias"]))
    )
    value_head = jnp.sqrt(
        jnp.sum(jnp.square(p["Dense_4"]["kernel"])) +
        jnp.sum(jnp.square(p["Dense_4"]["bias"]))
    )
    return {
        "update_delta_norm_global": float(optax.global_norm(delta)),
        "update_delta_norm_trunk": float(jnp.sqrt(conv * conv + dense * dense)),
        "update_delta_norm_policy_head": float(policy_head),
        "update_delta_norm_value_head": float(value_head),
    }


# ---------------------------------------------------------------------------
# Update-epoch: all minibatches, metrics stay on device until the end
# ---------------------------------------------------------------------------

def _run_update_epochs(
    train_state,
    obs_flat, actions_flat, logp_flat, values_flat, adv_flat, ret_flat,
    config: PPOConfig,
    rng: jax.Array,
):
    """Run config.num_epochs passes; return updated train_state and mean metrics."""
    batch_size = obs_flat.shape[0]
    minibatch_size = batch_size // config.num_minibatches
    all_metrics: List[Dict] = []

    for _ in range(config.num_epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, batch_size)
        for start in range(0, batch_size, minibatch_size):
            idx = perm[start: start + minibatch_size]
            train_state, metrics = ppo_update_step(
                train_state,
                obs=obs_flat[idx],
                actions=actions_flat[idx],
                old_logp=logp_flat[idx],
                old_values=values_flat[idx],
                advantages=adv_flat[idx],
                returns=ret_flat[idx],
                config=config,
            )
            # Keep metrics as JAX arrays — NO float() here.
            all_metrics.append(metrics)

    # Single device-to-host transfer: stack all metrics then mean.
    mean_metrics = {
        k: float(jnp.mean(jnp.stack([m[k] for m in all_metrics])))
        for k in all_metrics[0]
    }
    return train_state, mean_metrics, rng


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def ppo_run_jax(
    layout_name: str,
    seeds: List[int],
    config: PPOConfig,
    other_agent_type: str = "sp",
    self_play_horizon: Optional[tuple] = None,
    rew_shaping_horizon: Optional[int] = None,
    save_dir: str = "data/ppo_runs",
    ex_name: Optional[str] = None,
    lr_annealing: Optional[float] = None,
    best_bc_model_paths: Optional[dict] = None,
    diagnostics: bool = False,
) -> List[dict]:
    """
    JAX-optimised PPO training loop (self-play only).
    Falls back to ppo_run.py for BC-partner modes.
    """
    if other_agent_type != "sp":
        raise ValueError(
            f"ppo_run_jax only supports other_agent_type='sp'; "
            f"got '{other_agent_type}'. Use ppo_run() for BC-partner training."
        )

    t0 = time.time()

    if self_play_horizon is None:
        self_play_horizon = config.self_play_horizon
    if rew_shaping_horizon is None:
        rew_shaping_horizon = config.rew_shaping_horizon
    if lr_annealing is None:
        lr_annealing = config.lr_annealing

    print(f"[{_ts(t0)}] Parsing layout: {layout_name}")
    terrain = parse_layout(layout_name)
    print(f"[{_ts(t0)}] Layout parsed.")
    sys.stdout.flush()

    run_name = ex_name or f"ppo_sp_jax_{layout_name}"
    root_dir = Path(save_dir) / run_name
    root_dir.mkdir(parents=True, exist_ok=True)
    with (root_dir / "config.pkl").open("wb") as f:
        pickle.dump({"layout_name": layout_name, "config": asdict(config), "seeds": seeds}, f)

    summaries: List[dict] = []
    batch_size = int(config.num_envs * config.horizon)
    num_updates = int(config.total_timesteps // batch_size)
    minibatch_size = batch_size // config.num_minibatches

    print(f"Saving data to {root_dir}/")
    print(
        f"num_updates={num_updates}  batch_size={batch_size}  "
        f"minibatch_size={minibatch_size}  num_envs={config.num_envs}  "
        f"horizon={config.horizon}"
    )
    sys.stdout.flush()

    for seed in seeds:
        print(f"\n[{_ts(t0)}] === Seed {seed} ===")
        sys.stdout.flush()

        seed_dir = root_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "config.pkl").open("wb") as f:
            pickle.dump({"layout_name": layout_name, "config": asdict(config), "seed": seed}, f)

        rng = set_global_seed(int(seed))

        # ---- Infer obs shape via a probe env ---------------------------------
        print(f"[{_ts(t0)}] Probing obs shape...")
        sys.stdout.flush()
        rng, probe_rng = jax.random.split(rng)
        probe_bstate = make_batched_state(terrain, 1, probe_rng)
        probe_obs0, _ = encode_obs(terrain, probe_bstate)
        obs_shape = probe_obs0.shape[1:]   # strip batch dim
        print(f"[{_ts(t0)}] obs_shape={obs_shape}")
        sys.stdout.flush()

        # ---- Train state -----------------------------------------------------
        print(f"[{_ts(t0)}] Creating train state...")
        sys.stdout.flush()
        if lr_annealing != 1.0:
            lr_schedule = optax.linear_schedule(
                init_value=config.learning_rate,
                end_value=config.learning_rate * float(lr_annealing),
                transition_steps=max(1, num_updates * config.num_epochs * config.num_minibatches),
            )
            train_state = create_train_state(rng, obs_shape, config, learning_rate=lr_schedule)
        else:
            train_state = create_train_state(rng, obs_shape, config)
        print(f"[{_ts(t0)}] Train state created.")
        sys.stdout.flush()

        # ---- Batched env + initial obs ---------------------------------------
        print(f"[{_ts(t0)}] Creating BatchedEnvState ({config.num_envs} envs)...")
        sys.stdout.flush()
        rng, env_rng = jax.random.split(rng)
        bstate = make_batched_state(
            terrain, config.num_envs, env_rng,
            randomize_agent_idx=config.randomize_agent_idx,
        )
        obs0, _ = encode_obs(terrain, bstate)
        print(f"[{_ts(t0)}] BatchedEnvState ready.")
        sys.stdout.flush()

        # ---- Build JIT-compiled rollout function ----------------------------
        print(f"[{_ts(t0)}] Building rollout function (JIT compile on first call)...")
        sys.stdout.flush()
        rollout_fn = make_rollout_fn(
            terrain=terrain,
            horizon=config.horizon,
            num_envs=config.num_envs,
            randomize_agent_idx=config.randomize_agent_idx,
            bootstrap_with_zero_obs=config.bootstrap_with_zero_obs,
        )

        # ---- Logging buffers -------------------------------------------------
        logs: Dict[str, list] = {
            "eprewmean": [], "ep_sparse_rew_mean": [], "loss": [],
            "policy_loss": [], "value_loss": [], "policy_entropy": [],
            "approxkl": [], "clipfrac": [], "explained_variance": [],
        }
        if diagnostics:
            logs.update({
                "grad_norm_global": [],
                "grad_norm_conv": [],
                "grad_norm_dense": [],
                "grad_norm_policy_head": [],
                "grad_norm_value_head": [],
                "delta_norm_global": [],
                "delta_norm_conv": [],
                "delta_norm_dense": [],
                "delta_norm_policy_head": [],
                "delta_norm_value_head": [],
                "update_delta_norm_global": [],
                "update_delta_norm_trunk": [],
                "update_delta_norm_policy_head": [],
                "update_delta_norm_value_head": [],
                "adv_mean": [],
                "adv_std": [],
                "adv_min": [],
                "adv_max": [],
                "adv_norm_mean": [],
                "adv_norm_std": [],
                "adv_norm_min": [],
                "adv_norm_max": [],
                "loss_component_actor": [],
                "loss_component_value_scaled": [],
                "loss_component_entropy_scaled": [],
            })
        best_sparse = float("-inf")
        total_steps = 0
        t_start = time.time()
        eprew_buffer = deque(maxlen=100)
        ep_sparse_buffer = deque(maxlen=100)

        # Match TF baseline: first rollout uses sf=1.0; annealing is applied
        # AFTER each update (using total_steps at the end of the update).
        shaping_factor = 1.0
        sp_factor = compute_self_play_factor(0, self_play_horizon)

        for update in range(num_updates):
            total_steps += batch_size

            rng, rollout_rng = jax.random.split(rng)

            if update == 0:
                print(f"[{_ts(t0)}] First rollout — XLA compile expected now...")
                sys.stdout.flush()
            t_rollout = time.time()

            rollout, bstate, obs0 = rollout_fn(
                train_state, bstate, obs0, shaping_factor, sp_factor, rollout_rng
            )

            if update == 0:
                elapsed_r = time.time() - t_rollout
                sps = int(config.horizon * config.num_envs / max(elapsed_r, 1e-6))
                print(
                    f"[{_ts(t0)}] First rollout done in {elapsed_r:.1f}s  "
                    f"({sps} steps/s)"
                )
                sys.stdout.flush()

            # ---- GAE ---------------------------------------------------------
            adv, ret = compute_gae(
                jnp.asarray(rollout.rewards, dtype=jnp.float32),
                jnp.asarray(rollout.values,  dtype=jnp.float32),
                jnp.asarray(rollout.dones,   dtype=jnp.float32),
                float(config.gamma),
                float(config.gae_lambda),
                bootstrap_value=jnp.asarray(rollout.next_value, dtype=jnp.float32),
            )

            obs_flat     = _flatten(jnp.asarray(rollout.obs,       dtype=jnp.float32))
            actions_flat = _flatten(jnp.asarray(rollout.actions,   dtype=jnp.int32))
            logp_flat    = _flatten(jnp.asarray(rollout.log_probs, dtype=jnp.float32))
            values_flat  = _flatten(jnp.asarray(rollout.values,    dtype=jnp.float32))
            adv_flat     = _flatten(jnp.asarray(adv))
            ret_flat     = _flatten(jnp.asarray(ret))

            params_before_update = train_state.params
            rng, update_rng = jax.random.split(rng)
            train_state, mean_metrics, rng = _run_update_epochs(
                train_state,
                obs_flat, actions_flat, logp_flat, values_flat, adv_flat, ret_flat,
                config, update_rng,
            )
            mean_metrics.update(_update_delta_norms(params_before_update, train_state.params))

            # ---- Logging -----------------------------------------------------
            t_now   = time.time()
            elapsed = t_now - t_start
            fps     = int(total_steps / max(elapsed, 1e-6))
            remaining = elapsed * (num_updates - update - 1) / max(update + 1, 1)
            current_lr = _current_lr(train_state, float(config.learning_rate))
            ev = _explained_variance(np.asarray(values_flat), np.asarray(ret_flat))

            completed_eprew = np.asarray(
                rollout.infos.get("completed_eprew", []), dtype=np.float32
            ).reshape(-1)
            completed_ep_sparse = np.asarray(
                rollout.infos.get("completed_ep_sparse_rew", []), dtype=np.float32
            ).reshape(-1)
            for r in completed_eprew:
                eprew_buffer.append(float(r))
            for r in completed_ep_sparse:
                ep_sparse_buffer.append(float(r))

            # Match TF/Baselines: rolling mean over most recent 100 finished episodes.
            eprewmean = float(np.mean(eprew_buffer)) if eprew_buffer else float(rollout.infos["eprewmean"])
            ep_sparse = float(np.mean(ep_sparse_buffer)) if ep_sparse_buffer else float(rollout.infos["ep_sparse_rew_mean"])

            for k, v in [
                ("eprewmean", eprewmean), ("ep_sparse_rew_mean", ep_sparse),
                ("loss",      mean_metrics.get("loss", 0.0)),
                ("policy_loss",    mean_metrics.get("policy_loss", 0.0)),
                ("value_loss",     mean_metrics.get("value_loss",  0.0)),
                ("policy_entropy", mean_metrics.get("entropy",     0.0)),
                ("approxkl",       mean_metrics.get("approxkl",    0.0)),
                ("clipfrac",       mean_metrics.get("clipfrac",    0.0)),
                ("explained_variance", ev),
            ]:
                logs[k].append(v)
            if diagnostics:
                for k in (
                    "grad_norm_global",
                    "grad_norm_conv",
                    "grad_norm_dense",
                    "grad_norm_policy_head",
                    "grad_norm_value_head",
                    "delta_norm_global",
                    "delta_norm_conv",
                    "delta_norm_dense",
                    "delta_norm_policy_head",
                    "delta_norm_value_head",
                    "update_delta_norm_global",
                    "update_delta_norm_trunk",
                    "update_delta_norm_policy_head",
                    "update_delta_norm_value_head",
                    "adv_mean",
                    "adv_std",
                    "adv_min",
                    "adv_max",
                    "adv_norm_mean",
                    "adv_norm_std",
                    "adv_norm_min",
                    "adv_norm_max",
                    "loss_component_actor",
                    "loss_component_value_scaled",
                    "loss_component_entropy_scaled",
                ):
                    logs[k].append(float(mean_metrics.get(k, 0.0)))

            rew_per_step = eprewmean / max(config.horizon, 1)
            print(f"Curr learning rate {current_lr} \t Curr reward per step {rew_per_step:.6g}")
            _log_table({
                "approxkl":           mean_metrics.get("approxkl",    0.0),
                "clipfrac":           mean_metrics.get("clipfrac",    0.0),
                "eplenmean":          config.horizon,
                "eprewmean":          eprewmean,
                "explained_variance": ev,
                "fps":                fps,
                "nupdates":           update + 1,
                "policy_entropy":     mean_metrics.get("entropy",     0.0),
                "policy_loss":        mean_metrics.get("policy_loss", 0.0),
                "serial_timesteps":   total_steps // config.num_envs,
                "time_elapsed":       round(elapsed, 1),
                "time_remaining":     round(remaining, 1),
                "total_timesteps":    total_steps,
                "true_eprew":         ep_sparse,
                "value_loss":         mean_metrics.get("value_loss",  0.0),
            })
            if diagnostics:
                _log_table({
                    "grad_norm_global": mean_metrics.get("grad_norm_global", 0.0),
                    "grad_norm_conv": mean_metrics.get("grad_norm_conv", 0.0),
                    "grad_norm_dense": mean_metrics.get("grad_norm_dense", 0.0),
                    "grad_norm_policy_head": mean_metrics.get("grad_norm_policy_head", 0.0),
                    "grad_norm_value_head": mean_metrics.get("grad_norm_value_head", 0.0),
                    "delta_norm_global": mean_metrics.get("delta_norm_global", 0.0),
                    "delta_norm_conv": mean_metrics.get("delta_norm_conv", 0.0),
                    "delta_norm_dense": mean_metrics.get("delta_norm_dense", 0.0),
                    "delta_norm_policy_head": mean_metrics.get("delta_norm_policy_head", 0.0),
                    "delta_norm_value_head": mean_metrics.get("delta_norm_value_head", 0.0),
                    "update_delta_norm_global": mean_metrics.get("update_delta_norm_global", 0.0),
                    "update_delta_norm_trunk": mean_metrics.get("update_delta_norm_trunk", 0.0),
                    "update_delta_norm_policy_head": mean_metrics.get("update_delta_norm_policy_head", 0.0),
                    "update_delta_norm_value_head": mean_metrics.get("update_delta_norm_value_head", 0.0),
                    "adv_mean": mean_metrics.get("adv_mean", 0.0),
                    "adv_std": mean_metrics.get("adv_std", 0.0),
                    "adv_min": mean_metrics.get("adv_min", 0.0),
                    "adv_max": mean_metrics.get("adv_max", 0.0),
                    "adv_norm_mean": mean_metrics.get("adv_norm_mean", 0.0),
                    "adv_norm_std": mean_metrics.get("adv_norm_std", 0.0),
                    "adv_norm_min": mean_metrics.get("adv_norm_min", 0.0),
                    "adv_norm_max": mean_metrics.get("adv_norm_max", 0.0),
                    "loss_component_actor": mean_metrics.get("loss_component_actor", 0.0),
                    "loss_component_value_scaled": mean_metrics.get("loss_component_value_scaled", 0.0),
                    "loss_component_entropy_scaled": mean_metrics.get("loss_component_entropy_scaled", 0.0),
                })
            # Anneal shaping AFTER logging (matches TF baseline order:
            # rollout → PPO update → log → update shaping for next rollout).
            shaping_factor = float(
                annealed_shaping_factor(1.0, float(rew_shaping_horizon), jnp.asarray(total_steps))
            )
            sp_factor = compute_self_play_factor(total_steps, self_play_horizon)
            print(f"Current reward shaping {shaping_factor:.4f}")

            if ep_sparse > best_sparse:
                best_sparse = ep_sparse
                save_ppo_checkpoint(train_state.params, seed_dir / "best")

        save_ppo_checkpoint(train_state.params, seed_dir / "ppo_agent")
        save_training_info(logs, seed_dir / "training_info.pkl")
        summaries.append({
            "seed":                     seed,
            "best_sparse_rew_mean":     best_sparse,
            "final_eprewmean":          logs["eprewmean"][-1] if logs["eprewmean"] else 0.0,
            "final_ep_sparse_rew_mean": logs["ep_sparse_rew_mean"][-1] if logs["ep_sparse_rew_mean"] else 0.0,
            "seed_dir":                 str(seed_dir),
        })

    return summaries
