"""Full PPO training loop with legacy-faithful run structure."""

import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.agents.ppo.train import compute_gae, create_train_state, ppo_update_step
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.reward_shaping import DEFAULT_REW_SHAPING_PARAMS, annealed_shaping_factor
from human_aware_rl_jax_lift.reproducibility.seed import set_global_seed

from .checkpoints import save_ppo_checkpoint, save_training_info
from .partners import BCPartner, MixedPartner, SelfPlayPartner
from .runner import RolloutRunner
from .vec_env import VectorizedEnv

# ---------------------------------------------------------------------------
# Logging helpers  (mirrors OpenAI Baselines logger.dumpkvs table format)
# ---------------------------------------------------------------------------

def _log_table(kvs: Dict[str, object]) -> None:
    """Print a Baselines-style key/value table to stdout and flush immediately."""
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
    """Elapsed seconds since t0, formatted."""
    return f"+{time.time() - t0:.1f}s"


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return float("nan") if var_y < 1e-8 else float(1.0 - np.var(y_true - y_pred) / var_y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_self_play_factor(timestep: int, self_play_horizon: tuple[int, int] | None) -> float:
    """Piecewise linear schedule matching legacy ppo2 behavior."""
    if self_play_horizon is None:
        return 0.0
    thresh, timeline = self_play_horizon
    t = float(timestep)
    if thresh != 0 and timeline - (timeline / float(thresh)) * t > 1:
        return 1.0
    if timeline == thresh:
        return 0.0
    val = -1.0 * (t - float(thresh)) / (float(timeline - thresh)) + 1.0
    return max(val, 0.0)


def _flatten_rollout(x: np.ndarray) -> jnp.ndarray:
    t, n = x.shape[:2]
    return jnp.asarray(x.reshape((t * n,) + x.shape[2:]))


def _scaled_reward_params(shaping_factor: float) -> Dict[str, float]:
    out = {}
    for k, v in DEFAULT_REW_SHAPING_PARAMS.items():
        out[k] = float(v * shaping_factor) if k.endswith("_REW") or k.endswith("_REWARD") else float(v)
    return out


def _current_lr(train_state, fallback: float) -> float:
    """Extract current learning rate from optax state, with fallback."""
    try:
        return float(train_state.opt_state.hyperparams["learning_rate"])
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def ppo_run(
    layout_name: str,
    seeds: List[int],
    config: PPOConfig,
    other_agent_type: str = "sp",
    self_play_horizon: tuple[int, int] | None = None,
    rew_shaping_horizon: int | None = None,
    save_dir: str = "data/ppo_runs",
    ex_name: str | None = None,
    lr_annealing: float | None = None,
    best_bc_model_paths: dict | None = None,
) -> List[dict]:
    """
    Run PPO training over one or more seeds.
    Prints Baselines-style per-update tables and timing milestones.
    """
    t0 = time.time()

    if self_play_horizon is None:
        self_play_horizon = config.self_play_horizon
    if rew_shaping_horizon is None:
        rew_shaping_horizon = config.rew_shaping_horizon
    if lr_annealing is None:
        lr_annealing = config.lr_annealing

    print(f"[{_ts(t0)}] Parsing layout: {layout_name}")
    sys.stdout.flush()
    terrain = parse_layout(layout_name)
    print(f"[{_ts(t0)}] Layout parsed.")
    sys.stdout.flush()

    run_name = ex_name or f"ppo_{other_agent_type}_{layout_name}"
    root_dir = Path(save_dir) / run_name
    root_dir.mkdir(parents=True, exist_ok=True)
    with (root_dir / "config.pkl").open("wb") as f:
        pickle.dump({"layout_name": layout_name, "config": asdict(config), "seeds": seeds}, f)

    summaries: List[dict] = []
    batch_size = int(config.num_envs * config.horizon)
    num_updates = int(config.total_timesteps // batch_size)
    minibatch_size = int(batch_size // config.num_minibatches)

    print(f"Saving data to {root_dir}/")
    grad_updates_per_agent = config.num_epochs * config.num_minibatches * num_updates
    print(f"Grad updates per agent {grad_updates_per_agent:.1f}")
    print(f"num_updates={num_updates}  batch_size={batch_size}  minibatch_size={minibatch_size}  num_envs={config.num_envs}  horizon={config.horizon}")
    sys.stdout.flush()

    for seed in seeds:
        print(f"\n[{_ts(t0)}] === Seed {seed} ===")
        sys.stdout.flush()

        seed_dir = root_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "config.pkl").open("wb") as f:
            pickle.dump({"layout_name": layout_name, "config": asdict(config), "seed": seed}, f)

        rng = set_global_seed(int(seed))

        print(f"[{_ts(t0)}] Creating probe VectorizedEnv (1 env)...")
        sys.stdout.flush()
        vec_probe = VectorizedEnv(terrain=terrain, num_envs=1, horizon=config.horizon)
        _, probe_obs0, _, _ = vec_probe.reset_all()
        obs_shape = probe_obs0.shape[1:]
        print(f"[{_ts(t0)}] obs_shape={obs_shape}")
        sys.stdout.flush()

        print(f"[{_ts(t0)}] Creating train state...")
        sys.stdout.flush()
        if lr_annealing != 1.0:
            # TF Baselines: anneal from lr to lr / lr_annealing (reduction factor)
            lr_schedule = optax.linear_schedule(
                init_value=config.learning_rate,
                end_value=config.learning_rate / float(lr_annealing),
                transition_steps=max(1, num_updates * config.num_epochs * config.num_minibatches),
            )
            train_state = create_train_state(rng, obs_shape, config, learning_rate=lr_schedule)
        else:
            train_state = create_train_state(rng, obs_shape, config)
        print(f"[{_ts(t0)}] Train state created.")
        sys.stdout.flush()

        print(f"[{_ts(t0)}] Creating VectorizedEnv ({config.num_envs} envs)...")
        sys.stdout.flush()
        vec_env = VectorizedEnv(
            terrain=terrain,
            num_envs=config.num_envs,
            horizon=config.horizon,
            reward_shaping_params=_scaled_reward_params(1.0),
        )

        if other_agent_type == "sp":
            partner = SelfPlayPartner(stochastic=True)
        else:
            if best_bc_model_paths is None:
                raise ValueError("best_bc_model_paths is required for bc partner training")
            split = "train" if other_agent_type == "bc_train" else "test"
            model_path = Path(best_bc_model_paths[split][layout_name])
            if model_path.is_dir():
                model_path = model_path / "model.pkl"
            bc_partner = BCPartner.from_path(model_path, terrain=terrain, stochastic=True)
            partner = MixedPartner(
                bc_partner=bc_partner,
                trajectory_sp=bool(config.trajectory_self_play),
                stochastic=True,
            )

        print(f"[{_ts(t0)}] Creating RolloutRunner (triggers reset_all)...")
        sys.stdout.flush()
        runner = RolloutRunner(
            vec_env=vec_env,
            train_state=train_state,
            other_agent=partner,
            horizon=config.horizon,
            trajectory_self_play=bool(config.trajectory_self_play),
        )
        print(f"[{_ts(t0)}] RolloutRunner ready.")
        sys.stdout.flush()

        # Warmup: trigger policy + partner JIT compiles so first rollout step doesn't block
        print(f"[{_ts(t0)}] Warmup: compiling policy...")
        sys.stdout.flush()
        _a, _v, _l = runner._policy_step(runner.obs0, rng)
        jax.block_until_ready(_a)
        print(f"[{_ts(t0)}] Warmup: policy compiled.")
        sys.stdout.flush()
        rng, rng_partner = jax.random.split(rng)
        print(f"[{_ts(t0)}] Warmup: compiling partner (one act call)...")
        sys.stdout.flush()
        _oa = runner.other_agent.act(
            runner.obs1,
            rng_partner,
            train_state=runner.train_state,
            self_play_randomization=0.0,
            trajectory_sp=runner.trajectory_self_play,
            states=runner.vec_env.states,
            agent_idx=runner.vec_env.agent_idx,
        )
        rng, _ = jax.random.split(rng)
        print(f"[{_ts(t0)}] Warmup: one env step (compile step_all)...")
        sys.stdout.flush()
        _step = runner.vec_env.step_all(_a, _oa)
        # Reset so first real rollout starts from clean state
        _, runner.obs0, runner.obs1, runner.agent_idx = runner.vec_env.reset_all()
        print(f"[{_ts(t0)}] Warmup done.")
        sys.stdout.flush()

        logs: Dict[str, list] = {
            "eprewmean":          [],
            "ep_sparse_rew_mean": [],
            "loss":               [],
            "policy_loss":        [],
            "value_loss":         [],
            "policy_entropy":     [],
            "approxkl":           [],
            "clipfrac":           [],
            "explained_variance": [],
        }
        best_sparse = float("-inf")
        total_steps = 0
        t_start = time.time()

        for update in range(num_updates):
            total_steps += batch_size
            shaping_factor = float(
                annealed_shaping_factor(1.0, float(rew_shaping_horizon), jnp.asarray(total_steps))
            )
            vec_env.reward_shaping_params = _scaled_reward_params(shaping_factor)
            sp_factor = compute_self_play_factor(total_steps, self_play_horizon)

            rng, rollout_rng = jax.random.split(rng)
            runner.train_state = train_state

            if update == 0:
                print(f"[{_ts(t0)}] Starting first rollout (JIT compile expected here)...")
                sys.stdout.flush()
            t_rollout = time.time()
            rollout = runner.collect_rollout(rollout_rng, self_play_randomization=sp_factor)
            if update == 0:
                print(f"[{_ts(t0)}] First rollout done in {time.time()-t_rollout:.1f}s")
                sys.stdout.flush()

            adv, ret = compute_gae(
                jnp.asarray(rollout.rewards, dtype=jnp.float32),
                jnp.asarray(rollout.values, dtype=jnp.float32),
                jnp.asarray(rollout.dones, dtype=jnp.float32),
                float(config.gamma),
                float(config.gae_lambda),
                bootstrap_value=jnp.asarray(rollout.next_value, dtype=jnp.float32),
            )
            obs_flat     = _flatten_rollout(rollout.obs)
            actions_flat = _flatten_rollout(rollout.actions)
            logp_flat    = _flatten_rollout(rollout.log_probs)
            values_flat  = _flatten_rollout(rollout.values)
            adv_flat     = _flatten_rollout(np.asarray(adv))
            ret_flat     = _flatten_rollout(np.asarray(ret))

            mb_loss, mb_pl, mb_vl, mb_ent, mb_kl, mb_cf = [], [], [], [], [], []

            for _ in range(config.num_epochs):
                perm = np.random.permutation(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    idx = perm[start : start + minibatch_size]
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
                    mb_loss.append(float(metrics.get("loss",        0.0)))
                    mb_pl.append(  float(metrics.get("policy_loss", 0.0)))
                    mb_vl.append(  float(metrics.get("value_loss",  0.0)))
                    mb_ent.append( float(metrics.get("entropy",     0.0)))
                    mb_kl.append(  float(metrics.get("approxkl",    0.0)))
                    mb_cf.append(  float(metrics.get("clipfrac",    0.0)))

            mean_loss = float(np.mean(mb_loss)) if mb_loss else 0.0
            mean_pl   = float(np.mean(mb_pl))
            mean_vl   = float(np.mean(mb_vl))
            mean_ent  = float(np.mean(mb_ent))
            mean_kl   = float(np.mean(mb_kl))
            mean_cf   = float(np.mean(mb_cf))
            ev        = _explained_variance(np.asarray(values_flat), np.asarray(ret_flat))

            eprewmean = float(rollout.infos["eprewmean"])
            ep_sparse = float(rollout.infos["ep_sparse_rew_mean"])

            t_now     = time.time()
            elapsed   = t_now - t_start
            fps       = int(total_steps / max(elapsed, 1e-6))
            remaining = elapsed * (num_updates - update - 1) / max(update + 1, 1)
            current_lr = _current_lr(train_state, float(config.learning_rate))

            logs["loss"].append(mean_loss)
            logs["eprewmean"].append(eprewmean)
            logs["ep_sparse_rew_mean"].append(ep_sparse)
            logs["policy_loss"].append(mean_pl)
            logs["value_loss"].append(mean_vl)
            logs["policy_entropy"].append(mean_ent)
            logs["approxkl"].append(mean_kl)
            logs["clipfrac"].append(mean_cf)
            logs["explained_variance"].append(ev)

            rew_per_step = eprewmean / max(config.horizon, 1)
            print(
                f"Curr learning rate {current_lr} \t "
                f"Curr reward per step {rew_per_step:.6g}"
            )
            _log_table({
                "approxkl":           mean_kl,
                "clipfrac":           mean_cf,
                "eplenmean":          config.horizon,
                "eprewmean":          eprewmean,
                "explained_variance": ev,
                "fps":                fps,
                "nupdates":           update + 1,
                "policy_entropy":     mean_ent,
                "policy_loss":        mean_pl,
                "serial_timesteps":   total_steps // config.num_envs,
                "time_elapsed":       round(elapsed, 1),
                "time_remaining":     round(remaining, 1),
                "total_timesteps":    total_steps,
                "true_eprew":         ep_sparse,
                "value_loss":         mean_vl,
            })
            print(f"Current reward shaping {shaping_factor:.4f}")

            if ep_sparse > best_sparse:
                best_sparse = ep_sparse
                save_ppo_checkpoint(train_state.params, seed_dir / "best")

            runner.train_state = train_state

        save_ppo_checkpoint(train_state.params, seed_dir / "ppo_agent")
        save_training_info(logs, seed_dir / "training_info.pkl")
        summaries.append(
            {
                "seed":                       seed,
                "best_sparse_rew_mean":       best_sparse,
                "final_eprewmean":            logs["eprewmean"][-1] if logs["eprewmean"] else 0.0,
                "final_ep_sparse_rew_mean":   logs["ep_sparse_rew_mean"][-1] if logs["ep_sparse_rew_mean"] else 0.0,
                "seed_dir":                   str(seed_dir),
            }
        )
    return summaries
