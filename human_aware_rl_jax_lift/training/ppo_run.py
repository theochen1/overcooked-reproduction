"""Full PPO training loop with legacy-faithful run structure."""

import pickle
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

    Returns per-seed summary dictionaries.
    """
    if self_play_horizon is None:
        self_play_horizon = config.self_play_horizon
    if rew_shaping_horizon is None:
        rew_shaping_horizon = config.rew_shaping_horizon
    if lr_annealing is None:
        lr_annealing = config.lr_annealing

    terrain = parse_layout(layout_name)
    run_name = ex_name or f"ppo_{other_agent_type}_{layout_name}"
    root_dir = Path(save_dir) / run_name
    root_dir.mkdir(parents=True, exist_ok=True)
    with (root_dir / "config.pkl").open("wb") as f:
        pickle.dump({"layout_name": layout_name, "config": asdict(config), "seeds": seeds}, f)

    summaries: List[dict] = []
    batch_size = int(config.num_envs * config.horizon)
    num_updates = int(config.total_timesteps // batch_size)
    minibatch_size = int(batch_size // config.num_minibatches)

    for seed in seeds:
        seed_dir = root_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "config.pkl").open("wb") as f:
            pickle.dump({"layout_name": layout_name, "config": asdict(config), "seed": seed}, f)

        rng = set_global_seed(int(seed))
        # Use one encoded observation to infer shape robustly.
        vec_probe = VectorizedEnv(terrain=terrain, num_envs=1, horizon=config.horizon)
        _, probe_obs0, _, _ = vec_probe.reset_all()
        obs_shape = probe_obs0.shape[1:]

        if lr_annealing != 1.0:
            lr_schedule = optax.linear_schedule(
                init_value=config.learning_rate,
                end_value=config.learning_rate * float(lr_annealing),
                transition_steps=max(1, num_updates * config.num_epochs * config.num_minibatches),
            )
            train_state = create_train_state(rng, obs_shape, config, learning_rate=lr_schedule)
        else:
            train_state = create_train_state(rng, obs_shape, config)

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

        runner = RolloutRunner(
            vec_env=vec_env,
            train_state=train_state,
            other_agent=partner,
            horizon=config.horizon,
            trajectory_self_play=bool(config.trajectory_self_play),
        )

        logs = {"eprewmean": [], "ep_sparse_rew_mean": [], "loss": []}
        best_sparse = float("-inf")
        total_steps = 0

        for update in range(num_updates):
            total_steps += batch_size
            shaping_factor = float(annealed_shaping_factor(1.0, float(rew_shaping_horizon), jnp.asarray(total_steps)))
            vec_env.reward_shaping_params = _scaled_reward_params(shaping_factor)
            sp_factor = compute_self_play_factor(total_steps, self_play_horizon)

            rng, rollout_rng = jax.random.split(rng)
            runner.train_state = train_state
            rollout = runner.collect_rollout(rollout_rng, self_play_randomization=sp_factor)

            adv, ret = compute_gae(
                jnp.asarray(rollout.rewards, dtype=jnp.float32),
                jnp.asarray(rollout.values, dtype=jnp.float32),
                jnp.asarray(rollout.dones, dtype=jnp.float32),
                float(config.gamma),
                float(config.gae_lambda),
                bootstrap_value=jnp.asarray(rollout.next_value, dtype=jnp.float32),
            )
            obs_flat = _flatten_rollout(rollout.obs)
            actions_flat = _flatten_rollout(rollout.actions)
            logp_flat = _flatten_rollout(rollout.log_probs)
            values_flat = _flatten_rollout(rollout.values)
            adv_flat = _flatten_rollout(np.asarray(adv))
            ret_flat = _flatten_rollout(np.asarray(ret))

            epoch_losses = []
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
                    epoch_losses.append(float(metrics["loss"]))

            logs["loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
            logs["eprewmean"].append(float(rollout.infos["eprewmean"]))
            logs["ep_sparse_rew_mean"].append(float(rollout.infos["ep_sparse_rew_mean"]))

            if rollout.infos["ep_sparse_rew_mean"] > best_sparse:
                best_sparse = float(rollout.infos["ep_sparse_rew_mean"])
                save_ppo_checkpoint(train_state.params, seed_dir / "best")

            runner.train_state = train_state

        save_ppo_checkpoint(train_state.params, seed_dir / "ppo_agent")
        save_training_info(logs, seed_dir / "training_info.pkl")
        summaries.append(
            {
                "seed": seed,
                "best_sparse_rew_mean": best_sparse,
                "final_eprewmean": logs["eprewmean"][-1] if logs["eprewmean"] else 0.0,
                "final_ep_sparse_rew_mean": logs["ep_sparse_rew_mean"][-1] if logs["ep_sparse_rew_mean"] else 0.0,
                "seed_dir": str(seed_dir),
            }
        )
    return summaries
