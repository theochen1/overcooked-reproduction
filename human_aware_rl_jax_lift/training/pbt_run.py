"""Population-based training loop built on PPO runner primitives."""

from dataclasses import dataclass, replace
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from human_aware_rl_jax_lift.agents.pbt.config import PBTConfig
from human_aware_rl_jax_lift.agents.pbt.trainer import PBTTrainer
from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.agents.ppo.train import compute_gae, create_train_state, ppo_update_step
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.reward_shaping import DEFAULT_REW_SHAPING_PARAMS, annealed_shaping_factor
from human_aware_rl_jax_lift.reproducibility.seed import set_global_seed
from human_aware_rl_jax_lift.training.partners import BCPartner, MixedPartner, SelfPlayPartner
from human_aware_rl_jax_lift.training.runner import RolloutRunner
from human_aware_rl_jax_lift.training.vec_env import VectorizedEnv

from .checkpoints import save_ppo_checkpoint, save_training_info
from .ppo_run import compute_self_play_factor


def _scaled_reward_params(shaping_factor: float) -> Dict[str, float]:
    out = {}
    for k, v in DEFAULT_REW_SHAPING_PARAMS.items():
        out[k] = float(v * shaping_factor) if k.endswith("_REW") or k.endswith("_REWARD") else float(v)
    return out


def _flatten_rollout(x: np.ndarray) -> jnp.ndarray:
    t, n = x.shape[:2]
    return jnp.asarray(x.reshape((t * n,) + x.shape[2:]))


def _member_params_from_cfg(cfg: PPOConfig) -> Dict[str, float]:
    return {
        "LAM": float(cfg.gae_lambda),
        "CLIPPING": float(cfg.clip_eps),
        "LR": float(cfg.learning_rate),
        "STEPS_PER_UPDATE": int(cfg.num_epochs),
        "ENTROPY": float(cfg.ent_coef),
        "VF_COEF": float(cfg.vf_coef),
    }


def _cfg_from_member_params(base_cfg: PPOConfig, params: Dict[str, float]) -> PPOConfig:
    return replace(
        base_cfg,
        gae_lambda=float(params.get("LAM", base_cfg.gae_lambda)),
        clip_eps=float(params.get("CLIPPING", base_cfg.clip_eps)),
        learning_rate=float(params.get("LR", base_cfg.learning_rate)),
        num_epochs=int(params.get("STEPS_PER_UPDATE", base_cfg.num_epochs)),
        ent_coef=float(params.get("ENTROPY", base_cfg.ent_coef)),
        vf_coef=float(params.get("VF_COEF", base_cfg.vf_coef)),
    )


def _rebuild_state_with_lr(state: TrainState, learning_rate: float, max_grad_norm: float) -> TrainState:
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    # Preserve optimizer accumulators and params while swapping schedule hyperparameters.
    return state.replace(tx=tx)


@dataclass
class _MemberRuntime:
    cfg: PPOConfig
    train_state: TrainState
    vec_env: VectorizedEnv
    runner: RolloutRunner
    total_steps: int
    logs: Dict[str, list]
    seed: int


def _evaluate_member(member: _MemberRuntime, num_selection_games: int) -> float:
    """
    Evaluate by rolling out until at least `num_selection_games` episodes are observed.
    Returns mean sparse episode reward.
    """
    rng = set_global_seed(member.seed + 999_999 + member.total_steps)
    sparse_means = []
    episodes = 0
    while episodes < num_selection_games:
        rng, eval_rng = jax.random.split(rng)
        rollout = member.runner.collect_rollout(eval_rng, self_play_randomization=0.0)
        eps = int(rollout.infos.get("episodes_this_rollout", 0))
        if eps > 0:
            sparse_means.append(float(rollout.infos["ep_sparse_rew_mean"]))
            episodes += eps
        else:
            # Avoid infinite loops if no episodes terminate in one rollout.
            sparse_means.append(float(rollout.infos["ep_sparse_rew_mean"]))
            episodes += 1
    return float(np.mean(sparse_means))


def pbt_run(
    layout_name: str,
    seeds: List[int],
    ppo_config: PPOConfig,
    pbt_config: PBTConfig,
    total_steps_per_agent: int,
    other_agent_type: str = "sp",
    best_bc_model_paths: dict | None = None,
    save_dir: str = "data/pbt_runs",
    ex_name: str | None = None,
) -> dict:
    """Run population-based training and return aggregate metrics."""
    if len(seeds) < pbt_config.population_size:
        raise ValueError("Need at least one seed per population member")
    terrain = parse_layout(layout_name)
    # Paper Appendix D: PBT uses 50 parallel environments.
    ppo_config = replace(ppo_config, num_envs=50)
    run_name = ex_name or f"pbt_{layout_name}"
    batch_size = int(ppo_config.num_envs * ppo_config.horizon)
    updates_total = int(total_steps_per_agent // batch_size)

    trainer = PBTTrainer(base_hparams=_member_params_from_cfg(ppo_config), config=pbt_config)
    members: List[_MemberRuntime] = []
    for m_idx in range(pbt_config.population_size):
        seed = int(seeds[m_idx])
        rng = set_global_seed(seed)
        probe_env = VectorizedEnv(terrain=terrain, num_envs=1, horizon=ppo_config.horizon)
        _, probe_obs0, _, _ = probe_env.reset_all()
        cfg = _cfg_from_member_params(ppo_config, trainer.population[m_idx].params)
        state = create_train_state(rng, probe_obs0.shape[1:], cfg)
        vec_env = VectorizedEnv(
            terrain=terrain,
            num_envs=cfg.num_envs,
            horizon=cfg.horizon,
            reward_shaping_params=_scaled_reward_params(1.0),
        )
        if other_agent_type == "sp":
            partner = SelfPlayPartner(stochastic=True)
        else:
            if best_bc_model_paths is None:
                raise ValueError("best_bc_model_paths is required for BC-partner PBT runs")
            split = "train" if other_agent_type == "bc_train" else "test"
            bc_partner = BCPartner.from_path(best_bc_model_paths[split][layout_name], terrain=terrain, stochastic=True)
            partner = MixedPartner(bc_partner=bc_partner, trajectory_sp=cfg.trajectory_self_play, stochastic=True)
        runner = RolloutRunner(vec_env=vec_env, train_state=state, other_agent=partner, horizon=cfg.horizon)
        members.append(
            _MemberRuntime(
                cfg=cfg,
                train_state=state,
                vec_env=vec_env,
                runner=runner,
                total_steps=0,
                logs={"ep_sparse_rew_mean": [], "eprewmean": [], "loss": []},
                seed=seed,
            )
        )

    selections = int(np.ceil(updates_total / pbt_config.iter_per_selection))
    updates_done = 0
    selection_history = []
    for _sel in range(selections):
        updates_this_selection = min(pbt_config.iter_per_selection, updates_total - updates_done)
        if updates_this_selection <= 0:
            break
        for member in members:
            rng = set_global_seed(member.seed + updates_done)
            for _ in range(updates_this_selection):
                member.total_steps += batch_size
                shaping = float(
                    annealed_shaping_factor(1.0, float(member.cfg.rew_shaping_horizon), jnp.asarray(member.total_steps))
                )
                member.vec_env.reward_shaping_params = _scaled_reward_params(shaping)
                sp_factor = compute_self_play_factor(member.total_steps, member.cfg.self_play_horizon)
                rng, rollout_rng = jax.random.split(rng)
                member.runner.train_state = member.train_state
                rollout = member.runner.collect_rollout(rollout_rng, self_play_randomization=sp_factor)

                adv, ret = compute_gae(
                    jnp.asarray(rollout.rewards, dtype=jnp.float32),
                    jnp.asarray(rollout.values, dtype=jnp.float32),
                    jnp.asarray(rollout.dones, dtype=jnp.float32),
                    float(member.cfg.gamma),
                    float(member.cfg.gae_lambda),
                    bootstrap_value=jnp.asarray(rollout.next_value, dtype=jnp.float32),
                )
                obs_flat = _flatten_rollout(rollout.obs)
                actions_flat = _flatten_rollout(rollout.actions)
                logp_flat = _flatten_rollout(rollout.log_probs)
                values_flat = _flatten_rollout(rollout.values)
                adv_flat = _flatten_rollout(np.asarray(adv))
                ret_flat = _flatten_rollout(np.asarray(ret))
                minibatch_size = (member.cfg.num_envs * member.cfg.horizon) // member.cfg.num_minibatches
                epoch_losses = []
                for _ in range(member.cfg.num_epochs):
                    perm = np.random.permutation(member.cfg.num_envs * member.cfg.horizon)
                    for start in range(0, len(perm), minibatch_size):
                        idx = perm[start : start + minibatch_size]
                        member.train_state, metrics = ppo_update_step(
                            member.train_state,
                            obs=obs_flat[idx],
                            actions=actions_flat[idx],
                            old_logp=logp_flat[idx],
                            old_values=values_flat[idx],
                            advantages=adv_flat[idx],
                            returns=ret_flat[idx],
                            config=member.cfg,
                        )
                        epoch_losses.append(float(metrics["loss"]))
                member.runner.train_state = member.train_state
                member.logs["loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
                member.logs["eprewmean"].append(float(rollout.infos["eprewmean"]))
                member.logs["ep_sparse_rew_mean"].append(float(rollout.infos["ep_sparse_rew_mean"]))

        # Selection fitness from explicit evaluation episodes (Appendix D).
        fitnesses = [_evaluate_member(member, pbt_config.num_selection_games) for member in members]

        trainer.update_fitness(fitnesses)
        trainer.exploit_and_explore()
        updates_done += updates_this_selection
        selection_history.append({"fitnesses": list(map(float, fitnesses))})

        # Apply mutated hyperparams for next selection window.
        for midx, member in enumerate(members):
            old_lr = member.cfg.learning_rate
            member.cfg = _cfg_from_member_params(ppo_config, trainer.population[midx].params)
            if member.cfg.learning_rate != old_lr:
                member.train_state = _rebuild_state_with_lr(
                    member.train_state,
                    learning_rate=member.cfg.learning_rate,
                    max_grad_norm=member.cfg.max_grad_norm,
                )
            member.runner.train_state = member.train_state

    # Persist per-member artifacts in ppo-compatible format.
    from pathlib import Path

    root = Path(save_dir) / run_name
    root.mkdir(parents=True, exist_ok=True)
    for midx, member in enumerate(members):
        out = root / f"member{midx}" / f"seed{member.seed}"
        save_ppo_checkpoint(member.train_state.params, out / "ppo_agent")
        save_training_info(member.logs, out / "training_info.pkl")

    return {
        "run_name": run_name,
        "selection_history": selection_history,
        "final_fitnesses": [m.fitness for m in trainer.population],
        "population_params": [m.params for m in trainer.population],
    }
