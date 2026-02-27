"""Population-based training loop using JAX rollout primitives.

Ported to use runner_jax instead of legacy VectorizedEnv/RolloutRunner/partners.
"""

import copy
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

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
from human_aware_rl_jax_lift.reproducibility.seed import set_global_seed

from .checkpoints import save_ppo_checkpoint, save_training_info
from .ppo_run import compute_self_play_factor
from .runner_jax import make_rollout_fn
from .vec_env_jax import encode_obs, make_batched_state


def _flatten(x: jnp.ndarray) -> jnp.ndarray:
    t, n = x.shape[:2]
    return x.reshape((t * n,) + x.shape[2:])


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
    return state.replace(tx=tx)


@dataclass
class _MemberRuntime:
    cfg: PPOConfig
    train_state: TrainState
    rollout_fn: callable
    bstate: object
    obs0: jnp.ndarray
    total_steps: int
    logs: Dict[str, list]
    seed: int
    bc_params: Optional[dict]


def _evaluate_member(member: _MemberRuntime, num_selection_games: int, terrain) -> float:
    """Evaluate by rolling out until at least `num_selection_games` episodes are observed."""
    rng = set_global_seed(member.seed + 999_999 + member.total_steps)
    sparse_means = []
    episodes = 0
    # For eval, use sp_factor=0 (pure BC partner if BC, else pure SP)
    while episodes < num_selection_games:
        rng, eval_rng = jax.random.split(rng)
        rollout, member.bstate, member.obs0 = member.rollout_fn(
            member.train_state,
            member.bstate,
            member.obs0,
            shaping_factor=1.0,
            sp_factor=0.0,
            rng=eval_rng,
        )
        eps = int(rollout.infos.get("episodes_this_rollout", 0))
        if eps > 0:
            sparse_means.append(float(rollout.infos["ep_sparse_rew_mean"]))
            episodes += eps
    return float(np.mean(sparse_means))


def pbt_run(
    layout_name: str,
    seeds: List[int],
    ppo_config: PPOConfig,
    pbt_config: PBTConfig,
    total_steps_per_agent: int,
    other_agent_type: str = "sp",
    best_bc_model_paths: Optional[dict] = None,
    save_dir: str = "data/pbt_runs",
    ex_name: Optional[str] = None,
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

    # Load BC params if needed
    bc_params = None
    if other_agent_type in ("bc_train", "bc_test"):
        if best_bc_model_paths is None:
            raise ValueError("best_bc_model_paths required for BC-partner PBT")
        split = "train" if other_agent_type == "bc_train" else "test"
        bc_path = Path(best_bc_model_paths[split][layout_name])
        if bc_path.is_dir():
            bc_path = bc_path / "model.pkl"
        with bc_path.open("rb") as f:
            payload = pickle.load(f)
        bc_params = payload.get("params", payload) if isinstance(payload, dict) else payload

    trainer = PBTTrainer(base_hparams=_member_params_from_cfg(ppo_config), config=pbt_config)
    members: List[_MemberRuntime] = []

    for m_idx in range(pbt_config.population_size):
        seed = int(seeds[m_idx])
        rng = set_global_seed(seed)

        # Probe obs shape
        rng, probe_rng = jax.random.split(rng)
        probe_bstate = make_batched_state(terrain, 1, probe_rng)
        probe_obs0, _ = encode_obs(terrain, probe_bstate)
        obs_shape = probe_obs0.shape[1:]

        # Create train state with mutated hyperparams
        cfg = _cfg_from_member_params(ppo_config, trainer.population[m_idx].params)
        state = create_train_state(rng, obs_shape, cfg)

        # Create batched env state
        rng, env_rng = jax.random.split(rng)
        bstate = make_batched_state(
            terrain, cfg.num_envs, env_rng, randomize_agent_idx=cfg.randomize_agent_idx
        )
        obs0, _ = encode_obs(terrain, bstate)

        # Build rollout fn
        rollout_fn = make_rollout_fn(
            terrain=terrain,
            horizon=cfg.horizon,
            num_envs=cfg.num_envs,
            randomize_agent_idx=cfg.randomize_agent_idx,
            bootstrap_with_zero_obs=cfg.bootstrap_with_zero_obs,
            bc_params=bc_params,
            trajectory_sp=getattr(cfg, "trajectory_self_play", True),
        )

        members.append(
            _MemberRuntime(
                cfg=cfg,
                train_state=state,
                rollout_fn=rollout_fn,
                bstate=bstate,
                obs0=obs0,
                total_steps=0,
                logs={"ep_sparse_rew_mean": [], "eprewmean": [], "loss": []},
                seed=seed,
                bc_params=bc_params,
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
                shaping = 1.0  # Could anneal if needed
                sp_factor = compute_self_play_factor(member.total_steps, member.cfg.self_play_horizon)

                rng, rollout_rng = jax.random.split(rng)
                rollout, member.bstate, member.obs0 = member.rollout_fn(
                    member.train_state,
                    member.bstate,
                    member.obs0,
                    shaping_factor=shaping,
                    sp_factor=sp_factor,
                    rng=rollout_rng,
                )

                adv, ret = compute_gae(
                    jnp.asarray(rollout.rewards, dtype=jnp.float32),
                    jnp.asarray(rollout.values, dtype=jnp.float32),
                    jnp.asarray(rollout.dones, dtype=jnp.float32),
                    float(member.cfg.gamma),
                    float(member.cfg.gae_lambda),
                    bootstrap_value=jnp.asarray(rollout.next_value, dtype=jnp.float32),
                )

                obs_flat = _flatten(jnp.asarray(rollout.obs, dtype=jnp.float32))
                actions_flat = _flatten(jnp.asarray(rollout.actions, dtype=jnp.int32))
                logp_flat = _flatten(jnp.asarray(rollout.log_probs, dtype=jnp.float32))
                values_flat = _flatten(jnp.asarray(rollout.values, dtype=jnp.float32))
                adv_flat = _flatten(jnp.asarray(adv))
                ret_flat = _flatten(jnp.asarray(ret))

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

                member.logs["loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
                member.logs["eprewmean"].append(float(rollout.infos["eprewmean"]))
                member.logs["ep_sparse_rew_mean"].append(float(rollout.infos["ep_sparse_rew_mean"]))

        # Selection fitness from explicit evaluation episodes
        fitnesses = [_evaluate_member(member, pbt_config.num_selection_games, terrain) for member in members]

        trainer.update_fitness(fitnesses)
        copy_map = trainer.exploit_and_explore()
        updates_done += updates_this_selection
        selection_history.append({"fitnesses": list(map(float, fitnesses))})

        # Copy network weights best -> replaced members
        for replaced_idx, src_idx in copy_map.items():
            members[replaced_idx].train_state = members[replaced_idx].train_state.replace(
                params=copy.deepcopy(members[src_idx].train_state.params)
            )

        # Apply mutated hyperparams for next selection window
        for midx, member in enumerate(members):
            old_lr = member.cfg.learning_rate
            member.cfg = _cfg_from_member_params(ppo_config, trainer.population[midx].params)
            if member.cfg.learning_rate != old_lr:
                member.train_state = _rebuild_state_with_lr(
                    member.train_state,
                    learning_rate=member.cfg.learning_rate,
                    max_grad_norm=member.cfg.max_grad_norm,
                )

    # Persist per-member artifacts in ppo-compatible format
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
