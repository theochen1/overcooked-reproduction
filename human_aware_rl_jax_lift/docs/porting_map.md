# JAX Lift Porting Map (TF → JAX)

This document is a **traceability map** for the JAX lift.

Goal: ensure that a maintainer familiar with the deprecated TensorFlow-era `human_aware_rl` codebase can (1) locate the corresponding implementation in `human_aware_rl_jax_lift`, and (2) understand the contract each component is responsible for so failures can be localized quickly.

> Conventions
> - “Legacy/TF” refers to the deprecated `human_aware_rl` repository and its training stack.
> - “JAX lift” refers to this repository’s `human_aware_rl_jax_lift` package.
> - When a legacy concept does not exist as a 1:1 class (e.g., Python step loops), this map points to the *responsibility* in the JAX implementation.

---

## Public training API

The intended default training entrypoint is:

- `human_aware_rl_jax_lift.training.ppo_run.ppo_run`

The package init exports `ppo_run` only to keep the default public surface minimal.

To use PBT, import it explicitly:

- `from human_aware_rl_jax_lift.training.pbt_run import pbt_run`

---

## Training stack (conceptual 1-to-1)

This table maps common legacy training-stack concepts to their JAX-native locations.

| Legacy/TF concept | Responsibility | JAX lift location | What to inspect |
| --- | --- | --- | --- |
| Vectorized env / batched stepping | Maintain N env instances, step/reset them, return reward/done/info | `human_aware_rl_jax_lift.training.runner_jax` | `make_rollout_fn(...)` calls `batched_step(...)` with per-env reset keys and returns tensors + `infos` |
| Python rollout loop | Collect trajectories (obs, acts, rews, dones, values, logp) over horizon | `human_aware_rl_jax_lift.training.runner_jax` | Rollout is a single `jax.lax.scan` (no Python per-step loop) |
| Partner policy (self-play) | Partner action distribution from same policy | `human_aware_rl_jax_lift.training.runner_jax` | Partner samples via `jax.random.categorical` from the same policy network |
| Partner policy (BC) | Partner action distribution from BC logits; optional “unstuck” rule | `human_aware_rl_jax_lift.training.runner_jax` | BC logits from `BCPolicy().apply(...)`, then `_unstuck_adjust_probs(...)`, then sampling |
| SP-vs-BC mixing | Mix self-play and BC partner based on a scalar probability | `human_aware_rl_jax_lift.training.runner_jax` | Controlled by `sp_factor`; trajectory-level if `trajectory_sp=True` (sample at episode reset and hold) |
| PPO driver | Rollout → GAE → minibatch PPO updates → checkpoint/logging | `human_aware_rl_jax_lift.training.ppo_run` | `ppo_run(...)`, `_run_update_epochs(...)`, `compute_gae(...)`, `ppo_update_step(...)` |
| Reward shaping schedule | Anneal shaped reward contribution over training | `human_aware_rl_jax_lift.training.ppo_run` | `annealed_shaping_factor(...)` updates `shaping_factor` |
| Training reward metrics | Compute smoothed `eprewmean` and sparse `true_eprew` | `runner_jax` + `ppo_run` | Runner returns `completed_eprew` and `completed_ep_sparse_rew`; PPO buffers them (deque) and logs |
| PBT orchestration | Population, selection windows, exploit/explore, weight copying, hparam mutation | `human_aware_rl_jax_lift.training.pbt_run` | `PBTTrainer`, `_evaluate_member(...)`, and member loops |
| PBT “native JAX” minibatching | Avoid NumPy randomness in minibatch shuffling | `human_aware_rl_jax_lift.training.pbt_run` | Minibatch shuffle uses `jax.random.permutation` with PRNG threading |

---

## PPO end-to-end walkthrough (where to debug)

This section mirrors the mental model of the TF training loop but points to the JAX-native code.

1. **Entry**: `ppo_run(layout_name, seeds, config, other_agent_type, ...)` creates a run directory and iterates seeds.
2. **Partner selection**:
   - If `other_agent_type in {"bc_train","bc_test"}`, it loads a BC checkpoint from `best_bc_model_paths[split][layout_name]` and passes `bc_params` into the rollout builder.
   - Otherwise the partner is pure self-play.
3. **Rollout (on-device)**: `make_rollout_fn(...)` produces a JIT’d function that runs a full horizon via `jax.lax.scan`.
4. **GAE/returns**: `compute_gae(rewards, values, dones, ...)` produces advantages and returns.
5. **Update**: PPO flattens `[T,N,...]` into `[T*N,...]` and runs multiple epochs/minibatches through `ppo_update_step(...)`.
6. **Logging**:
   - `eprewmean` comes from accumulated `rollout.rewards` episode returns.
   - `true_eprew` in the printed table is the *sparse* episode return mean (`ep_sparse_rew_mean`).

Interpretability note: the metric computation is deterministic given a rollout, but rollouts are stochastic because actions are sampled (`jax.random.categorical`) for both the agent and the partner.

---

## Partner behavior details (BC vs self-play)

When `bc_params` are provided, the rollout includes:

- **Self-play partner**: partner actions sampled from the PPO policy network.
- **BC partner**: partner actions sampled from BC logits (softmax probabilities).
- **Unstuck rule**: when the partner is “stuck” (repeated position history), the code masks recently-taken actions and renormalizes probabilities.
- **Mixing semantics**: SP-vs-BC is decided by `sp_factor` and optionally held constant over an episode when `trajectory_sp=True`.

These mechanics are centralized in `human_aware_rl_jax_lift.training.runner_jax`.

---

## PBT notes (what is JAX-native vs what remains Python)

- The heavy path (rollouts + PPO updates) uses the same JAX rollout primitives as PPO.
- The outer loop (selection windows, exploit/explore decisions, artifact writing) is host-controlled Python, which matches the legacy structure while keeping the compute-heavy portion JAX-friendly.

---

## Canonical JAX modules

The training directory is intended to be natively JAX:

- PPO training: `human_aware_rl_jax_lift/training/ppo_run.py`
- Rollout/partner logic: `human_aware_rl_jax_lift/training/runner_jax.py`
- PBT training: `human_aware_rl_jax_lift/training/pbt_run.py`

There is no wrapper module `training/ppo_run_jax.py`, and there is no legacy non-JAX `training/vec_env.py`.

---

## Appendix: legacy-to-JAX symbol map

The following table lists specific known symbol-level equivalences and is useful as an index for quickly jumping between the legacy repo and the lifted repo.

| Legacy path + symbol | JAX path + symbol | Semantic contract | Known differences | Test that proves equivalence |
| --- | --- | --- | --- | --- |
| `human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py::get_state_transition` | `human_aware_rl_jax_lift/env/overcooked_mdp.py::step` | Deterministic transition: resolve interacts, then movement, then environment effects |  | `tests/test_env_equivalence.py::test_step_level_parity` |
| `human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py::resolve_interacts` | `human_aware_rl_jax_lift/env/interactions.py::resolve_interacts` | Interactions over `X/O/T/D/P/S` in player order; sparse reward only on serving |  | `tests/test_env_equivalence.py::test_interact_semantics` |
| `human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py::_handle_collisions` | `human_aware_rl_jax_lift/env/collisions.py::resolve_player_collisions` | Same-cell and swap collisions force both players to stay in old positions |  | `tests/test_env_equivalence.py::test_collision_parity` |
| `human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py::step_environment_effects` | `human_aware_rl_jax_lift/env/overcooked_mdp.py::_step_environment_effects` | Full soups in pots increment cook time up to `cook_time` cap |  | `tests/test_env_equivalence.py::test_pot_timer_parity` |
| `human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py::lossless_state_encoding` | `human_aware_rl_jax_lift/encoding/ppo_masks.py::lossless_state_encoding_20` | 20-channel lossless state encoding with player-perspective channel ordering |  | `tests/test_encoding_parity.py::test_ppo_masks_match_legacy` |
| `human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py::featurize_state` | `human_aware_rl_jax_lift/encoding/bc_features.py::featurize_state_64` | Handcrafted BC features with object/wall/distance features and relative positions |  | `tests/test_encoding_parity.py::test_bc_features_match_legacy` |
| `human_aware_rl/human_aware_rl/imitation/behavioural_cloning.py::bc_from_dataset_and_params` | `human_aware_rl_jax_lift/agents/bc/train.py::train_bc` | Supervised cross-entropy BC optimization with Adam |  | `tests/test_model_io.py::test_bc_logits_shape_and_dtype` |
| `human_aware_rl/human_aware_rl/imitation/behavioural_cloning.py::ImitationAgentFromPolicy.unblock_if_stuck` | `human_aware_rl_jax_lift/agents/bc/agent.py::BCAgent._unstuck_adjust` | If stuck for 3+ timesteps, suppress previously repeated stuck actions and renormalize probs |  | `tests/test_model_io.py::test_unstuck_rule_masks_actions` |
| `human_aware_rl/human_aware_rl/baselines_utils.py::conv_network_fn` | `human_aware_rl_jax_lift/agents/ppo/model.py::ActorCriticCNN.__call__` | 3 conv layers (5x5, 3x3, 3x3, 25 filters) then 3 FC layers |  | `tests/test_model_io.py::test_ppo_network_output_shapes` |
| `human_aware_rl/human_aware_rl/ppo/ppo.py::ppo_run` | `human_aware_rl_jax_lift/agents/ppo/train.py::ppo_update_step` | PPO clipped policy/value optimization with entropy regularization |  | `tests/test_model_io.py::test_ppo_loss_terms_finite` |
| `human_aware_rl/human_aware_rl/pbt/pbt.py::PBTAgent.mutate_params` | `human_aware_rl_jax_lift/agents/pbt/trainer.py::PopulationMember.mutate` | Mutation over `{LAM, CLIPPING, LR, STEPS_PER_UPDATE, ENTROPY, VF_COEF}` with resample prob 0.33 and factors [0.75, 1.25] |  | `tests/test_model_io.py::test_pbt_mutation_ranges` |
| `human_aware_rl/human_aware_rl/experiments/ppo_sp_experiments.py::run_all_ppo_sp_experiments` | `human_aware_rl_jax_lift/experiments/figure4.py::run` | Figure script orchestrates fixed seeds and layout-specific hyperparameters |  | `tests/test_e2e_figures.py::test_figure4_statistical_reproduction` |
| `human_aware_rl/human_aware_rl/experiments/pbt_experiments.py::run_all_pbt_experiments` | `human_aware_rl_jax_lift/experiments/figure5.py::run` | Figure script orchestrates PBT population and seed sweeps |  | `tests/test_e2e_figures.py::test_figure5_statistical_reproduction` |
| `human_aware_rl/human_aware_rl/experiments/bc_experiments.py::run_all_bc_experiments` | `human_aware_rl_jax_lift/experiments/figure6.py::run` | Figure script orchestrates BC training/eval sweeps |  | `tests/test_e2e_figures.py::test_figure6_statistical_reproduction` |
