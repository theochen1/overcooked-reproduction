# Deprecated to JAX Mapping

This document provides a 1-to-1 migration map from deprecated TF1 pipelines to
the current JAX/PyTorch stack used in `overcooked_ai-master`.

## Mapping Table

| Deprecated file + function(s) | JAX/PyTorch file + function(s) | Equivalence class | Intentional deviations | Validation / parity test |
| --- | --- | --- | --- | --- |
| `human_aware_rl/human_aware_rl/ppo/ppo.py` (run naming + results layout) | `src/human_aware_rl/ppo/run_paths.py` (`build_training_output_paths`, `format_run_template`), `src/human_aware_rl/ppo/run_registry_defaults.py` (`DEFAULT_RUN_NAME_TEMPLATES`, `DEFAULT_AGENT_DIRS`) | Canonical run/seed/agent path resolution | Canonicalized templates and stricter seed/agent partitioning | `jaxmarl/checkpoint_eval_parity.py`, strict run-registry loading in `evaluation/evaluate_paper.py` |
| `human_aware_rl/human_aware_rl/experiments/ppo_sp_experiments.py` | `src/human_aware_rl/ppo/train_ppo_sp.py` (`train_ppo_sp`, `train_all_layouts`) | PPO self-play orchestration | JAX trainer, deterministic run registry, explicit `canonical_paper_entrypoint` | `jaxmarl/rollout_parity_canary.py`, `jaxmarl/parity_check.py` |
| `human_aware_rl/human_aware_rl/experiments/ppo_bc_experiments.py` | `src/human_aware_rl/ppo/train_ppo_bc.py` (`train_ppo_bc`, `train_all_layouts`) | PPO with BC-partner schedule | Uses `bc_schedule` tuples + JAX env wrappers | `jaxmarl/baselines_identity_checks.py`, end-to-end Figure 4 checks |
| `human_aware_rl/human_aware_rl/experiments/ppo_hm_experiments.py` | `src/human_aware_rl/ppo/train_ppo_hp.py` (`train_ppo_hp`, `train_all_layouts`) | Gold-standard PPO trained with HP partner | Run template normalization + parity enforcement on canonical entrypoints | Figure 4 gold-line parity checks via `evaluation/evaluate_paper.py` |
| `human_aware_rl/human_aware_rl/ppo/ppo.py` (TF baselines PPO core) | `src/human_aware_rl/jaxmarl/ppo.py` (`PPOConfig`, `PPOTrainer`, `assert_paper_parity`) | PPO algorithm implementation | JAX/Flax backend, explicit schedule semantics, strict parity assertions when canonical mode is enabled | `jaxmarl/parity_check.py`, `jaxmarl/minibatch_tf_vs_jax.py` |
| `human_aware_rl/human_aware_rl/imitation/behavioural_cloning.py` | `src/human_aware_rl/imitation/behavior_cloning.py` (`train_bc_model`, `load_bc_model`) | Behavior cloning training/inference | Migrated to PyTorch + tensorboard tooling | `src/human_aware_rl/imitation/behavior_cloning_test.py` |
| Notebook checkpoint search logic (`NeurIPS Experiments and Visualizations.ipynb`) | `src/human_aware_rl/evaluation/evaluate_paper.py` (`find_checkpoint_from_run`, `evaluate_paper_config`), `src/human_aware_rl/evaluation/run_paper_evaluation.py` | Paper evaluation loader and aggregation | `paper_strict` mode blocks legacy scans for paper runs | Figure 4(a)/(b) deterministic seed-by-seed runs |
| Legacy ad-hoc plotting cells in notebook | `src/human_aware_rl/visualization/plot_results.py`, `src/human_aware_rl/visualization/plot_paper_figure4.py`, `src/human_aware_rl/visualization/plot_paper_figures.py` | Figure generation pipeline | Scripted CLI outputs (PDF/PNG/CSV-backed) instead of manual notebook cells | Visual regression against paper reference figures |

## Notes on Equivalence Scope

- "Equivalent" means reproducing training/evaluation behavior at the level of
  paper statistics and trend ordering, not bitwise equality between TF and JAX.
- Canonical paper runs are expected to use:
  - `paper_configs.get_ppo_sp_config`
  - `paper_configs.get_ppo_bc_config`
  - strict run-registry path loading (`paper_strict=True`)

## High-Risk Deviation Classes to Monitor

- Changing run-name templates or agent-dir names without updating all loaders.
- Enabling legacy `results/*` fallback in paper-critical evaluations.
- Silent hyperparameter overrides on canonical paper entrypoints.
- Mismatch between `layout` and `LAYOUT_TO_ENV` mappings for legacy MDP parity.
