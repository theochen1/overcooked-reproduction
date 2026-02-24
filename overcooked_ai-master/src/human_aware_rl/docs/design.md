# Design: Faithful Figure 4-7 Reproduction

## Scope

This document defines invariants and intentional differences for paper-faithful reproduction in `human_aware_rl`.

## Invariants

- Canonical training artifacts are written under `DATA_DIR/ppo_runs/<run>/seed<seed>/<agent>/checkpoint_*`.
- Canonical run naming and agent subdirectories come from `run_registry_defaults.py`.
- Paper evaluation in strict mode resolves PPO-like checkpoints only from canonical run-registry paths.
- Paper seeds are fixed to `0,10,20,30,40` unless explicitly overridden for non-paper ablations.
- `canonical_paper_entrypoint=True` in PPO training enforces parity checks against `paper_configs`.

## Paper-Critical Hyperparameters

When `canonical_paper_entrypoint=True`, training rejects mismatches for critical fields including:

- dynamics and encoding: `old_dynamics`, `use_legacy_encoding`
- PPO objective/schedules: `clip_eps`, `cliprange_schedule`, `gae_lambda`
- optimization shape: `num_minibatches`, `num_epochs`, `max_grad_norm`, `vf_coef`
- entropy behavior: `ent_coef`, `entropy_coeff_start`, `entropy_coeff_end`, `use_entropy_annealing`
- rollout/environment: `horizon`, `num_envs`
- shaping schedule: `reward_shaping_horizon`

## Strict Evaluation Rules

- `--paper_strict` defaults to enabled for paper evaluation commands.
- In strict mode, checkpoint lookup does not scan legacy `results/*` trees.
- Missing canonical checkpoints raise deterministic errors with expected canonical path details.
- Strict mode requires run-registry loading for PPO-like sources (`ppo_sp`, `ppo_bc`, `ppo_hp`, `pbt`).

## Intentional Differences vs. Deprecated Stack

- Runtime framework differs (JAX/Flax and PyTorch vs. TF1), so bitwise determinism is not expected.
- Equivalence target is statistical parity (means, stderr, trend/order), not exact trajectory replay.
- Legacy fallback discovery behavior is intentionally removed in strict mode for reproducibility safety.

## Validation Strategy

- Unit/static checks:
  - strict checkpoint resolver raises on missing canonical paths
  - canonical parity assertion raises on paper-critical override mismatches
- Reproduction checks:
  - run acceptance criteria in `reproduction_acceptance.md`
  - compare generated Figure 4-7 artifacts against expected trend/order and tolerance bands
- Provenance:
  - verify run manifests and data provenance metadata (`data_provenance.md`)
