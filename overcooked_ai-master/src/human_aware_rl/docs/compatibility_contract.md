# Compatibility Contract

This document defines the paper-reproduction compatibility contract for the
modern `overcooked_ai-master/src/human_aware_rl` stack.

## Canonical Run Directory Contract

All PPO-family runs must be written under:

`DATA_DIR/ppo_runs/<run_name>/seed<seed>/<agent_name>/checkpoint_*`

Example:

`DATA_DIR/ppo_runs/ppo_sp__layout-cramped_room/seed0/ppo_agent/checkpoint_000500`

### Required artifacts

For each `checkpoint_*` directory:

- `params.pkl` (model parameters)
- `config.pkl` (serialized `PPOConfig`)
- `params.sha256` (checksum, when available)

For the run root (`.../seed<seed>/<agent_name>/`):

- `config.json` (JSON training config used by script entrypoint)
- `metrics.json` (summary metrics used by evaluation/export pipelines)
- `run_manifest.json` (resolved config, git SHA, package versions)

## Canonical Naming Templates

Canonical template/agent names are shared by training, evaluation, and plotting
entrypoints via `human_aware_rl.ppo.run_registry_defaults`.

### `DEFAULT_RUN_NAME_TEMPLATES`

- `ppo_sp`: `ppo_sp__layout-{layout}`
- `ppo_bc`: `ppo_bc__partner-bc_train__layout-{layout}`
- `ppo_hp`: `ppo_hp__layout-{layout}`

### `DEFAULT_AGENT_DIRS`

- `ppo_sp`: `ppo_agent`
- `ppo_bc`: `ppo_bc_agent`
- `ppo_hp`: `ppo_hp_agent`

## Paper Layout Contract

Paper layout list (`PAPER_LAYOUTS`) and canonical environment mapping
(`LAYOUT_TO_ENV`) are defined in
`human_aware_rl.ppo.configs.paper_configs`:

- `cramped_room` -> `cramped_room_legacy`
- `asymmetric_advantages` -> `asymmetric_advantages_legacy`
- `coordination_ring` -> `coordination_ring_legacy`
- `forced_coordination` -> `random0_legacy`
- `counter_circuit` -> `random3_legacy`

These mappings are required for parity with the 2019 paper MDP settings.

## Paper-Critical Hyperparameters

The following fields are considered paper-critical and must match
`paper_configs.get_ppo_*_config(...)` when running canonical paper entrypoints:

- `old_dynamics=True`
- `clip_eps=0.05`
- `gae_lambda=0.98`
- fixed entropy (`ent_coef=0.1`, `entropy_coeff_start=0.1`,
  `entropy_coeff_end=0.1`, `use_entropy_annealing=False`)
- `max_grad_norm=0.1`
- `horizon=400`
- `num_envs=30` (`num_workers` in paper configs)
- `num_minibatches`:
  - PPO-SP: `6`
  - PPO-BC/PPO-HP: per-layout values from Table 3
- `reward_shaping_horizon`: per-layout values from `paper_configs`
- `cliprange_schedule="constant"`
- `use_legacy_encoding=False` (lossless 26-channel encoding)

Note on value loss coefficient:

- PPO-SP uses `vf_coef=0.5` in current paper config table.
- PPO-BC/PPO-HP use per-layout values (`0.5` for some layouts, `0.1` for
  `forced_coordination` and `counter_circuit`).
