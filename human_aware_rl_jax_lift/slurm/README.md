# SLURM Runbook

This directory contains the full batch suite for Overcooked reproduction on Engaging.

## Scripts

- `submit_all.sh` orchestrates the full pipeline.
- `01_bc.slurm` trains BC models for all layouts and both splits.
- `01b_bc_fig6.slurm` runs Figure 6 BC ablation (layout-specific epochs).
- `02_ppo_sp.slurm` runs PPO self-play.
- `03_ppo_bc.slurm` runs PPO with BC partners (train/test).
- `04_pbt.slurm` runs PBT per `(layout, seed)` with `AGENT_TYPE` in `{sp, bc_train, bc_test}`.

## Assumptions

- Run from module root: `human_aware_rl_jax_lift`.
- Conda env exists and is named `overcooked`.
- Required data exists under `data/bc_data/`.
- Cluster supports requested partitions/QOS/GRES (notably `gpu:h200:1`).

## Preflight

Run these before submitting:

```bash
# 1) Verify H200 gres string
sinfo -p mit_normal_gpu --format="%N %G" | grep h200

# 2) Confirm conda env
conda env list | grep overcooked

# 3) Confirm CUDA module
module avail cuda 2>&1 | grep -i cuda

# 4) Confirm BC data files
ls -lh data/bc_data/{simple,unident_s,random0,random1,random3}.pkl

# 5) Dry-run scheduler validation
sbatch --test-only slurm/02_ppo_sp.slurm
sbatch --test-only slurm/04_pbt.slurm
```

## Submit Everything

```bash
mkdir -p logs
bash slurm/submit_all.sh
```

## Monitor

```bash
watch -n 60 'squeue -u $USER -o "%.10i %.25j %.8T %.10M %.3D %R" | sort -k4'
```

## Outputs

Expected output roots:

- BC: `data/bc_runs`
- BC fig6: `data/bc_runs_fig6`
- PPO-SP / PPO-BC: `data/ppo_runs`
- PBT: `data/pbt_runs`

Logs:

- `logs/bc_*`
- `logs/bc_fig6_*`
- `logs/ppo_sp_*`
- `logs/ppo_bc_*`
- `logs/pbt_*`

## Notes

- `03_ppo_bc.slurm` and non-SP runs of `04_pbt.slurm` require `data/bc_runs/best_bc_model_paths.pkl`.
- `submit_all.sh` already encodes the dependency from BC completion to PPO-BC / PBT-BC stages.
