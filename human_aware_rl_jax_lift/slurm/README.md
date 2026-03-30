# SLURM Runbook

Full batch suite for reproducing Carroll et al. (NeurIPS 2019) on MIT Engaging.

## Pipeline Overview

```
00  prepare_bc_data   (CPU)        Featurize human trajectories
01  bc                (CPU)        Train BC models (train+test splits, 5 layouts)
01b bc_fig6           (CPU)        Figure 6 BC ablation (layout-specific epochs)
02  ppo_sp            (GPU)        PPO self-play
02b plot_figure8      (CPU)        Plot PPO-SP training curves (Figure 8)
03  ppo_bc            (GPU, dep: 01)  PPO with BC partner (train+test splits)
03b plot_figure9      (CPU)        Plot PPO-BC training curves (Figure 9)
04  pbt               (GPU, dep: 01 for bc_train/bc_test)  PBT (sp / bc_train / bc_test)
05  eval_figure4a     (GPU, dep: 01+02+03)  Evaluate all agents vs human proxy
05b plot_figure4a     (CPU, dep: 05)  Aggregate and plot Figure 4a
```

## Prerequisites

- Run all commands from `human_aware_rl_jax_lift/` (the repo root for the JAX package).
- Conda env named `overcooked` with `human_aware_rl_jax_lift` installed (`pip install -e .`).
- Human trajectory data available in `../human_aware_rl/` (the legacy TF repo).
- Cluster supports requested partitions/QOS/GRES (MIT Engaging defaults).

## Scripts

| Script | Description |
|--------|-------------|
| `submit_all.sh` | Orchestrates the full pipeline with SLURM dependencies |
| `00_prepare_bc_data.slurm` | Featurize human trajectories into `data/bc_data/*.pkl` |
| `01_bc.slurm` | Train BC models for all 5 layouts, train+test splits (10 array tasks) |
| `01b_bc_fig6.slurm` | Figure 6 BC ablation with layout-specific epoch counts (10 array tasks) |
| `02_ppo_sp.slurm` | PPO self-play, all 5 layouts (5 array tasks) |
| `02b_plot_figure8.slurm` | Plot PPO-SP training curves (Figure 8) |
| `03_ppo_bc.slurm` | PPO with BC partner, train+test splits (10 array tasks) |
| `03b_plot_figure9.slurm` | Plot PPO-BC training curves (Figure 9) |
| `04_pbt.slurm` | PBT per `(layout, seed)` with `AGENT_TYPE` in `{sp, bc_train, bc_test}` (25 array tasks each) |
| `05_eval_figure4a.slurm` | Evaluate all agent types vs human proxy (5 array tasks) |
| `05b_plot_figure4a.slurm` | Aggregate eval results and plot Figure 4a |

## Preflight

```bash
# 1) Verify GPU GRES
sinfo -p mit_normal_gpu --format="%N %G" | head

# 2) Confirm conda env
conda env list | grep overcooked

# 3) Confirm CUDA module
module avail cuda 2>&1 | grep -i cuda

# 4) Confirm human trajectory data exists
ls ../human_aware_rl/human_aware_rl/data/human/anonymized/

# 5) Dry-run scheduler validation
sbatch --test-only slurm/02_ppo_sp.slurm
```

## Step-by-Step Reproduction

### Option A: Full pipeline (automated)

```bash
mkdir -p logs
bash slurm/submit_all.sh
```

This submits all stages with correct dependencies. BC must finish before PPO-BC and PBT-BC stages start.

### Option B: Manual, stage by stage

```bash
mkdir -p logs

# 0) Prepare BC data (one-time, CPU)
sbatch slurm/00_prepare_bc_data.slurm
# Wait for completion, then verify:
ls -lh data/bc_data/{simple,unident_s,random0,random1,random3}.pkl

# 1) Train BC models
BC_JOB=$(sbatch --parsable slurm/01_bc.slurm)
# Optional: Figure 6 ablation
sbatch slurm/01b_bc_fig6.slurm

# 2) PPO self-play (no BC dependency)
SP_JOB=$(sbatch --parsable slurm/02_ppo_sp.slurm)
# Optional: plot training curves after SP finishes
sbatch --dependency=afterok:${SP_JOB} slurm/02b_plot_figure8.slurm

# 3) PPO with BC partner (depends on BC)
PPOBC_JOB=$(sbatch --parsable --dependency=afterok:${BC_JOB} slurm/03_ppo_bc.slurm)
# Optional: plot training curves after PPO-BC finishes
sbatch --dependency=afterok:${PPOBC_JOB} slurm/03b_plot_figure9.slurm

# 4) PBT (sp has no dependency; bc_train/bc_test depend on BC)
sbatch slurm/04_pbt.slurm                                          # AGENT_TYPE=sp (default)
sbatch --dependency=afterok:${BC_JOB} --export=ALL,AGENT_TYPE=bc_train slurm/04_pbt.slurm
sbatch --dependency=afterok:${BC_JOB} --export=ALL,AGENT_TYPE=bc_test  slurm/04_pbt.slurm

# 5) Evaluate all agents (depends on BC, PPO-SP, PPO-BC)
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${BC_JOB}:${SP_JOB}:${PPOBC_JOB} \
    slurm/05_eval_figure4a.slurm)

# 5b) Plot Figure 4a
sbatch --dependency=afterok:${EVAL_JOB} slurm/05b_plot_figure4a.slurm
```

## Monitor

```bash
watch -n 60 'squeue -u $USER -o "%.10i %.25j %.8T %.10M %.3D %R" | sort -k4'
```

## Outputs

| Stage | Output directory |
|-------|-----------------|
| BC data | `data/bc_data/` |
| BC models | `data/bc_runs/` (+ `best_bc_model_paths.pkl`) |
| BC fig6 | `data/bc_runs_fig6/` |
| PPO-SP / PPO-BC | `data/ppo_runs/` |
| PBT | `data/pbt_runs/` |
| Eval results | `experiments/eval_results/` |
| Figure 4a | `experiments/figure_4a.png` |

Logs: `logs/{prep_bc,bc,bc_fig6,ppo_sp,ppo_bc,pbt,fig4a_eval,fig4a_plot}_*.{out,err}`

## Dependencies Between Stages

- `03_ppo_bc.slurm` and PBT-BC runs require `data/bc_runs/best_bc_model_paths.pkl` (produced by `01_bc.slurm`).
- `05_eval_figure4a.slurm` requires trained BC, PPO-SP, and PPO-BC models.
- `submit_all.sh` encodes these dependencies automatically.

## Notes

- Layouts: `simple`, `unident_s`, `random1`, `random0`, `random3`.
- `04_pbt.slurm` uses the `AGENT_TYPE` environment variable (`sp`, `bc_train`, or `bc_test`). Default is `sp`.
- Plot scripts (`02b`, `03b`, `05b`) are CPU-only and fast; they can also be run locally outside SLURM.
