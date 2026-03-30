# Running the Original HARL Experiment via Apptainer + SLURM

This directory contains everything needed to reproduce the
[Carroll et al. (2019)](https://arxiv.org/abs/1910.05789) Human-Aware RL
experiments (PPO-SP and PPO-BC) on a modern SLURM cluster using Apptainer.

The container freezes the original TensorFlow 1.13 + Python 3.7 environment so
the code runs unchanged on clusters that no longer support those versions
natively.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Apptainer ≥ 1.1 | Must be available on build *and* compute nodes |
| SLURM | `sbatch`, `squeue`, `sacct` |
| `fakeroot` support | Required for `apptainer build --fakeroot` |
| ~8 GB scratch space | For the `.sif` image and Apptainer tmp/cache |

On MIT ORCD, load Apptainer with:
```bash
module load apptainer/1.4.2
```
Or set `APPTAINER_MODULE_CMD="module load apptainer/1.4.2"` when calling any
script below.

---

## Step 1 — Build the container image

The container bundles the full TF1 conda environment and the `human_aware_rl`
source tree.

### Option A: via SLURM (recommended)

```bash
cd <repo-root>          # must be the repo root, not hpc_scripts/

sbatch \
  --export=ALL,\
OUT_SIF=$HOME/containers/harl_tf1.sif,\
HOST_REPO_ROOT=$PWD,\
SCRATCH_ROOT=/scratch/$USER \
  human_aware_rl/hpc_scripts/build_harl_sif.sbatch
```

Monitor with:
```bash
squeue -u $USER
tail -f human_aware_rl/hpc_scripts/logs/build_harl_sif_<JOBID>.out
```

The job requests **2 h / 16 GB / 4 CPUs** by default. The finished image is
written to `$HOME/containers/harl_tf1.sif`.

### Option B: interactive / local

```bash
bash human_aware_rl/hpc_scripts/build_harl_sif.sh \
  $HOME/containers/harl_tf1.sif
```

`build_harl_sif.sh` wraps:
```bash
apptainer build --fakeroot --mksquashfs-args "-processors 1" \
  <OUT_SIF> human_aware_rl/hpc_scripts/harl_apptainer.def
```

The `.def` file copies the entire `human_aware_rl/` tree into `/opt/human_aware_rl`
inside the image at build time.

---

## Step 2 — Submit the reproduction jobs

### Full parallel submission (all layouts, all seeds)

`submit_harl_ground_truth_parallel.sh` submits one SLURM job per experiment
(layout × partner type). By default it submits both `ppo_sp` and `ppo_bc`
phases.

```bash
cd <repo-root>

SIF_PATH=$HOME/containers/harl_tf1.sif \
HOST_REPO_ROOT=$PWD \
HOST_GT_DIR=$PWD/ground_truth_runs \
PARTITION=mit_normal \
  bash human_aware_rl/hpc_scripts/submit_harl_ground_truth_parallel.sh
```

**Key environment variables:**

| Variable | Default | Description |
|---|---|---|
| `SIF_PATH` | `/home/tchen22/containers/harl_tf1.sif` | Path to the built `.sif` |
| `HOST_REPO_ROOT` | `$PWD` | Repo root (must contain `human_aware_rl/`) |
| `HOST_GT_DIR` | `$HOST_REPO_ROOT/ground_truth_runs` | Where outputs are written |
| `PARTITION` | `mit_normal` | SLURM partition |
| `PHASES` | `ppo_sp,ppo_bc` | Comma-separated phases to submit |
| `SPLIT_SEEDS` | `0` | `1` = one SLURM job per seed (more parallel) |
| `TIME_LIMIT` | `23:59:59` | Per-job wall-clock limit |
| `MEMORY` | `32G` | Per-job memory |
| `CPUS` | `8` | CPUs per job |
| `APPTAINER_MODULE_CMD` | *(empty)* | e.g. `module load apptainer/1.4.2` |

To submit only PPO-SP:
```bash
PHASES=ppo_sp \
SIF_PATH=$HOME/containers/harl_tf1.sif \
HOST_REPO_ROOT=$PWD \
  bash human_aware_rl/hpc_scripts/submit_harl_ground_truth_parallel.sh
```

### Single experiment (one job)

```bash
SIF_PATH=$HOME/containers/harl_tf1.sif \
HOST_REPO_ROOT=$PWD \
HOST_GT_DIR=$PWD/ground_truth_runs \
RUN_NAME=ppo_sp_simple \
LAYOUT=simple \
OTHER_AGENT_TYPE=sp \
SEEDS_JSON="[2229, 7649, 7225, 9807, 386]" \
PPO_RUN_TOT_TIMESTEPS=6000000 \
  sbatch human_aware_rl/hpc_scripts/run_harl_ground_truth.sbatch
```

---

## Step 3 — (SPLIT_SEEDS=1 only) Merge per-seed outputs

If you used `SPLIT_SEEDS=1`, each seed is written to its own directory
(`ppo_sp_simple_seed2229/`, etc.). Merge them into the format expected by the
analysis notebooks:

```bash
BASE_RUN_NAME=ppo_sp_simple \
SEEDS="2229,7649,7225,9807,386" \
HOST_GT_DIR=$PWD/ground_truth_runs \
  bash human_aware_rl/hpc_scripts/merge_seed_runs.sh
```

This produces `ground_truth_runs/ppo_sp_simple_combined/` with
`seed<N>/training_info.pickle` and `seed<N>/ppo_agent/` arranged the way
`load_training_data` expects.

---

## Step 4 — (Optional) Run BC phases inside the container

`run_harl_full_native_bc.sbatch` runs the original experiment shell scripts
(`experiments/ppo_sp_experiments_64cpu.sh`, etc.) directly inside the
container, using the BC data already embedded in `data/bc_runs`.

```bash
SIF_PATH=$HOME/containers/harl_tf1.sif \
HOST_REPO_ROOT=$PWD \
PHASES=ppo_sp,ppo_bc \
  sbatch human_aware_rl/hpc_scripts/run_harl_full_native_bc.sbatch
```

This is the closest equivalent to running the original paper scripts verbatim.
It requires `data/bc_runs/best_bc_model_paths.pickle` to exist inside the
repo tree.

---

## Output layout

After the jobs finish, `HOST_GT_DIR` will contain:

```
ground_truth_runs/
  ppo_sp_simple_seed2229/
    config.pickle
    seed2229/
      training_info.pickle
      ppo_agent/
  ...
  ppo_sp_simple_combined/       # after merge_seed_runs.sh
    config.pickle
    seed2229/
    seed7649/
    ...
```

These directories are consumed directly by the analysis notebooks in
`human_aware_rl/human_aware_rl/`.

---

## Hyperparameters used

The values below match those in `submit_harl_ground_truth_parallel.sh` and
replicate the paper's Table 1 conditions.

### PPO-SP

| Layout | Timesteps | LR | Rew-shaping horizon | VF coef |
|---|---|---|---|---|
| `simple` | 6 000 000 | 1e-3 | 2.5e6 | 0.5 |
| `unident_s` | 7 000 000 | 1e-3 | 2.5e6 | 0.5 |
| `random0` | 7 500 000 | 8e-4 | 2.5e6 | 0.5 |
| `random1` | 10 000 000 | 6e-4 | 3.5e6 | 0.5 |
| `random3` | 8 000 000 | 8e-4 | 2.5e6 | 0.5 |

Seeds (all layouts): `[2229, 7649, 7225, 9807, 386]`

### PPO-BC

| Layout | Timesteps | LR | VF coef | Minibatches |
|---|---|---|---|---|
| `simple` | 8 000 000 | 1e-3 | 0.5 | 10 |
| `unident_s` | 10 000 000 | 1e-3 | 0.5 | 12 |
| `random0` | 9 000 000 | 1.5e-3 | 0.1 | 15 |
| `random1` | 16 000 000 | 1e-3 | 0.5 | 15 |
| `random3` | 12 000 000 | 1.5e-3 | 0.1 | 15 |

Train seeds: `[9456, 1887, 5578, 5987, 516]`  
Test seeds: `[2888, 7424, 7360, 4467, 184]`

---

## Files in this directory

| File | Purpose |
|---|---|
| `harl_apptainer.def` | Apptainer definition — builds TF1/Python 3.7 container |
| `build_harl_sif.sh` | Interactive wrapper around `apptainer build` |
| `build_harl_sif.sbatch` | SLURM job to build the `.sif` on the cluster |
| `run_harl_ground_truth.sbatch` | SLURM job: runs one PPO experiment inside the container |
| `run_harl_full_native_bc.sbatch` | SLURM job: runs all original experiment scripts (BC phases) |
| `submit_harl_ground_truth_parallel.sh` | Submits all layouts × seeds as parallel SLURM jobs |
| `merge_seed_runs.sh` | Merges per-seed output directories into the combined format |
