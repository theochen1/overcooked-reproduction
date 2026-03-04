#!/bin/bash
#SBATCH --job-name=oc_xparity_bc_tf
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xparity_bc_tf.out
#SBATCH --error=logs/xparity_bc_tf.err

set -euo pipefail

# ---- user config ----
REPO_ROOT="${SLURM_SUBMIT_DIR}"
TRAIN_PKL="${REPO_ROOT}/human_aware_rl/human_aware_rl/data/human/anonymized/clean_train_trials.pkl"
TEST_PKL="${REPO_ROOT}/human_aware_rl/human_aware_rl/data/human/anonymized/clean_test_trials.pkl"
OUT_DIR="${REPO_ROOT}/experiments/parity_bc_data"
# ---------------------

cd "${REPO_ROOT}"
mkdir -p logs "${OUT_DIR}"

# If your cluster uses modules, uncomment + edit:
# module purge
# module load cuda/12.2
# module load anaconda3
# source activate overcooked

# Select layout by array index (0..4)
LAYOUTS=(simple unident_s random0 random1 random3)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LAYOUT="${LAYOUTS[$TASK_ID]}"

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] SLURM_JOB_ID=${SLURM_JOB_ID} TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Layout=${LAYOUT}"
echo "[$(date)] Train pkl=${TRAIN_PKL}"
echo "[$(date)] Test  pkl=${TEST_PKL}"

# Optional: force JAX to CPU to avoid CUDA init issues on some setups
# (even on GPU nodes, this can make featurization more robust)
# export JAX_PLATFORMS=cpu
# export CUDA_VISIBLE_DEVICES=""

python ../scripts/parity_bc_data_tf_vs_jax.py \
  --layout "${LAYOUT}" \
  --tf_train_pkl "${TRAIN_PKL}" \
  --tf_test_pkl  "${TEST_PKL}" \
  --out_dir "${OUT_DIR}"

echo "[$(date)] Done."
