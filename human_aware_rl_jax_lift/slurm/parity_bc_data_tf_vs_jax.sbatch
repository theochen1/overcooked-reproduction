#!/bin/bash
#SBATCH --job-name=oc_parity_bc_tf
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/parity_bc_tf_%A_%a.out
#SBATCH --error=logs/parity_bc_tf_%A_%a.err

set -euo pipefail

# REPO_ROOT is two levels up from slurm/:
#   overcooked-reproduction/human_aware_rl_jax_lift/slurm  ->  overcooked-reproduction/
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
TRAIN_PKL="${REPO_ROOT}/human_aware_rl/human_aware_rl/data/human/anonymized/clean_train_trials.pkl"
TEST_PKL="${REPO_ROOT}/human_aware_rl/human_aware_rl/data/human/anonymized/clean_test_trials.pkl"
OUT_DIR="${REPO_ROOT}/experiments/parity_bc_data"

mkdir -p "${SLURM_SUBMIT_DIR}/../logs" "${OUT_DIR}"

module load miniforge
conda activate overcooked

export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

LAYOUTS=(simple unident_s random0 random1 random3)
LAYOUT="${LAYOUTS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] SLURM_JOB_ID=${SLURM_JOB_ID} TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Layout=${LAYOUT}"
echo "[$(date)] Train pkl=${TRAIN_PKL}"
echo "[$(date)] Test  pkl=${TEST_PKL}"

# NOTE: scripts/parity_bc_data_tf_vs_jax.py does not yet exist.
# Create it before submitting this job.
# See scripts/compare_tf_jax_minibatch.py as the closest existing reference.
cd "${SLURM_SUBMIT_DIR}/.."

python scripts/parity_bc_data_tf_vs_jax.py \
  --layout       "${LAYOUT}" \
  --tf_train_pkl "${TRAIN_PKL}" \
  --tf_test_pkl  "${TEST_PKL}" \
  --out_dir      "${OUT_DIR}"

echo "[$(date)] Done."
