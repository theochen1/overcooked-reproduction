#!/bin/bash
#SBATCH --job-name=jax_sweep_s1
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/jax_sweep_s1_%j.out
#SBATCH --error=logs/jax_sweep_s1_%j.err

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -d "${REPO_ROOT}/scripts" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/../scripts" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  echo "ERROR: Could not resolve REPO_ROOT. Set REPO_ROOT explicitly."
  exit 1
fi

RUN_TAG="${RUN_TAG:-s1_default}"
LAYOUT="${LAYOUT:-simple}"
SEEDS="${SEEDS:-2229 7649}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1200000}"
LR="${LR:-0.001}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
VF_COEF="${VF_COEF:-0.5}"
ENT_COEF="${ENT_COEF:-0.1}"

echo "Node: $(hostname) | Start: $(date)"
echo "RUN_TAG=${RUN_TAG}"
echo "SEEDS=${SEEDS}"
echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "LR=${LR} MAX_GRAD_NORM=${MAX_GRAD_NORM} VF_COEF=${VF_COEF} ENT_COEF=${ENT_COEF}"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate overcooked

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

cd "${REPO_ROOT}"

python -u "${REPO_ROOT}/scripts/train_ppo_sp.py" \
  --layout "${LAYOUT}" \
  --seeds ${SEEDS} \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_rate "${LR}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --vf_coef "${VF_COEF}" \
  --ent_coef "${ENT_COEF}" \
  --save_dir "data/ppo_runs_sweep/${RUN_TAG}" \
  --jax \
  --diagnostics

echo "Done: $(date)"
