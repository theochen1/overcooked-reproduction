#!/bin/bash
#SBATCH --job-name=oc_lr_vf_grid
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
# Array size must equal len(ENT_COEFS)*len(CLIP_EPS_RANGE)*len(MAX_GRAD_NORM_RANGE) - 1 (e.g. 3*3*3=27 -> 0-26)
#SBATCH --array=0-26
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/lr_vf_grid_%A_%a.out
#SBATCH --error=logs/lr_vf_grid_%A_%a.err

# Grid: ENT_COEF x CLIP_EPS_RANGE x MAX_GRAD_NORM_RANGE. Fixed: LR, VF_COEF. Layout: simple/random0.

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -d "${REPO_ROOT}/scripts" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/../scripts" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  echo "ERROR: Could not resolve REPO_ROOT. Set REPO_ROOT or submit from human_aware_rl_jax_lift/slurm with --chdir=\$PWD."
  exit 1
fi

# Fixed hyperparams
LR="${LR:-8.0e-4}"
VF_COEF="${VF_COEF:-0.05}"
LAYOUT="${LAYOUT:-simple}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-7000000}"
PPO_SP_SEEDS=(2229 7649 7225 9807 386)

# Ranges: one value per dimension (array index flattens ENT x CLIP x MGN)
ENT_COEFS=(0.01 0.05 0.1 0.5)
CLIP_EPS_RANGE=(0.05)
MAX_GRAD_NORM_RANGE=(0.1)

N_ENT=${#ENT_COEFS[@]}
N_CLIP=${#CLIP_EPS_RANGE[@]}
N_MGN=${#MAX_GRAD_NORM_RANGE[@]}
N_TOTAL=$((N_ENT * N_CLIP * N_MGN))

ID=${SLURM_ARRAY_TASK_ID}
ent_idx=$((ID / (N_CLIP * N_MGN)))
rem=$((ID % (N_CLIP * N_MGN)))
clip_idx=$((rem / N_MGN))
mgn_idx=$((rem % N_MGN))

ENT_COEF="${ENT_COEFS[$ent_idx]}"
CLIP_EPS="${CLIP_EPS_RANGE[$clip_idx]}"
MAX_GRAD_NORM="${MAX_GRAD_NORM_RANGE[$mgn_idx]}"

# Unique save dir per (ent, clip_eps, max_grad_norm)
RUN_TAG="ent${ENT_COEF}_clip${CLIP_EPS}_mgn${MAX_GRAD_NORM}"
SAVE_DIR="${REPO_ROOT}/data/ppo_runs_lr_vf_simple/${RUN_TAG}"

echo "Task=$SLURM_ARRAY_TASK_ID/$N_TOTAL -> LR=$LR VF_COEF=$VF_COEF ENT_COEF=$ENT_COEF clip_eps=$CLIP_EPS max_grad_norm=$MAX_GRAD_NORM layout=$LAYOUT"
echo "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS SAVE_DIR=$SAVE_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "Node: $(hostname) | Start: $(date)"

module load cuda/12.1 2>/dev/null || true
module load miniforge
conda activate overcooked

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/home/tchen22/.cache/jax_compile_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
python -c "
import jax
devs = jax.devices()
print('JAX devices:', devs)
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), \
    f'No GPU found by JAX. Devices: {devs}'
print('GPU check passed.')
" || exit 1

cd "${REPO_ROOT}"
python -u "${REPO_ROOT}/scripts/train_ppo_sp.py" \
  --layout         "$LAYOUT" \
  --seeds          "${PPO_SP_SEEDS[@]}" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --learning_rate  "$LR" \
  --max_grad_norm  "$MAX_GRAD_NORM" \
  --randomize_agent_idx \
  --clip_eps       "$CLIP_EPS" \
  --vf_coef        "$VF_COEF" \
  --ent_coef       "$ENT_COEF" \
  --save_dir       "$SAVE_DIR" \
  --jax

echo "Done: $(date)"
