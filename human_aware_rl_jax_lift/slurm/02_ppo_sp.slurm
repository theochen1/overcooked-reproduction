#!/bin/bash
#SBATCH --job-name=oc_ppo_sp
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-4
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ppo_sp_%A_%a.out
#SBATCH --error=logs/ppo_sp_%A_%a.err

LAYOUTS=(simple unident_s random0 random1 random3)
PPO_SP_SEEDS=(2229 7649 7225 9807 386)

LAYOUT=${LAYOUTS[$SLURM_ARRAY_TASK_ID]}
echo "Task=$SLURM_ARRAY_TASK_ID layout=$LAYOUT"
echo "Node: $(hostname) | Start: $(date)"

module load cuda/12.1 2>/dev/null || true
conda activate overcooked

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# Fail fast if JAX cannot see the GPU
python -c "
import jax
devs = jax.devices()
print('JAX devices:', devs)
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), \
    f'No GPU found by JAX — check jax[cuda12] install. Devices: {devs}'
print('GPU check passed.')
" || exit 1

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

python scripts/train_ppo_sp.py \
    --layout    "$LAYOUT" \
    --seeds     "${PPO_SP_SEEDS[@]}" \
    --save_dir  data/ppo_runs

echo "Done: $(date)"
