#!/bin/bash
#SBATCH --job-name=oc_gradclip
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ablation_grad_clip_%j.out
#SBATCH --error=logs/ablation_grad_clip_%j.err

set -euo pipefail

echo "Node: $(hostname) | Start: $(date)"

module load cuda/12.1 2>/dev/null || true
module load miniforge
conda activate overcooked

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_COMPILATION_CACHE_DIR="/home/tchen22/.cache/jax_compile_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

python -c "
import jax
devs = jax.devices()
print('JAX devices:', devs)
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), \
    f'No GPU found. Devices: {devs}'
print('GPU check passed.')
" || exit 1

cd "$SLURM_SUBMIT_DIR/.."

echo ""
echo "======================================================"
echo "Ablation: JAX PPO SP on simple with max_grad_norm=10.0"
echo "Purpose: Test if aggressive gradient clipping (0.1) is"
echo "         the root cause of training divergence."
echo "======================================================"

python -u scripts/train_ppo_sp.py \
    --layout simple \
    --seeds 2229 7225 \
    --total_timesteps 3600000 \
    --save_dir data/ppo_runs_ablation_gradclip \
    --jax \
    --diagnostics \
    --max_grad_norm 10.0

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"
