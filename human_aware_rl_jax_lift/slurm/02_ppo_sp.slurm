#!/bin/bash
#SBATCH --job-name=oc_ppo_sp
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-4%2
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ppo_sp_%A_%a.out
#SBATCH --error=logs/ppo_sp_%A_%a.err

# NOTE: The public mit_normal_gpu QOS enforces per-user CPU limits.
# This array requests 16 CPUs per task; the "%2" throttles concurrency to
# keep total CPUs <= 32 across the array (avoids QOSMaxCpuPerUserLimit).

# Hyperparameters and seeds come from paper_config.yaml (loaded by train_ppo_sp.py via get_hparams).
# Layouts and seeds must match paper_config.yaml (environment.layouts and seeds.ppo_sp).
LAYOUTS=(simple unident_s random0 random1 random3)
PPO_SP_SEEDS=(2229 7649 7225 9807 386)

LAYOUT=${LAYOUTS[$SLURM_ARRAY_TASK_ID]}
echo "Task=$SLURM_ARRAY_TASK_ID layout=$LAYOUT"
echo "total_timesteps per layout from paper_config.yaml (Fig. 8)"
echo "Node: $(hostname) | Start: $(date)"

module load cuda/12.1 2>/dev/null || true
module load miniforge
conda activate overcooked

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Persistent compilation cache — avoids recompiling kernels on every job.
# After the first run, subsequent jobs on the same GPU model skip the XLA compile.
export JAX_COMPILATION_CACHE_DIR="/home/tchen22/.cache/jax_compile_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# Fail fast if GPU not visible
python -c "
import jax
devs = jax.devices()
print('Devices:', devs)
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), \
    f'No GPU found — check CUDA/backend install. Devices: {devs}'
print('GPU check passed.')
" || exit 1

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

export OVERCOOKED_LAYOUT_DIR="$(pwd)/../human_aware_rl/overcooked_ai/overcooked_ai_py/data/layouts"

python -u scripts/train_ppo_sp.py \
    --layout   "$LAYOUT" \
    --seeds    "${PPO_SP_SEEDS[@]}" \
    --save_dir data/ppo_runs

echo "Done: $(date)"
