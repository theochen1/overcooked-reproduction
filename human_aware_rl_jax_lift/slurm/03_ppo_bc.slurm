#!/bin/bash
#SBATCH --job-name=oc_ppo_bc
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-9%2
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=logs/ppo_bc_%A_%a.out
#SBATCH --error=logs/ppo_bc_%A_%a.err

# NOTE: The public mit_normal_gpu QOS enforces per-user CPU limits.
# This array requests 16 CPUs per task; the "%2" throttles concurrency to
# keep total CPUs <= 32 across the array (avoids QOSMaxCpuPerUserLimit).

LAYOUTS=(simple unident_s random0 random1 random3)
SEEDS_TRAIN=(9456 1887 5578 5987 516)
SEEDS_TEST=(2888 7424 7360 4467 184)

IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt 5 ]; then
    LAYOUT=${LAYOUTS[$IDX]}
    SPLIT=train
    SEEDS=("${SEEDS_TRAIN[@]}")
else
    LAYOUT=${LAYOUTS[$((IDX-5))]}
    SPLIT=test
    SEEDS=("${SEEDS_TEST[@]}")
fi

echo "Task=$IDX layout=$LAYOUT split=$SPLIT"
echo "Node: $(hostname) | Start: $(date)"

module load cuda/12.1 2>/dev/null || true
module load miniforge
conda activate overcooked

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# Optional: reduce XLA autotuner warning spam (dot_search_space fallback); harmless to leave unset
# export TF_CPP_MIN_LOG_LEVEL=2

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

BC_PATHS_FILE="data/bc_runs/best_bc_model_paths.pkl"
if [ ! -f "$BC_PATHS_FILE" ]; then
    echo "ERROR: $BC_PATHS_FILE missing — BC stage must complete first" && exit 1
fi

python -u scripts/train_ppo_bc.py \
    --layout        "$LAYOUT" \
    --bc_split      "$SPLIT" \
    --seeds         "${SEEDS[@]}" \
    --bc_paths_file "$BC_PATHS_FILE" \
    --save_dir      data/ppo_runs

echo "Done: $(date)"
