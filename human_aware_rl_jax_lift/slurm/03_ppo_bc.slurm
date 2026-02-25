#!/bin/bash
#SBATCH --job-name=oc_ppo_bc
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=05:30:00
#SBATCH --output=logs/ppo_bc_%A_%a.out
#SBATCH --error=logs/ppo_bc_%A_%a.err

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
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Node: $(hostname) | Start: $(date)"

module load anaconda/2023b 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
source activate overcooked

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

BC_PATHS_FILE="data/bc_runs/best_bc_model_paths.pkl"
if [ ! -f "$BC_PATHS_FILE" ]; then
    echo "ERROR: $BC_PATHS_FILE missing — BC stage must complete first" && exit 1
fi

python scripts/train_ppo_bc.py \
    --layout        "$LAYOUT" \
    --bc_split      "$SPLIT" \
    --seeds         "${SEEDS[@]}" \
    --bc_paths_file "$BC_PATHS_FILE" \
    --save_dir      data/ppo_runs

echo "Done: $(date)"
