#!/bin/bash
#SBATCH --job-name=oc_pbt
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-24
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/pbt_%x_%A_%a.out
#SBATCH --error=logs/pbt_%x_%A_%a.err

# AGENT_TYPE injected by submit_all.sh via --export=ALL,AGENT_TYPE=...
AGENT_TYPE=${AGENT_TYPE:-sp}

# Decode task ID -> layout + seed
LAYOUTS=(simple unident_s random0 random1 random3)
PBT_SEEDS=(8015 3554 581 5608 4221)

LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))
LAYOUT=${LAYOUTS[$LAYOUT_IDX]}
SEED=${PBT_SEEDS[$SEED_IDX]}

echo "Task=$SLURM_ARRAY_TASK_ID layout=$LAYOUT seed=$SEED agent_type=$AGENT_TYPE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Node: $(hostname) | Start: $(date)"

module load anaconda/2023b 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
source activate overcooked

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

EXTRA_ARGS=""
if [ "$AGENT_TYPE" != "sp" ]; then
    BC_PATHS_FILE="data/bc_runs/best_bc_model_paths.pkl"
    if [ ! -f "$BC_PATHS_FILE" ]; then
        echo "ERROR: $BC_PATHS_FILE missing" && exit 1
    fi
    EXTRA_ARGS="--bc_paths_file $BC_PATHS_FILE"
fi

python scripts/train_pbt.py \
    --layout             "$LAYOUT" \
    --seeds              "$SEED" \
    --other_agent_type   "$AGENT_TYPE" \
    --save_dir           "data/pbt_runs" \
    $EXTRA_ARGS

echo "Done: $(date)"
