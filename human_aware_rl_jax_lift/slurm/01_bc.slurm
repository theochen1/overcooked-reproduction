#!/bin/bash
#SBATCH --job-name=oc_bc
#SBATCH --account=mit_general
#SBATCH --qos=mit_preemptable
#SBATCH --partition=mit_normal
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bc_%A_%a.out
#SBATCH --error=logs/bc_%A_%a.err

# Array layout: 0-4 = train split, 5-9 = test split
LAYOUTS=(simple unident_s random0 random1 random3)
BC_SEEDS=(5415 2652 6440 1965 6647)

IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt 5 ]; then
    LAYOUT=${LAYOUTS[$IDX]}
    SPLIT=train
else
    LAYOUT=${LAYOUTS[$((IDX-5))]}
    SPLIT=test
fi

echo "Task=$IDX layout=$LAYOUT split=$SPLIT | $(hostname) | $(date)"

module load anaconda/2023b 2>/dev/null || true
source activate overcooked
export JAX_PLATFORM_NAME=cpu

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

python scripts/train_bc.py \
    --layout     "$LAYOUT" \
    --data_path  "data/bc_data/${LAYOUT}.pkl" \
    --split      "$SPLIT" \
    --seeds      "${BC_SEEDS[@]}" \
    --epochs     100 \
    --lr         1e-3 \
    --batch_size 64 \
    --adam_eps   1e-8 \
    --save_dir   data/bc_runs

echo "Done: $(date)"
