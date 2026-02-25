#!/bin/bash
#SBATCH --job-name=oc_bc_fig6
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:40:00
#SBATCH --output=logs/bc_fig6_%A_%a.out
#SBATCH --error=logs/bc_fig6_%A_%a.err

# Layout-specific epoch counts from experiments/figure6.py
LAYOUTS=(simple unident_s random0 random1 random3)
EPOCHS=(100    120       90      120     110)
BC_SEEDS=(5415 2652 6440 1965 6647)

IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt 5 ]; then
    LAYOUT=${LAYOUTS[$IDX]}
    SPLIT=train
    EPOCH=${EPOCHS[$IDX]}
else
    LAYOUT=${LAYOUTS[$((IDX-5))]}
    SPLIT=test
    EPOCH=${EPOCHS[$((IDX-5))]}
fi

echo "Task=$IDX layout=$LAYOUT split=$SPLIT epochs=$EPOCH | $(hostname) | $(date)"

module load anaconda/2023b 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate overcooked
export JAX_PLATFORM_NAME=cpu

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

python scripts/train_bc.py \
    --layout     "$LAYOUT" \
    --data_path  "data/bc_data/${LAYOUT}.pkl" \
    --split      "$SPLIT" \
    --seeds      "${BC_SEEDS[@]}" \
    --epochs     "$EPOCH" \
    --lr         1e-3 \
    --batch_size 64 \
    --adam_eps   1e-8 \
    --save_dir   data/bc_runs_fig6

echo "Done: $(date)"
