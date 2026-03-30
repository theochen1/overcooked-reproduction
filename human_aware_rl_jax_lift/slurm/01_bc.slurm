#!/bin/bash
#SBATCH --job-name=oc_bc
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bc_%A_%a.out
#SBATCH --error=logs/bc_%A_%a.err

# Array layout: 0-4 = train split, 5-9 = test split
# Epochs per layout (paper Appendix A): simple=100, unident_s=120, random0=90, random1=120, random3=110
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

module load miniforge
conda activate overcooked
export JAX_PLATFORMS=cpu

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

# Per-layout seed overrides to match TF manual selection (TF bc_experiments.py lines 114-120).
# TF noted "Automatic selection caused imbalances that made interpretation more difficult,
# better to select manually non-best ones."  All layouts use explicit indices.
#   simple:    train=0  test=1
#   unident_s: train=0  test=0
#   random1:   train=4  test=2
#   random0:   train=2  test=1
#   random3:   train=3  test=3
declare -A TRAIN_IDX=([simple]=0 [unident_s]=0 [random1]=4 [random0]=2 [random3]=3)
declare -A TEST_IDX=( [simple]=1 [unident_s]=0 [random1]=2 [random0]=1 [random3]=3)

if [ "$SPLIT" = "train" ]; then
    SEED_OVERRIDE_ARGS="--seed_idx_override ${TRAIN_IDX[$LAYOUT]}"
else
    SEED_OVERRIDE_ARGS="--seed_idx_override ${TEST_IDX[$LAYOUT]}"
fi

python scripts/train_bc.py \
    --layout     "$LAYOUT" \
    --data_path  "data/bc_data/${LAYOUT}.pkl" \
    --split      "$SPLIT" \
    --seeds      "${BC_SEEDS[@]}" \
    --epochs     "$EPOCH" \
    --lr         1e-3 \
    --batch_size 64 \
    --adam_eps   1e-8 \
    --save_dir   data/bc_runs \
    --model_selection final \
    $SEED_OVERRIDE_ARGS

echo "Done: $(date)"
