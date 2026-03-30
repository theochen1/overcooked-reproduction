#!/bin/bash
#SBATCH --job-name=oc_prep_bc
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_bc_%j.out
#SBATCH --error=logs/prep_bc_%j.err

# Prepares BC feature data for all layouts. Run before 01_bc.slurm.
# Reads from human_aware_rl human trial DataFrames, writes data/bc_data/{layout}.pkl.

echo "Job $SLURM_JOB_ID | $(hostname) | $(date)"

module load miniforge
conda activate overcooked
export JAX_PLATFORMS=cpu

mkdir -p "$SLURM_SUBMIT_DIR/../logs"
cd "$SLURM_SUBMIT_DIR/.."

python scripts/prepare_bc_data.py --use_path_cost

echo "Done: $(date)"
