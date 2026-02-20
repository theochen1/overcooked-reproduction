#!/bin/bash
#SBATCH --job-name=bc_train
#SBATCH --time=47:00:00
#SBATCH --mem=32G
#SBATCH -n 16

# Create logs directory
mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to project
cd $SLURM_SUBMIT_DIR/src/human_aware_rl

# Train BC models for all layouts
python -m human_aware_rl.imitation.train_bc_models --all_layouts

echo "BC training complete!"

