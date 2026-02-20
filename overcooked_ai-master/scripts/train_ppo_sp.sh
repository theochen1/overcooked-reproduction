#!/bin/bash
#SBATCH --job-name=ppo_sp
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

# Train PPO Self-Play for all layouts with 5 seeds
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40

echo "PPO Self-Play training complete!"

