#!/bin/bash
#SBATCH --job-name=overcooked_all
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Create logs directory
mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

# Navigate to project
cd $SLURM_SUBMIT_DIR/src/human_aware_rl

echo "========================================="
echo "Step 1: Training BC models"
echo "========================================="
python -m human_aware_rl.imitation.train_bc_models --all_layouts

echo "========================================="
echo "Step 2: Training PPO Self-Play"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40

echo "========================================="
echo "Step 3: Training PPO with BC partner"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0,10,20,30,40

echo "========================================="
echo "All training complete!"
echo "========================================="

