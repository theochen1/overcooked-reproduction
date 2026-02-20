#!/bin/bash
#SBATCH --job-name=overcooked_fast
#SBATCH --time=47:00:00
#SBATCH --mem=32G
#SBATCH -n 16

# Fast training version (~2-4 hours total instead of 48+)
# Uses 1M timesteps and early stopping

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
echo "Step 2: Training PPO Self-Play (FAST)"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0 --fast

echo "========================================="
echo "Step 3: Training PPO with BC partner (FAST)"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0 --fast

echo "========================================="
echo "All training complete!"
echo "========================================="

