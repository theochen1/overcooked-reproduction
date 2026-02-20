#!/bin/bash
#SBATCH --job-name=ppo_sp_cramped_room_s10
#SBATCH --output=../logs/ppo_sp_cramped_room_seed10_%j.out
#SBATCH --error=../logs/ppo_sp_cramped_room_seed10_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# PPO Self-Play Training: cramped_room (seed=10)

# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
source "${HPC_CONFIG:-$HOME/home/overcooked_ai/hpc_scripts/config.sh}"

log_start

echo "Layout: cramped_room"
echo "Seed: 10"
echo "Training PPO Self-Play..."
echo ""

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout cramped_room \
    --seed 10 \
    --results_dir "${RESULTS_DIR}/ppo_sp"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
