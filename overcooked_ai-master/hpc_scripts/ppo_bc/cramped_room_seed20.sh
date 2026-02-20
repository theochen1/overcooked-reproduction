#!/bin/bash
#SBATCH --job-name=ppo_bc_cramped_room_s20
#SBATCH --output=../logs/ppo_bc_cramped_room_seed20_%j.out
#SBATCH --error=../logs/ppo_bc_cramped_room_seed20_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# PPO with BC Partner Training: cramped_room (seed=20)

# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
source "${HPC_CONFIG:-$HOME/home/overcooked_ai/hpc_scripts/config.sh}"

log_start

echo "Layout: cramped_room -> legacy version"
echo "Seed: 20"
echo "Training PPO with BC partner..."
echo ""

python -m human_aware_rl.ppo.train_ppo_bc \
    --layout cramped_room \
    --seed 20 \
    --results_dir "${RESULTS_DIR}/ppo_bc"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
