#!/bin/bash
#SBATCH --job-name=ppo_gail_cramped_room_s30
#SBATCH --output=../logs/ppo_gail_cramped_room_seed30_%j.out
#SBATCH --error=../logs/ppo_gail_cramped_room_seed30_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# PPO with GAIL Partner Training: cramped_room (seed=30)

# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
source "${HPC_CONFIG:-$HOME/home/overcooked_ai/hpc_scripts/config.sh}"

log_start

echo "Layout: cramped_room -> legacy version"
echo "Seed: 30"
echo "Training PPO with GAIL partner..."
echo ""

python -m human_aware_rl.ppo.train_ppo_gail \
    --layout cramped_room \
    --seed 30 \
    --results_dir "${RESULTS_DIR}/ppo_gail"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
