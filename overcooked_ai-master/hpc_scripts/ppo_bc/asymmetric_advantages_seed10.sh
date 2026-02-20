#!/bin/bash
#SBATCH --job-name=ppo_bc_asymmetric_advantages_s10
#SBATCH --output=../logs/ppo_bc_asymmetric_advantages_seed10_%j.out
#SBATCH --error=../logs/ppo_bc_asymmetric_advantages_seed10_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# PPO with BC Partner Training: asymmetric_advantages (seed=10)

# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
source "${HPC_CONFIG:-$HOME/home/overcooked_ai/hpc_scripts/config.sh}"

log_start

echo "Layout: asymmetric_advantages -> legacy version"
echo "Seed: 10"
echo "Training PPO with BC partner..."
echo ""

python -m human_aware_rl.ppo.train_ppo_bc \
    --layout asymmetric_advantages \
    --seed 10 \
    --results_dir "${RESULTS_DIR}/ppo_bc"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
