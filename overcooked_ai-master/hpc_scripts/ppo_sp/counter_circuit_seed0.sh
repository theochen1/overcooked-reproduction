#!/bin/bash
#SBATCH --job-name=ppo_sp_counter_circuit_s0
#SBATCH --output=../logs/ppo_sp_counter_circuit_seed0_%j.out
#SBATCH --error=../logs/ppo_sp_counter_circuit_seed0_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# PPO Self-Play Training: counter_circuit (seed=0)

# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
source "${HPC_CONFIG:-$HOME/home/overcooked_ai/hpc_scripts/config.sh}"

log_start

echo "Layout: counter_circuit"
echo "Seed: 0"
echo "Training PPO Self-Play..."
echo ""

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout counter_circuit \
    --seed 0 \
    --results_dir "${RESULTS_DIR}/ppo_sp"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
