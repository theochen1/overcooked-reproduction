#!/bin/bash
#SBATCH --job-name=gail_counter_circuit
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# Train GAIL model for counter_circuit layout

# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
source "${HPC_CONFIG:-$HOME/home/overcooked_ai/hpc_scripts/config.sh}"

log_start

python -m human_aware_rl.imitation.gail \
    --layout counter_circuit

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
