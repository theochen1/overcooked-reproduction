#!/bin/bash
#SBATCH --job-name=ppo_sp_coordination_ring_s10
#SBATCH --output=../logs/ppo_sp_coordination_ring_seed10_%j.out
#SBATCH --error=../logs/ppo_sp_coordination_ring_seed10_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# PPO Self-Play Training: coordination_ring (seed=10)

# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/config.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh"
else
    DEFAULT_HPC_CONFIG="$(cd "${SCRIPT_DIR}/.." && pwd)/config.sh"
fi
source "${HPC_CONFIG:-${DEFAULT_HPC_CONFIG}}"

log_start

echo "Layout: coordination_ring"
echo "Seed: 10"
echo "Training PPO Self-Play..."
echo ""

python -m human_aware_rl.ppo.train_ppo_sp \
    --layout coordination_ring \
    --seed 10 \
    --results_dir "${RESULTS_DIR}/ppo_sp"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
