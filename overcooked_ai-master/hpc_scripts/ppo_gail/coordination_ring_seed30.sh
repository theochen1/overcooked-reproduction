#!/bin/bash
#SBATCH --job-name=ppo_gail_coordination_ring_s30
#SBATCH --output=../logs/ppo_gail_coordination_ring_seed30_%j.out
#SBATCH --error=../logs/ppo_gail_coordination_ring_seed30_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# PPO with GAIL Partner Training: coordination_ring (seed=30)

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

echo "Layout: coordination_ring -> legacy version"
echo "Seed: 30"
echo "Training PPO with GAIL partner..."
echo ""

python -m human_aware_rl.ppo.train_ppo_gail \
    --layout coordination_ring \
    --seed 30 \
    --results_dir "${RESULTS_DIR}/ppo_gail"

EXIT_CODE=$?

log_end $EXIT_CODE
exit $EXIT_CODE
