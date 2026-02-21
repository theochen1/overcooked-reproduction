#!/bin/bash
#SBATCH --job-name=gail_asymmetric_advantages
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

# Train GAIL model for asymmetric_advantages layout

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

python -m human_aware_rl.imitation.gail \
    --layout asymmetric_advantages

EXIT_CODE=$?
log_end $EXIT_CODE
exit $EXIT_CODE
