#!/bin/bash
#SBATCH --job-name=ppo2_identity
#SBATCH --output=logs/ppo2_identity_%j.out
#SBATCH --error=logs/ppo2_identity_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=mit_normal

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/config.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh"
else
    DEFAULT_HPC_CONFIG="${SCRIPT_DIR}/config.sh"
fi
source "${HPC_CONFIG:-${DEFAULT_HPC_CONFIG}}"

log_start
mkdir -p "${PROJECT_ROOT}/parity_artifacts"

OUT_JSON="${PROJECT_ROOT}/parity_artifacts/baselines_identity_${SLURM_JOB_ID}.json"
python -m human_aware_rl.jaxmarl.baselines_identity_checks --out-json "${OUT_JSON}"

EXIT_CODE=$?
echo "Identity report: ${OUT_JSON}"
log_end "${EXIT_CODE}"
exit "${EXIT_CODE}"
