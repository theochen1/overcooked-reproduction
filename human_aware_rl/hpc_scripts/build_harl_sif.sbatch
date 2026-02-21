#!/bin/bash
#SBATCH --job-name=build_harl_sif
#SBATCH --output=logs/build_harl_sif_%j.out
#SBATCH --error=logs/build_harl_sif_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

mkdir -p logs

# Optional: set to "module load apptainer" if your cluster needs modules.
APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"

# Path for built image; can be overridden at submit time.
# Example:
#   sbatch --export=ALL,OUT_SIF=$HOME/containers/harl_tf1.sif \
#     human_aware_rl/hpc_scripts/build_harl_sif.sbatch
OUT_SIF="${OUT_SIF:-$HOME/containers/harl_tf1.sif}"

# Repo root must contain human_aware_rl/hpc_scripts.
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$PWD}"

# Scratch-backed temp/cache paths to avoid home/tmp pressure.
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH_ROOT/apptainer-tmp}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_ROOT/apptainer-cache}"

if [[ -n "${APPTAINER_MODULE_CMD}" ]]; then
  eval "${APPTAINER_MODULE_CMD}"
fi

if [[ ! -d "${HOST_REPO_ROOT}/human_aware_rl/hpc_scripts" ]]; then
  echo "ERROR: HOST_REPO_ROOT does not look like the repo root:"
  echo "  ${HOST_REPO_ROOT}"
  echo "Expected directory:"
  echo "  ${HOST_REPO_ROOT}/human_aware_rl/hpc_scripts"
  exit 1
fi

mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}" "$(dirname "${OUT_SIF}")"
export APPTAINER_TMPDIR
export APPTAINER_CACHEDIR

echo "==============================================================="
echo "HARL Apptainer Build via Slurm"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Repo root: ${HOST_REPO_ROOT}"
echo "Output SIF: ${OUT_SIF}"
echo "APPTAINER_TMPDIR: ${APPTAINER_TMPDIR}"
echo "APPTAINER_CACHEDIR: ${APPTAINER_CACHEDIR}"
echo "Start: $(date)"
echo "==============================================================="

bash "${HOST_REPO_ROOT}/human_aware_rl/hpc_scripts/build_harl_sif.sh" "${OUT_SIF}"

echo "==============================================================="
echo "Build complete: ${OUT_SIF}"
echo "End: $(date)"
echo "==============================================================="
