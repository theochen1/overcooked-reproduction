#!/bin/bash
#SBATCH --job-name=build_jax_minibatch_sif
#SBATCH --output=logs/build_jax_minibatch_sif_%j.out
#SBATCH --error=logs/build_jax_minibatch_sif_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

mkdir -p logs

APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"
OUT_SIF="${OUT_SIF:-$HOME/containers/jax_minibatch.sif}"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$PWD}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH_ROOT/apptainer-tmp}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_ROOT/apptainer-cache}"

if [[ -n "${APPTAINER_MODULE_CMD}" ]]; then
  eval "${APPTAINER_MODULE_CMD}"
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH."
  echo "Set APPTAINER_MODULE_CMD, e.g.:"
  echo "  sbatch --export=ALL,APPTAINER_MODULE_CMD='module load apptainer' human_aware_rl/hpc_scripts/build_jax_minibatch_sif.sbatch"
  exit 1
fi

if [[ ! -d "${HOST_REPO_ROOT}/human_aware_rl/hpc_scripts" ]]; then
  echo "ERROR: HOST_REPO_ROOT does not look like repo root: ${HOST_REPO_ROOT}"
  exit 1
fi

mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}" "$(dirname "${OUT_SIF}")"
export APPTAINER_TMPDIR
export APPTAINER_CACHEDIR

echo "==============================================================="
echo "Build JAX Minibatch SIF via Slurm"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Repo root: ${HOST_REPO_ROOT}"
echo "Output SIF: ${OUT_SIF}"
echo "APPTAINER_TMPDIR: ${APPTAINER_TMPDIR}"
echo "APPTAINER_CACHEDIR: ${APPTAINER_CACHEDIR}"
echo "Start: $(date)"
echo "==============================================================="

bash "${HOST_REPO_ROOT}/human_aware_rl/hpc_scripts/build_jax_minibatch_sif.sh" "${OUT_SIF}"

echo "==============================================================="
echo "Build complete: ${OUT_SIF}"
echo "End: $(date)"
echo "==============================================================="
