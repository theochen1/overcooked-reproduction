#!/bin/bash
#SBATCH --job-name=overcooked_parity
#SBATCH --output=logs/parity_%j.out
#SBATCH --error=logs/parity_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Same-minibatch TF vs JAX parity template (HPC + Apptainer)
#
# Usage:
#   1) Build/pull your image to a .sif file first.
#   2) Edit variables in the "USER CONFIG" section.
#   3) sbatch human_aware_rl/hpc_scripts/slurm_parity_apptainer.sbatch
#
# Notes:
# - This script runs two commands inside the same container:
#   (a) TF dump step (minibatch + params + TF metrics)
#   (b) JAX parity step (load dump, compute JAX metrics, compare)
# - Replace TF_DUMP_CMD and JAX_PARITY_CMD with your concrete scripts/modules.

set -euo pipefail

mkdir -p logs

echo "==============================================================="
echo "Same-Minibatch Parity Job"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "==============================================================="

############################
# USER CONFIG
############################

# Path to your Apptainer image (.sif)
SIF_PATH="${SIF_PATH:-/path/to/harl.sif}"

# Host repo root (directory that contains human_aware_rl and/or overcooked_ai-master)
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$PWD}"

# Host output directory for parity artifacts
HOST_PARITY_DIR="${HOST_PARITY_DIR:-$PWD/parity_artifacts}"

# Container mount points
CONT_REPO_ROOT="/opt/workdir"
CONT_PARITY_DIR="/opt/parity"

# Set to "module load apptainer" if your cluster requires modules.
APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"

# Command that runs TF and writes:
#   - minibatch dump (obs/actions/old_log_probs/returns/old_values/advs)
#   - TF params dump
#   - TF metrics json
TF_DUMP_CMD="${TF_DUMP_CMD:-python -m human_aware_rl.ppo.debug_tf_dump_minibatch --out_dir ${CONT_PARITY_DIR}}"

# Command that loads TF dump + params and writes JAX parity metrics
JAX_PARITY_CMD="${JAX_PARITY_CMD:-python -m human_aware_rl.jaxmarl.parity_check --in_dir ${CONT_PARITY_DIR} --out_json ${CONT_PARITY_DIR}/jax_metrics.json}"

############################
# END USER CONFIG
############################

if [[ -n "${APPTAINER_MODULE_CMD}" ]]; then
  eval "${APPTAINER_MODULE_CMD}"
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "ERROR: SIF image not found at ${SIF_PATH}"
  exit 1
fi

mkdir -p "${HOST_PARITY_DIR}"

echo "SIF_PATH=${SIF_PATH}"
echo "HOST_REPO_ROOT=${HOST_REPO_ROOT}"
echo "HOST_PARITY_DIR=${HOST_PARITY_DIR}"
echo ""

CONTAINER_BOOTSTRAP="
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate harl
export PYTHONPATH=${CONT_REPO_ROOT}/human_aware_rl:${CONT_REPO_ROOT}/human_aware_rl/overcooked_ai:${CONT_REPO_ROOT}/overcooked_ai-master/src:\${PYTHONPATH:-}
cd ${CONT_REPO_ROOT}
echo '[Step 1/2] TF dump...'
${TF_DUMP_CMD}
echo '[Step 2/2] JAX parity...'
${JAX_PARITY_CMD}
echo 'Done. Artifacts in ${CONT_PARITY_DIR}'
"

apptainer exec \
  --bind "${HOST_REPO_ROOT}:${CONT_REPO_ROOT}" \
  --bind "${HOST_PARITY_DIR}:${CONT_PARITY_DIR}" \
  "${SIF_PATH}" \
  bash -lc "${CONTAINER_BOOTSTRAP}"

echo "==============================================================="
echo "End: $(date)"
echo "Parity artifacts: ${HOST_PARITY_DIR}"
echo "==============================================================="

