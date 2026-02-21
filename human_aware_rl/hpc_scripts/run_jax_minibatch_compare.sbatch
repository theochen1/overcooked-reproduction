#!/bin/bash
#SBATCH --job-name=jax_minibatch_compare
#SBATCH --output=logs/jax_minibatch_compare_%j.out
#SBATCH --error=logs/jax_minibatch_compare_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

mkdir -p logs

# Required:
#   parity_artifacts/tf_minibatch_dump.npz
#   parity_artifacts/tf_metrics.json
#
# Example:
#   sbatch --export=ALL,JAX_SIF_PATH=$HOME/containers/jax_minibatch.sif,APPTAINER_MODULE_CMD='module load apptainer' \
#     human_aware_rl/hpc_scripts/run_jax_minibatch_compare.sbatch

APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"
JAX_SIF_PATH="${JAX_SIF_PATH:-$HOME/containers/jax_minibatch.sif}"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$PWD}"
HOST_PARITY_DIR="${HOST_PARITY_DIR:-$PWD/parity_artifacts}"

CONT_REPO_ROOT="/opt/workdir"
CONT_PARITY_DIR="/opt/parity"

if [[ -n "${APPTAINER_MODULE_CMD}" ]]; then
  eval "${APPTAINER_MODULE_CMD}"
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH."
  exit 1
fi

if [[ ! -f "${JAX_SIF_PATH}" ]]; then
  echo "ERROR: JAX SIF image not found: ${JAX_SIF_PATH}"
  exit 1
fi

if [[ ! -f "${HOST_PARITY_DIR}/tf_minibatch_dump.npz" ]]; then
  echo "ERROR: Missing ${HOST_PARITY_DIR}/tf_minibatch_dump.npz"
  echo "Run the TF dump parity job first."
  exit 1
fi

if [[ ! -f "${HOST_PARITY_DIR}/tf_metrics.json" ]]; then
  echo "ERROR: Missing ${HOST_PARITY_DIR}/tf_metrics.json"
  echo "Run the TF dump parity job first."
  exit 1
fi

echo "==============================================================="
echo "JAX Minibatch Compare Job"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "JAX SIF: ${JAX_SIF_PATH}"
echo "Host repo root: ${HOST_REPO_ROOT}"
echo "Host parity dir: ${HOST_PARITY_DIR}"
echo "Start: $(date)"
echo "==============================================================="

apptainer exec \
  --bind "${HOST_REPO_ROOT}:${CONT_REPO_ROOT}" \
  --bind "${HOST_PARITY_DIR}:${CONT_PARITY_DIR}" \
  "${JAX_SIF_PATH}" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=${CONT_REPO_ROOT}/overcooked_ai-master/src:\${PYTHONPATH:-}
    python ${CONT_REPO_ROOT}/overcooked_ai-master/src/human_aware_rl/jaxmarl/minibatch_tf_vs_jax.py \
      --in_dir ${CONT_PARITY_DIR} \
      --out_json ${CONT_PARITY_DIR}/jax_tf_minibatch_compare.json
  "

echo "==============================================================="
echo "Done"
echo "Artifact: ${HOST_PARITY_DIR}/jax_tf_minibatch_compare.json"
echo "End: $(date)"
echo "==============================================================="
