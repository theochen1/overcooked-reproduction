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
#   - TF metrics json
TF_DUMP_CMD="${TF_DUMP_CMD:-python ${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl/ppo/debug_tf_dump_minibatch.py --out_dir ${CONT_PARITY_DIR}}"

# Command that loads TF dump and writes parity metrics
JAX_PARITY_CMD="${JAX_PARITY_CMD:-python ${CONT_REPO_ROOT}/overcooked_ai-master/src/human_aware_rl/jaxmarl/parity_check.py --in_dir ${CONT_PARITY_DIR} --out_json ${CONT_PARITY_DIR}/jax_metrics.json}"

# Optional: true JAX model update comparison on the same dumped minibatch.
# Requires jax/flax/optax available in the runtime environment.
RUN_JAX_COMPARE="${RUN_JAX_COMPARE:-0}"
JAX_COMPARE_CMD="${JAX_COMPARE_CMD:-python ${CONT_REPO_ROOT}/overcooked_ai-master/src/human_aware_rl/jaxmarl/minibatch_tf_vs_jax.py --in_dir ${CONT_PARITY_DIR} --out_json ${CONT_PARITY_DIR}/jax_tf_minibatch_compare.json}"

############################
# END USER CONFIG
############################

if [[ -n "${APPTAINER_MODULE_CMD}" ]]; then
  eval "${APPTAINER_MODULE_CMD}"
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH."
  echo "Set APPTAINER_MODULE_CMD, e.g.:"
  echo "  sbatch --export=ALL,SIF_PATH=${SIF_PATH},APPTAINER_MODULE_CMD='module load apptainer',TF_DUMP_CMD='...',JAX_PARITY_CMD='...' human_aware_rl/hpc_scripts/slurm_parity_apptainer.sbatch"
  exit 1
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
# Use a single consistent import order to avoid mixing legacy and jax package trees.
export PYTHONPATH=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl:${CONT_REPO_ROOT}/human_aware_rl/overcooked_ai:${CONT_REPO_ROOT}/human_aware_rl:\${PYTHONPATH:-}
cd ${CONT_REPO_ROOT}
echo '[Step 1/2] TF dump...'
${TF_DUMP_CMD}
echo '[Step 2/2] JAX parity...'
${JAX_PARITY_CMD}
if [ \"${RUN_JAX_COMPARE}\" = \"1\" ]; then
  echo '[Step 3/3] JAX model update comparison...'
  if python - <<'PY'
import importlib.util
ok = all(importlib.util.find_spec(m) is not None for m in ("jax", "flax", "optax"))
raise SystemExit(0 if ok else 1)
PY
  then
    ${JAX_COMPARE_CMD}
  else
    echo 'Skipping JAX model update comparison: jax/flax/optax not installed in this environment.'
    echo 'Set RUN_JAX_COMPARE=0 (default) or run Step 3 in a JAX-capable container.'
  fi
fi
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

