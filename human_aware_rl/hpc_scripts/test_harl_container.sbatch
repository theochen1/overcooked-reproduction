#!/bin/bash
#SBATCH --job-name=harl_container_smoke
#SBATCH --output=logs/harl_container_smoke_%j.out
#SBATCH --error=logs/harl_container_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

mkdir -p logs

# Override at submit time:
#   sbatch --export=ALL,SIF_PATH=/path/to/harl_tf1.sif hpc_scripts/test_harl_container.sbatch
# Optional source override (bind mount host repo into container):
#   sbatch --export=ALL,SIF_PATH=/path/to/harl_tf1.sif,HOST_REPO_ROOT=/path/to/repo_root hpc_scripts/test_harl_container.sbatch
# Optional module load command for clusters that hide apptainer in batch jobs:
#   sbatch --export=ALL,SIF_PATH=/path/to/harl_tf1.sif,APPTAINER_MODULE_CMD='module load apptainer' hpc_scripts/test_harl_container.sbatch
SIF_PATH="${SIF_PATH:-$HOME/containers/harl_tf1.sif}"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-}"
APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"

if [[ -n "${APPTAINER_MODULE_CMD}" ]]; then
  eval "${APPTAINER_MODULE_CMD}"
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH."
  echo "Set APPTAINER_MODULE_CMD, e.g.:"
  echo "  sbatch --export=ALL,SIF_PATH=${SIF_PATH},APPTAINER_MODULE_CMD='module load apptainer' hpc_scripts/test_harl_container.sbatch"
  exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "ERROR: SIF not found: ${SIF_PATH}"
  exit 1
fi

echo "==============================================================="
echo "HARL container smoke test"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "SIF: ${SIF_PATH}"
if [[ -n "${HOST_REPO_ROOT}" ]]; then
  echo "Host repo root (bind): ${HOST_REPO_ROOT}"
else
  echo "Host repo root (bind): <none, using image-embedded source>"
fi
echo "Start: $(date)"
echo "==============================================================="

APPTAINER_BIND_ARGS=()
if [[ -n "${HOST_REPO_ROOT}" ]]; then
  APPTAINER_BIND_ARGS+=(--bind "${HOST_REPO_ROOT}:/opt/human_aware_rl")
fi

apptainer exec \
  "${APPTAINER_BIND_ARGS[@]}" \
  "${SIF_PATH}" \
  bash -lc "
    set -euo pipefail
    source /opt/conda/etc/profile.d/conda.sh
    conda activate harl
    cd /opt/human_aware_rl/human_aware_rl
    export PYTHONPATH=/opt/human_aware_rl:/opt/human_aware_rl/overcooked_ai:\${PYTHONPATH:-}
    python -c 'import tensorflow as tf; print(\"tf\", tf.__version__)'
    python -c 'import baselines; print(\"baselines import ok\")'
    python -c 'import stable_baselines; print(\"stable_baselines import ok\")'
    python -c 'import overcooked_ai_py; print(\"overcooked_ai_py import ok\")'
  "

echo "==============================================================="
echo "Smoke test PASSED"
echo "End: $(date)"
echo "==============================================================="
