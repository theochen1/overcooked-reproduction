#!/bin/bash
#SBATCH --job-name=harl_ground_truth
#SBATCH --output=logs/harl_ground_truth_%j.out
#SBATCH --error=logs/harl_ground_truth_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs

echo "==============================================================="
echo "Ground-Truth TF PPO Job"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "==============================================================="

# Path to TF container image.
SIF_PATH="${SIF_PATH:-/home/tchen22/containers/harl_tf1.sif}"

# Host repo root and output directory.
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$PWD}"
HOST_GT_DIR="${HOST_GT_DIR:-$PWD/ground_truth_runs}"

# Container mount points.
CONT_REPO_ROOT="/opt/workdir"
CONT_GT_DIR="/opt/ground_truth"

# Module load command for clusters requiring environment modules.
APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"

# PPO run knobs (all overridable via --export).
RUN_NAME="${RUN_NAME:-tf_gt_random0_sp_seed0}"
LAYOUT="${LAYOUT:-random0}"
OTHER_AGENT_TYPE="${OTHER_AGENT_TYPE:-sp}"
SEEDS_JSON="${SEEDS_JSON:-[0]}"
PPO_RUN_TOT_TIMESTEPS="${PPO_RUN_TOT_TIMESTEPS:-5000000}"
GPU_ID="${GPU_ID:-0}"
# Additional Sacred CLI overrides appended verbatim.
EXTRA_SACRED_ARGS="${EXTRA_SACRED_ARGS:-}"

# Initialize environment modules in non-interactive Slurm shells.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck source=/dev/null
    source /usr/share/lmod/lmod/init/bash
  fi
fi

if [[ -n "${APPTAINER_MODULE_CMD}" ]]; then
  eval "${APPTAINER_MODULE_CMD}"
fi

# Fallback: try common cluster module names if command still missing.
if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer/1.4.2 >/dev/null 2>&1 || module load apptainer >/dev/null 2>&1 || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH."
  echo "Set APPTAINER_MODULE_CMD, e.g. APPTAINER_MODULE_CMD='module load apptainer'"
  exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "ERROR: SIF image not found at ${SIF_PATH}"
  exit 1
fi

mkdir -p "${HOST_GT_DIR}"

echo "SIF_PATH=${SIF_PATH}"
echo "HOST_REPO_ROOT=${HOST_REPO_ROOT}"
echo "HOST_GT_DIR=${HOST_GT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "LAYOUT=${LAYOUT}"
echo "OTHER_AGENT_TYPE=${OTHER_AGENT_TYPE}"
echo "SEEDS_JSON=${SEEDS_JSON}"
echo "PPO_RUN_TOT_TIMESTEPS=${PPO_RUN_TOT_TIMESTEPS}"
echo "GPU_ID=${GPU_ID}"
echo "EXTRA_SACRED_ARGS=${EXTRA_SACRED_ARGS}"
echo ""

CONTAINER_BOOTSTRAP="
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate harl
export PYTHONNOUSERSITE=1
# Resolve repository layout under the bind mount.
# Supported forms:
#  A) <root>/human_aware_rl/ppo/ppo.py + <root>/overcooked_ai/...
#  B) <root>/human_aware_rl/human_aware_rl/ppo/ppo.py + <root>/human_aware_rl/overcooked_ai/...
if [[ -f ${CONT_REPO_ROOT}/human_aware_rl/ppo/ppo.py && -d ${CONT_REPO_ROOT}/overcooked_ai ]]; then
  PPO_ENTRY=${CONT_REPO_ROOT}/human_aware_rl/ppo/ppo.py
  BC_RUNS_DIR=${CONT_REPO_ROOT}/human_aware_rl/data/bc_runs
  export PYTHONPATH=${CONT_REPO_ROOT}/human_aware_rl:${CONT_REPO_ROOT}/overcooked_ai:${CONT_REPO_ROOT}
elif [[ -f ${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl/ppo/ppo.py && -d ${CONT_REPO_ROOT}/human_aware_rl/overcooked_ai ]]; then
  PPO_ENTRY=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl/ppo/ppo.py
  BC_RUNS_DIR=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl/data/bc_runs
  export PYTHONPATH=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl:${CONT_REPO_ROOT}/human_aware_rl/overcooked_ai:${CONT_REPO_ROOT}/human_aware_rl
else
  echo 'ERROR: Could not resolve ppo.py / overcooked_ai layout under bind mount.'
  echo 'CONT_REPO_ROOT='${CONT_REPO_ROOT}
  ls -la ${CONT_REPO_ROOT} || true
  exit 1
fi
mkdir -p ${CONT_GT_DIR}/${RUN_NAME}
RUN_WORKDIR=${CONT_GT_DIR}/${RUN_NAME}/workdir
mkdir -p \${RUN_WORKDIR}
cd \${RUN_WORKDIR}
# Isolate per-job Sacred outputs (avoids parallel FileExistsError on data/ppo_runs/ppo_exp)
mkdir -p data/ppo_runs
# Provide shared BC artifacts to per-job working dirs.
ln -sfn \${BC_RUNS_DIR} data/bc_runs
python \${PPO_ENTRY} with \
  params.TIMESTAMP_DIR=False \
  params.SAVE_DIR='${CONT_GT_DIR}/${RUN_NAME}/' \
  params.EX_NAME='${RUN_NAME}' \
  params.mdp_params.layout_name='${LAYOUT}' \
  params.OTHER_AGENT_TYPE='${OTHER_AGENT_TYPE}' \
  params.SEEDS='${SEEDS_JSON}' \
  params.PPO_RUN_TOT_TIMESTEPS=${PPO_RUN_TOT_TIMESTEPS} \
  params.GPU_ID=${GPU_ID} \
  ${EXTRA_SACRED_ARGS}
echo 'Done. Ground-truth outputs in ${CONT_GT_DIR}/${RUN_NAME}/'
"

apptainer exec \
  --bind "${HOST_REPO_ROOT}:${CONT_REPO_ROOT}" \
  --bind "${HOST_GT_DIR}:${CONT_GT_DIR}" \
  "${SIF_PATH}" \
  bash -lc "${CONTAINER_BOOTSTRAP}"

echo "==============================================================="
echo "End: $(date)"
echo "Ground-truth outputs: ${HOST_GT_DIR}/${RUN_NAME}/"
echo "==============================================================="
