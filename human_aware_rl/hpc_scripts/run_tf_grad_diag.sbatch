#!/bin/bash
#SBATCH --job-name=tf_grad_diag
#SBATCH --output=logs/tf_grad_diag_%j.out
#SBATCH --error=logs/tf_grad_diag_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

mkdir -p logs

echo "==============================================================="
echo "TF PPO Gradient Diagnostics (short run, 50 updates)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "==============================================================="

SIF_PATH="${SIF_PATH:-/home/tchen22/overcooked-reproduction/containers/harl_tf1.sif}"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$(dirname $PWD)}"
HOST_GT_DIR="${HOST_GT_DIR:-$PWD/ground_truth_runs}"

CONT_REPO_ROOT="/opt/workdir"
CONT_GT_DIR="/opt/ground_truth"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    source /usr/share/lmod/lmod/init/bash
  fi
fi

if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer/1.4.2 >/dev/null 2>&1 || module load apptainer >/dev/null 2>&1 || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH."
  exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "ERROR: SIF image not found at ${SIF_PATH}"
  exit 1
fi

mkdir -p "${HOST_GT_DIR}"

echo "SIF_PATH=${SIF_PATH}"
echo "HOST_REPO_ROOT=${HOST_REPO_ROOT}"
echo ""

CONTAINER_BOOTSTRAP="
set -eo pipefail
export PS1='\$ '
source /opt/conda/etc/profile.d/conda.sh
conda activate harl
export PYTHONNOUSERSITE=1
export TF_DELTA_DEBUG=${TF_DELTA_DEBUG:-0}
export TF_LOGIT_PROBE_DEBUG=${TF_LOGIT_PROBE_DEBUG:-0}
export TF_TRUNK_DECOMP_DEBUG=${TF_TRUNK_DECOMP_DEBUG:-0}
if [[ -f ${CONT_REPO_ROOT}/human_aware_rl/ppo/ppo.py && -d ${CONT_REPO_ROOT}/overcooked_ai ]]; then
  PPO_ENTRY=${CONT_REPO_ROOT}/human_aware_rl/ppo/ppo.py
  BC_RUNS_DIR=${CONT_REPO_ROOT}/human_aware_rl/data/bc_runs
  export PYTHONPATH=${CONT_REPO_ROOT}/human_aware_rl:${CONT_REPO_ROOT}/overcooked_ai:${CONT_REPO_ROOT}
elif [[ -f ${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl/ppo/ppo.py && -d ${CONT_REPO_ROOT}/human_aware_rl/overcooked_ai ]]; then
  PPO_ENTRY=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl/ppo/ppo.py
  BC_RUNS_DIR=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl/data/bc_runs
  export PYTHONPATH=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl:${CONT_REPO_ROOT}/human_aware_rl/overcooked_ai:${CONT_REPO_ROOT}/human_aware_rl
else
  echo 'ERROR: Could not resolve ppo.py layout.'
  ls -la ${CONT_REPO_ROOT}/ || true
  exit 1
fi
echo 'Checking baselines import...'
python -c \"import baselines; print('baselines from:', baselines.__file__)\"
echo 'Checking model.py loss_names...'
python -c \"
from baselines.ppo2.model import Model
import inspect
src = inspect.getsource(Model.__init__)
if 'raw_grad_norm' in src:
    print('SUCCESS: model.py has raw_grad_norm logging')
else:
    print('WARNING: model.py does NOT have raw_grad_norm - using container version')
\"
mkdir -p ${CONT_GT_DIR}/tf_grad_diag
RUN_WORKDIR=${CONT_GT_DIR}/tf_grad_diag/workdir
mkdir -p \${RUN_WORKDIR}
cd \${RUN_WORKDIR}
mkdir -p data/ppo_runs
ln -sfn \${BC_RUNS_DIR} data/bc_runs
python \${PPO_ENTRY} with \
  params.TIMESTAMP_DIR=False \
  params.SAVE_DIR='${CONT_GT_DIR}/tf_grad_diag/' \
  params.EX_NAME='tf_grad_diag' \
  params.mdp_params.layout_name='simple' \
  params.OTHER_AGENT_TYPE='sp' \
  params.SEEDS='[2229]' \
  params.PPO_RUN_TOT_TIMESTEPS=${PPO_RUN_TOT_TIMESTEPS:-600000} \
  params.GPU_ID=0 \
  params.LR=1e-3 params.REW_SHAPING_HORIZON=2.5e6 params.VF_COEF=0.5 params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000 SIZE_HIDDEN_LAYERS=32 sim_threads=30
echo 'Done.'
"

apptainer exec \
  --bind "${HOST_REPO_ROOT}:${CONT_REPO_ROOT}" \
  --bind "${HOST_GT_DIR}:${CONT_GT_DIR}" \
  --bind "${HOST_REPO_ROOT}/baselines:/opt/human_aware_rl/baselines" \
  "${SIF_PATH}" \
  bash -lc "${CONTAINER_BOOTSTRAP}"

echo "==============================================================="
echo "End: $(date)"
echo "==============================================================="
