#!/bin/bash
#SBATCH --job-name=harl_full_native_bc
#SBATCH --output=logs/harl_full_native_bc_%j.out
#SBATCH --error=logs/harl_full_native_bc_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -euo pipefail

mkdir -p logs

echo "==============================================================="
echo "HARL full experiment (native BC runs)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "==============================================================="

# Required container image.
SIF_PATH="${SIF_PATH:-/home/tchen22/containers/harl_tf1.sif}"

# Host repo root (must contain human_aware_rl/).
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$PWD}"

# Optional module command for systems where apptainer is provided by modules.
APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"

# Comma-separated phases: ppo_sp,pbt,ppo_bc,ppo_hm
PHASES="${PHASES:-ppo_sp,pbt,ppo_bc,ppo_hm}"

# Use 64cpu scripts by default (set USE_64CPU=0 for non-64cpu variants).
USE_64CPU="${USE_64CPU:-1}"

CONT_REPO_ROOT="/opt/workdir"
CONT_HARL_ROOT="${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl"

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

# Fallback: try common cluster module name when command still missing.
if ! command -v apptainer >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load apptainer/1.4.2 >/dev/null 2>&1 || module load apptainer >/dev/null 2>&1 || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not found in PATH."
  echo "Set APPTAINER_MODULE_CMD, e.g. APPTAINER_MODULE_CMD='module load apptainer/1.4.2'"
  exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "ERROR: SIF image not found at ${SIF_PATH}"
  exit 1
fi

if [[ ! -d "${HOST_REPO_ROOT}/human_aware_rl" ]]; then
  echo "ERROR: HOST_REPO_ROOT does not contain human_aware_rl/"
  echo "HOST_REPO_ROOT=${HOST_REPO_ROOT}"
  exit 1
fi

echo "SIF_PATH=${SIF_PATH}"
echo "HOST_REPO_ROOT=${HOST_REPO_ROOT}"
echo "PHASES=${PHASES}"
echo "USE_64CPU=${USE_64CPU}"
echo ""

CONTAINER_BOOTSTRAP="
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate harl
export PYTHONPATH=${CONT_REPO_ROOT}/human_aware_rl/human_aware_rl:${CONT_REPO_ROOT}/human_aware_rl/overcooked_ai:${CONT_REPO_ROOT}/human_aware_rl:\${PYTHONPATH:-}
cd ${CONT_HARL_ROOT}

if [[ ! -d data/bc_runs ]]; then
  echo 'ERROR: Missing native BC directory data/bc_runs'
  exit 1
fi
if [[ ! -f data/bc_runs/best_bc_model_paths.pickle ]]; then
  echo 'ERROR: Missing data/bc_runs/best_bc_model_paths.pickle'
  exit 1
fi

echo 'Using native BC runs from:' \$(pwd)/data/bc_runs
echo 'Available BC dirs:' \$(ls -1 data/bc_runs | wc -l)

IFS=',' read -r -a phase_arr <<< \"${PHASES}\"
for phase in \"\${phase_arr[@]}\"; do
  echo '---------------------------------------------------------------'
  echo \"Running phase: \${phase}\"
  echo '---------------------------------------------------------------'
  if [[ \"${USE_64CPU}\" == \"1\" ]]; then
    case \"\${phase}\" in
      ppo_sp) bash experiments/ppo_sp_experiments_64cpu.sh ;;
      pbt)    bash experiments/pbt_experiments_64cpu.sh ;;
      ppo_bc) bash experiments/ppo_bc_experiments_64cpu.sh ;;
      ppo_hm) bash experiments/ppo_hm_experiments_64cpu.sh ;;
      *) echo \"ERROR: Unknown phase '\${phase}'\"; exit 1 ;;
    esac
  else
    case \"\${phase}\" in
      ppo_sp) bash experiments/ppo_sp_experiments.sh ;;
      pbt)    bash experiments/pbt_experiments.sh ;;
      ppo_bc) bash experiments/ppo_bc_experiments.sh ;;
      ppo_hm) bash experiments/ppo_hm_experiments.sh ;;
      *) echo \"ERROR: Unknown phase '\${phase}'\"; exit 1 ;;
    esac
  fi
done

echo 'All requested phases completed.'
"

apptainer exec \
  --bind "${HOST_REPO_ROOT}:${CONT_REPO_ROOT}" \
  --bind "${HOST_REPO_ROOT}:/workspace" \
  "${SIF_PATH}" \
  bash -lc "${CONTAINER_BOOTSTRAP}"

echo "==============================================================="
echo "End: $(date)"
echo "==============================================================="

