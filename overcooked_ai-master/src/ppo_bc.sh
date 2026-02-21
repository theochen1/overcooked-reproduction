#!/bin/bash
#SBATCH -n 16
#SBATCH -t 47:00:00
#SBATCH --mem=32G

# Environment bootstrap (portable across clusters)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-overcooked}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/${CONDA_ENV_NAME}}"
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV_NAME}" || true
fi
if [ -d "${CONDA_ENV_PATH}/bin" ]; then
    export PATH="${CONDA_ENV_PATH}/bin:$PATH"
fi
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}/src/human_aware_rl"


python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0 --fast
