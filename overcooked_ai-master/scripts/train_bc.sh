#!/bin/bash
#SBATCH --job-name=bc_train
#SBATCH --time=47:00:00
#SBATCH --mem=32G
#SBATCH -n 16

# Create logs directory
mkdir -p logs

# Activate conda environment
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


# Train BC models for all layouts
python -m human_aware_rl.imitation.train_bc_models --all_layouts

echo "BC training complete!"

