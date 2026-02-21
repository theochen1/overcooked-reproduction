#!/bin/bash
#SBATCH --job-name=overcooked_all
#SBATCH --output=logs/train_all_%j.out
#SBATCH --error=logs/train_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

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


echo "========================================="
echo "Step 1: Training BC models"
echo "========================================="
python -m human_aware_rl.imitation.train_bc_models --all_layouts

echo "========================================="
echo "Step 2: Training PPO Self-Play"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40

echo "========================================="
echo "Step 3: Training PPO with BC partner"
echo "========================================="
python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0,10,20,30,40

echo "========================================="
echo "All training complete!"
echo "========================================="

