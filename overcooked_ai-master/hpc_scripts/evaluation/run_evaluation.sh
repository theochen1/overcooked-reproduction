#!/bin/bash
# ============================================================================
# HPC Evaluation Script - Human Proxy Evaluation
# ============================================================================
# Runs evaluation for BC, PPO_SP, PPO_BC, PPO_GAIL paired with Human Proxy.
# Generates JSON results and Figure 4-style plot.
#
# Usage (SLURM):
#   sbatch hpc_scripts/evaluation/run_evaluation.sh
#
# Usage (Interactive/Background):
#   nohup bash hpc_scripts/evaluation/run_evaluation.sh > hpc_scripts/logs/eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ============================================================================

#SBATCH --job-name=overcooked_eval
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/eval_%j.out
#SBATCH --error=../logs/eval_%j.err

set -eo pipefail

# ============================================================================
# Configuration
# ============================================================================
# Source shared config (sets PROJECT_ROOT, conda, PYTHONPATH, etc.)
# HPC_CONFIG is set by submit scripts; fallback to HOME-based path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/config.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh"
else
    DEFAULT_HPC_CONFIG="$(cd "${SCRIPT_DIR}/.." && pwd)/config.sh"
fi
source "${HPC_CONFIG:-${DEFAULT_HPC_CONFIG}}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOGS_DIR}"

# Output paths with timestamp
RESULTS_JSON="${RESULTS_DIR}/hpc_eval_results_${TIMESTAMP}.json"
FIGURE_PATH="${RESULTS_DIR}/hpc_eval_figure4_${TIMESTAMP}.png"

# ============================================================================
# Environment Setup
# ============================================================================
echo "============================================================"
echo "HPC Model Evaluation - Started at $(date)"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Results will be saved to:"
echo "  - JSON: ${RESULTS_JSON}"
echo "  - Figure: ${FIGURE_PATH}"
echo "============================================================"

echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "============================================================"

# ============================================================================
# Run Evaluation
# ============================================================================
echo ""
echo "Starting evaluation..."
echo ""

python -m human_aware_rl.evaluation.evaluate_hpc_models \
    --num_games 10 \
    --output "${RESULTS_JSON}"

echo ""
echo "============================================================"
echo "Evaluation complete. Generating plot..."
echo "============================================================"
echo ""

# ============================================================================
# Generate Plot
# ============================================================================
python -m human_aware_rl.evaluation.plot_hpc_results \
    --input "${RESULTS_JSON}" \
    --output "${FIGURE_PATH}" \
    --include_gail

# Also copy to a "latest" file for easy access
cp "${RESULTS_JSON}" "${RESULTS_DIR}/hpc_eval_results_latest.json"
cp "${FIGURE_PATH}" "${RESULTS_DIR}/hpc_eval_figure4_latest.png"

echo ""
echo "============================================================"
echo "HPC Model Evaluation - Completed at $(date)"
echo "============================================================"
echo "Results saved to:"
echo "  - JSON: ${RESULTS_JSON}"
echo "  - Figure: ${FIGURE_PATH}"
echo "  - Latest JSON: ${RESULTS_DIR}/hpc_eval_results_latest.json"
echo "  - Latest Figure: ${RESULTS_DIR}/hpc_eval_figure4_latest.png"
echo "============================================================"
