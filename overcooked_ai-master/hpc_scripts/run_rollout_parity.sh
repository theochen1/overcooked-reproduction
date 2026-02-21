#!/bin/bash
#SBATCH --job-name=rollout_parity
#SBATCH --output=logs/rollout_parity_%j.out
#SBATCH --error=logs/rollout_parity_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal

set -eo pipefail

# Source shared config (PROJECT_ROOT, PATH, PYTHONPATH, logging helpers).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/config.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh" ]; then
    DEFAULT_HPC_CONFIG="${SLURM_SUBMIT_DIR}/hpc_scripts/config.sh"
else
    DEFAULT_HPC_CONFIG="${SCRIPT_DIR}/config.sh"
fi
source "${HPC_CONFIG:-${DEFAULT_HPC_CONFIG}}"

log_start

MODE="${ROLL_PARITY_MODE:-all}"
LAYOUT="${ROLL_PARITY_LAYOUT:-coordination_ring_legacy}"
SEED="${ROLL_PARITY_SEED:-0}"
STEPS="${ROLL_PARITY_STEPS:-200}"
HORIZON="${ROLL_PARITY_HORIZON:-400}"
FIXED_AGENT_IDX="${ROLL_PARITY_FIXED_AGENT_IDX:-0}"
RANDOM_AGENT_IDX="${ROLL_PARITY_RANDOM_AGENT_IDX:-0}"

OUT_DIR="${PROJECT_ROOT}/parity_artifacts"
mkdir -p "${OUT_DIR}" "${LOGS_DIR}"

OUT_JSON="${OUT_DIR}/rollout_parity_${LAYOUT}_seed${SEED}_${SLURM_JOB_ID}.json"

echo "Running rollout parity canary..."
echo "  mode=${MODE}"
echo "  layout=${LAYOUT}"
echo "  seed=${SEED}"
echo "  steps=${STEPS}"
echo "  horizon=${HORIZON}"
echo "  fixed_agent_idx=${FIXED_AGENT_IDX}"
echo "  random_agent_idx=${RANDOM_AGENT_IDX}"
echo "  out_json=${OUT_JSON}"
echo ""

CMD=(
    python -m human_aware_rl.jaxmarl.rollout_parity_canary
    --mode "${MODE}"
    --layout-name "${LAYOUT}"
    --seed "${SEED}"
    --steps "${STEPS}"
    --horizon "${HORIZON}"
    --out-json "${OUT_JSON}"
)

if [ "${RANDOM_AGENT_IDX}" = "1" ]; then
    CMD+=(--random-agent-idx)
else
    CMD+=(--fixed-agent-idx "${FIXED_AGENT_IDX}")
fi

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "Rollout parity canary complete."
echo "Result JSON: ${OUT_JSON}"

log_end "${EXIT_CODE}"
exit "${EXIT_CODE}"
