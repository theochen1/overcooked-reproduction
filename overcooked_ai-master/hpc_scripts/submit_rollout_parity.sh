#!/bin/bash
# Submit rollout parity canary as a SLURM job.
#
# Usage:
#   bash hpc_scripts/submit_rollout_parity.sh [layout] [seed] [steps] [mode]
#
# Examples:
#   bash hpc_scripts/submit_rollout_parity.sh
#   bash hpc_scripts/submit_rollout_parity.sh coordination_ring_legacy 0 200 all
#   bash hpc_scripts/submit_rollout_parity.sh random0_legacy 0 400 step_hash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONFIG_PATH="${SCRIPT_DIR}/config.sh"
source "${HPC_CONFIG_PATH}"

LAYOUT="${1:-coordination_ring_legacy}"
SEED="${2:-0}"
STEPS="${3:-200}"
MODE="${4:-all}"

# Optional env overrides:
#   ROLL_PARITY_HORIZON (default 400)
#   ROLL_PARITY_FIXED_AGENT_IDX (default 0)
#   ROLL_PARITY_RANDOM_AGENT_IDX (default 0)
HORIZON="${ROLL_PARITY_HORIZON:-400}"
FIXED_AGENT_IDX="${ROLL_PARITY_FIXED_AGENT_IDX:-0}"
RANDOM_AGENT_IDX="${ROLL_PARITY_RANDOM_AGENT_IDX:-0}"

mkdir -p "${LOGS_DIR}"

JOB_ID=$(sbatch --parsable \
    --export=ALL,HPC_CONFIG="${HPC_CONFIG_PATH}",ROLL_PARITY_MODE="${MODE}",ROLL_PARITY_LAYOUT="${LAYOUT}",ROLL_PARITY_SEED="${SEED}",ROLL_PARITY_STEPS="${STEPS}",ROLL_PARITY_HORIZON="${HORIZON}",ROLL_PARITY_FIXED_AGENT_IDX="${FIXED_AGENT_IDX}",ROLL_PARITY_RANDOM_AGENT_IDX="${RANDOM_AGENT_IDX}" \
    --output="${LOGS_DIR}/rollout_parity_${LAYOUT}_seed${SEED}_%j.out" \
    --error="${LOGS_DIR}/rollout_parity_${LAYOUT}_seed${SEED}_%j.err" \
    "${SCRIPT_DIR}/run_rollout_parity.sh")

echo "Submitted rollout parity job:"
echo "  job_id=${JOB_ID}"
echo "  layout=${LAYOUT}"
echo "  seed=${SEED}"
echo "  steps=${STEPS}"
echo "  mode=${MODE}"
echo "  logs=${LOGS_DIR}/rollout_parity_${LAYOUT}_seed${SEED}_${JOB_ID}.out"
