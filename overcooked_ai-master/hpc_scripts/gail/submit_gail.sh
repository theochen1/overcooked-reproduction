#!/bin/bash
# ============================================================================
# Submit all GAIL training jobs
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${HPC_DIR}/logs"

mkdir -p "${LOGS_DIR}"

echo "Submitting GAIL training jobs..."
echo "============================================================================"

declare -a GAIL_JOB_IDS
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    JOB_ID=$(sbatch --parsable \
        --output="${LOGS_DIR}/gail_${layout}_%j.out" \
        --error="${LOGS_DIR}/gail_${layout}_%j.err" \
        "${SCRIPT_DIR}/${layout}.sh")
    GAIL_JOB_IDS+=("$JOB_ID")
    echo "Submitted gail_${layout}: Job ID ${JOB_ID}"
done

echo "============================================================================"
echo "Total GAIL jobs submitted: ${#GAIL_JOB_IDS[@]}"
echo "After GAIL jobs complete, submit PPO_GAIL with:"
echo "  cd ${HPC_DIR}/ppo_gail && ./submit_ppo_gail.sh"
echo "============================================================================"

export GAIL_JOB_IDS
