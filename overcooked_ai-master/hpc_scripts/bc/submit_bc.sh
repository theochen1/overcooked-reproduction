#!/bin/bash
# ============================================================================
# Submit all BC training jobs
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${HPC_DIR}/logs"

mkdir -p "${LOGS_DIR}"

echo "Submitting BC training jobs..."
echo "============================================================================"

declare -a BC_JOB_IDS
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    JOB_ID=$(sbatch --parsable \
        --output="${LOGS_DIR}/bc_${layout}_%j.out" \
        --error="${LOGS_DIR}/bc_${layout}_%j.err" \
        "${SCRIPT_DIR}/${layout}.sh")
    BC_JOB_IDS+=("$JOB_ID")
    echo "Submitted bc_${layout}: Job ID ${JOB_ID}"
done

echo "============================================================================"
echo "Total BC jobs submitted: ${#BC_JOB_IDS[@]}"
echo "Dependency string: afterok:$(IFS=:; echo "${BC_JOB_IDS[*]}")"
echo "============================================================================"

export BC_JOB_IDS
