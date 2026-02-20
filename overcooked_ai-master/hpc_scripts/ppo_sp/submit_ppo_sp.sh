#!/bin/bash
# ============================================================================
# Submit all PPO Self-Play training jobs (25 = 5 layouts × 5 seeds)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${HPC_DIR}/logs"

mkdir -p "${LOGS_DIR}"

echo "Submitting PPO Self-Play jobs (25 = 5 layouts × 5 seeds)..."
echo "============================================================================"

COUNT=0
for layout in cramped_room asymmetric_advantages coordination_ring forced_coordination counter_circuit; do
    for seed in 0 10 20 30 40; do
        JOB_ID=$(sbatch --parsable \
            --output="${LOGS_DIR}/ppo_sp_${layout}_seed${seed}_%j.out" \
            --error="${LOGS_DIR}/ppo_sp_${layout}_seed${seed}_%j.err" \
            "${SCRIPT_DIR}/${layout}_seed${seed}.sh")
        echo "  ppo_sp_${layout}_s${seed}: Job ${JOB_ID}"
        ((COUNT++))
    done
done

echo "============================================================================"
echo "PPO_SP jobs submitted: ${COUNT}"
echo "============================================================================"
