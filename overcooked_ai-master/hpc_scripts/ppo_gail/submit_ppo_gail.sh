#!/bin/bash
# ============================================================================
# Submit all PPO GAIL training jobs (25 = 5 layouts × 5 seeds)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${HPC_DIR}/logs"

mkdir -p "${LOGS_DIR}"

echo "Submitting PPO GAIL jobs (25 = 5 layouts × 5 seeds)..."
echo "============================================================================"

LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination" "counter_circuit")
SEEDS=(0 10 20 30 40)

COUNT=0
for layout in "${LAYOUTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        script="${SCRIPT_DIR}/${layout}_seed${seed}.sh"
        if [ -f "$script" ]; then
            JOB_ID=$(sbatch --parsable \
                --output="${LOGS_DIR}/ppo_gail_${layout}_seed${seed}_%j.out" \
                --error="${LOGS_DIR}/ppo_gail_${layout}_seed${seed}_%j.err" \
                "$script")
            echo "  ppo_gail_${layout}_s${seed}: Job ${JOB_ID}"
            ((COUNT++))
        else
            echo "WARNING: Script not found: $script"
        fi
    done
done

echo "============================================================================"
echo "PPO_GAIL jobs submitted: ${COUNT}"
echo "============================================================================"
echo "To check job status: squeue -u \$USER"
