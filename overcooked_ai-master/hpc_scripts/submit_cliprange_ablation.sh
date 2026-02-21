#!/bin/bash
# Submit cliprange ablation grid:
#   clip_eps_start: configurable list (default: 0.05,0.2)
#   schedules: configurable list (default: constant,linear_to_end)
#   layouts: coordination_ring, counter_circuit
#   fixed seed/timesteps for matched comparison

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_CONFIG_PATH="${SCRIPT_DIR}/config.sh"
source "${HPC_CONFIG_PATH}"

SEED="${1:-0}"
TIMESTEPS="${2:-600000}"  # 50 PPO updates with 30x400 batch
RESULTS_ROOT="${PROJECT_ROOT}/parity_artifacts/cliprange_ablation"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SCHEDULES_CSV="${SCHEDULES_CSV:-constant,linear_to_end}"
CLIP_STARTS_CSV="${CLIP_STARTS_CSV:-0.05,0.2}"
CLIP_EPS_END="${CLIP_EPS_END:-0.01}"
CLIP_END_FRACTION="${CLIP_END_FRACTION:-1.0}"
mkdir -p "${RESULTS_ROOT}" "${LOGS_DIR}"

submit_one() {
    local layout="$1"
    local schedule="$2"
    local clip_start="$3"
    local clip_tag
    clip_tag="$(echo "${clip_start}" | tr '.' 'p')"
    local end_tag
    local frac_tag
    end_tag="$(echo "${CLIP_EPS_END}" | tr '.' 'p')"
    frac_tag="$(echo "${CLIP_END_FRACTION}" | tr '.' 'p')"
    local variant_tag=""
    if [[ "${schedule}" == "linear_to_end" ]]; then
        variant_tag="_end${end_tag}_f${frac_tag}"
    fi
    local job_name="clip_${schedule}_eps${clip_tag}${variant_tag}_${layout}_s${SEED}"
    local out_file="${LOGS_DIR}/${job_name}_%j.out"
    local err_file="${LOGS_DIR}/${job_name}_%j.err"
    local run_dir="${RESULTS_ROOT}/${schedule}/eps_${clip_start}${variant_tag}/${layout}/seed${SEED}"

    local clip_end_arg=""
    if [[ "${schedule}" == "linear_to_end" ]]; then
        clip_end_arg="--clip_eps_end ${CLIP_EPS_END} --clip_end_fraction ${CLIP_END_FRACTION}"
    fi
    local cmd="source \"${HPC_CONFIG_PATH}\" && python -m human_aware_rl.ppo.train_ppo_sp --layout ${layout} --seed ${SEED} --timesteps ${TIMESTEPS} --cliprange_schedule ${schedule} --clip_eps ${clip_start} ${clip_end_arg} --results_dir \"${run_dir}\" ${EXTRA_ARGS}"
    local job_id
    job_id=$(sbatch --parsable \
        --job-name="${job_name}" \
        --output="${out_file}" \
        --error="${err_file}" \
        --time="06:00:00" \
        --mem="32G" \
        --cpus-per-task="8" \
        --partition="${SLURM_PARTITION}" \
        --export=ALL,HPC_CONFIG="${HPC_CONFIG_PATH}" \
        --wrap "${cmd}")

    echo "${job_id},${layout},${schedule},${clip_start},${CLIP_EPS_END},${CLIP_END_FRACTION},${SEED},${TIMESTEPS},${out_file},${err_file}"
}

echo "job_id,layout,schedule,clip_eps_start,clip_eps_end,clip_end_fraction,seed,timesteps,out_log,err_log"
IFS=',' read -r -a schedules <<< "${SCHEDULES_CSV}"
IFS=',' read -r -a clip_starts <<< "${CLIP_STARTS_CSV}"
for layout in coordination_ring counter_circuit; do
    for schedule in "${schedules[@]}"; do
        for clip_start in "${clip_starts[@]}"; do
            submit_one "${layout}" "${schedule}" "${clip_start}"
        done
    done
done
