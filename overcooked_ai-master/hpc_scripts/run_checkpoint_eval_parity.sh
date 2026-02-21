#!/bin/bash
#SBATCH --job-name=ckpt_eval_parity
#SBATCH --output=logs/ckpt_eval_parity_%j.out
#SBATCH --error=logs/ckpt_eval_parity_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

set -eo pipefail

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

LAYOUT="${CKPT_PARITY_LAYOUT:-coordination_ring}"
SEED="${CKPT_PARITY_SEED:-0}"
TIMESTEPS="${CKPT_PARITY_TIMESTEPS:-12000}"
NUM_GAMES="${CKPT_PARITY_NUM_GAMES:-20}"
BASE_DIR="${PROJECT_ROOT}/parity_artifacts/checkpoint_eval"
TRAIN_RESULTS_DIR="${BASE_DIR}/train_runs"
REPORT_JSON="${BASE_DIR}/checkpoint_eval_parity_${LAYOUT}_seed${SEED}_${SLURM_JOB_ID}.json"

mkdir -p "${BASE_DIR}" "${TRAIN_RESULTS_DIR}" "${LOGS_DIR}"

echo "Training short checkpoint for parity..."
python -m human_aware_rl.ppo.train_ppo_sp \
    --layout "${LAYOUT}" \
    --seed "${SEED}" \
    --timesteps "${TIMESTEPS}" \
    --results_dir "${TRAIN_RESULTS_DIR}" \
    --quiet

CKPT_DIR=$(ls -dt "${TRAIN_RESULTS_DIR}"/*/checkpoint_* | sed -n '1p')
if [ -z "${CKPT_DIR}" ]; then
    echo "ERROR: Could not find checkpoint under ${TRAIN_RESULTS_DIR}"
    exit 2
fi

echo "Using checkpoint: ${CKPT_DIR}"
python -m human_aware_rl.jaxmarl.checkpoint_eval_parity \
    --checkpoint-dir "${CKPT_DIR}" \
    --num-games "${NUM_GAMES}" \
    --seed 123 \
    --out-json "${REPORT_JSON}"

EXIT_CODE=$?
echo "Checkpoint evaluator parity report: ${REPORT_JSON}"
log_end "${EXIT_CODE}"
exit "${EXIT_CODE}"
