#!/bin/bash
set -euo pipefail

# Submit paper-reproduction TF jobs as one Slurm job per experiment.
# This parallelizes runs across nodes and keeps logs readable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_SBATCH="${SCRIPT_DIR}/run_harl_ground_truth.sbatch"
if [[ ! -f "${RUN_SBATCH}" ]]; then
  echo "ERROR: Missing ${RUN_SBATCH}"
  exit 1
fi

# Comma-separated phases to submit: ppo_sp,ppo_bc
PHASES="${PHASES:-ppo_sp,ppo_bc}"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SIF_PATH="${SIF_PATH:-/home/tchen22/containers/harl_tf1.sif}"
HOST_GT_DIR="${HOST_GT_DIR:-${HOST_REPO_ROOT}/ground_truth_runs}"
APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD:-}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"

# Resource overrides for submitted jobs.
PARTITION="${PARTITION:-mit_normal}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MEMORY="${MEMORY:-32G}"
CPUS="${CPUS:-8}"
EXCLUSIVE_PER_JOB="${EXCLUSIVE_PER_JOB:-1}"   # 1 => one job per node
EXTRA_SBATCH_ARGS="${EXTRA_SBATCH_ARGS:-}"    # Optional raw sbatch args

submit_one() {
  local run_name="$1"
  local layout="$2"
  local other_agent_type="$3"
  local seeds_json="$4"
  local timesteps="$5"
  local extra_sacred_args="$6"
  local effective_run_name="${run_name}${RUN_NAME_SUFFIX}"

  local job_name="gt_${effective_run_name}"
  local out_path="${LOG_DIR}/${job_name}_%j.out"
  local err_path="${LOG_DIR}/${job_name}_%j.err"

  local job_id
  local -a node_args
  node_args=(--nodes=1 --ntasks=1 --cpus-per-task="${CPUS}")
  if [[ "${EXCLUSIVE_PER_JOB}" == "1" ]]; then
    node_args+=(--exclusive)
  fi

  job_id=$(HOST_REPO_ROOT="${HOST_REPO_ROOT}" \
    HOST_GT_DIR="${HOST_GT_DIR}" \
    SIF_PATH="${SIF_PATH}" \
    APPTAINER_MODULE_CMD="${APPTAINER_MODULE_CMD}" \
    RUN_NAME="${effective_run_name}" \
    LAYOUT="${layout}" \
    OTHER_AGENT_TYPE="${other_agent_type}" \
    SEEDS_JSON="${seeds_json}" \
    PPO_RUN_TOT_TIMESTEPS="${timesteps}" \
    EXTRA_SACRED_ARGS="${extra_sacred_args}" \
    sbatch --parsable \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --mem="${MEMORY}" \
    "${node_args[@]}" \
    --job-name="${job_name}" \
    --output="${out_path}" \
    --error="${err_path}" \
    --export=ALL \
    ${EXTRA_SBATCH_ARGS} \
    "${RUN_SBATCH}")

  echo "${job_id}  ${job_name}  layout=${layout}  partner=${other_agent_type}"
}

phase_enabled() {
  local needle="$1"
  [[ ",${PHASES}," == *",${needle},"* ]]
}

echo "Submitting HARL ground-truth jobs in parallel"
echo "PHASES=${PHASES}"
echo "HOST_REPO_ROOT=${HOST_REPO_ROOT}"
echo "HOST_GT_DIR=${HOST_GT_DIR}"
echo "SIF_PATH=${SIF_PATH}"
echo "RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-<none>}"
echo "EXCLUSIVE_PER_JOB=${EXCLUSIVE_PER_JOB}"
echo "EXTRA_SBATCH_ARGS=${EXTRA_SBATCH_ARGS:-<none>}"
echo ""

if phase_enabled "ppo_sp"; then
  # PPO-SP (paper table): VF=0.5 for all layouts.
  SP_SEEDS="[2229, 7649, 7225, 9807, 386]"
  submit_one "ppo_sp_simple"    "simple"    "sp" "${SP_SEEDS}" "6000000"  "params.LR=1e-3 params.REW_SHAPING_HORIZON=2.5e6 params.VF_COEF=0.5 params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_sp_unident_s" "unident_s" "sp" "${SP_SEEDS}" "7000000"  "params.LR=1e-3 params.REW_SHAPING_HORIZON=2.5e6 params.VF_COEF=0.5 params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_sp_random1"   "random1"   "sp" "${SP_SEEDS}" "10000000" "params.LR=6e-4 params.REW_SHAPING_HORIZON=3.5e6 params.VF_COEF=0.5 params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_sp_random0"   "random0"   "sp" "${SP_SEEDS}" "7500000"  "params.LR=8e-4 params.REW_SHAPING_HORIZON=2.5e6 params.VF_COEF=0.5 params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_sp_random3"   "random3"   "sp" "${SP_SEEDS}" "8000000"  "params.LR=8e-4 params.REW_SHAPING_HORIZON=2.5e6 params.VF_COEF=0.5 params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
fi

if phase_enabled "ppo_bc"; then
  # PPO-BC (and PPO-HP-equivalent hyperparameters) table.
  BC_TRAIN_SEEDS="[9456, 1887, 5578, 5987, 516]"
  BC_TEST_SEEDS="[2888, 7424, 7360, 4467, 184]"

  submit_one "ppo_bc_train_simple"    "simple"    "bc_train" "${BC_TRAIN_SEEDS}" "8000000"  "params.LR=1e-3 params.REW_SHAPING_HORIZON=1e6 params.VF_COEF=0.5 params.MINIBATCHES=10 params.LR_ANNEALING=3 params.SELF_PLAY_HORIZON='[5e5, 3e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_bc_test_simple"     "simple"    "bc_test"  "${BC_TEST_SEEDS}"  "8000000"  "params.LR=1e-3 params.REW_SHAPING_HORIZON=1e6 params.VF_COEF=0.5 params.MINIBATCHES=10 params.LR_ANNEALING=3 params.SELF_PLAY_HORIZON='[5e5, 3e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"

  submit_one "ppo_bc_train_unident_s" "unident_s" "bc_train" "${BC_TRAIN_SEEDS}" "10000000" "params.LR=1e-3 params.REW_SHAPING_HORIZON=6e6 params.VF_COEF=0.5 params.MINIBATCHES=12 params.LR_ANNEALING=3 params.SELF_PLAY_HORIZON='[1e6, 7e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_bc_test_unident_s"  "unident_s" "bc_test"  "${BC_TEST_SEEDS}"  "10000000" "params.LR=1e-3 params.REW_SHAPING_HORIZON=6e6 params.VF_COEF=0.5 params.MINIBATCHES=12 params.LR_ANNEALING=3 params.SELF_PLAY_HORIZON='[1e6, 7e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"

  submit_one "ppo_bc_train_random1"   "random1"   "bc_train" "${BC_TRAIN_SEEDS}" "16000000" "params.LR=1e-3 params.REW_SHAPING_HORIZON=5e6 params.VF_COEF=0.5 params.MINIBATCHES=15 params.LR_ANNEALING=1.5 params.SELF_PLAY_HORIZON='[2e6, 6e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_bc_test_random1"    "random1"   "bc_test"  "${BC_TEST_SEEDS}"  "16000000" "params.LR=1e-3 params.REW_SHAPING_HORIZON=5e6 params.VF_COEF=0.5 params.MINIBATCHES=15 params.LR_ANNEALING=1.5 params.SELF_PLAY_HORIZON='[2e6, 6e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"

  submit_one "ppo_bc_train_random0"   "random0"   "bc_train" "${BC_TRAIN_SEEDS}" "9000000"  "params.LR=1.5e-3 params.REW_SHAPING_HORIZON=4e6 params.VF_COEF=0.1 params.MINIBATCHES=15 params.LR_ANNEALING=2 params.SELF_PLAY_HORIZON=None params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_bc_test_random0"    "random0"   "bc_test"  "${BC_TEST_SEEDS}"  "9000000"  "params.LR=1.5e-3 params.REW_SHAPING_HORIZON=4e6 params.VF_COEF=0.1 params.MINIBATCHES=15 params.LR_ANNEALING=2 params.SELF_PLAY_HORIZON=None params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"

  submit_one "ppo_bc_train_random3"   "random3"   "bc_train" "${BC_TRAIN_SEEDS}" "12000000" "params.LR=1.5e-3 params.REW_SHAPING_HORIZON=4e6 params.VF_COEF=0.1 params.MINIBATCHES=15 params.LR_ANNEALING=3 params.SELF_PLAY_HORIZON='[1e6, 4e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
  submit_one "ppo_bc_test_random3"    "random3"   "bc_test"  "${BC_TEST_SEEDS}"  "12000000" "params.LR=1.5e-3 params.REW_SHAPING_HORIZON=4e6 params.VF_COEF=0.1 params.MINIBATCHES=15 params.LR_ANNEALING=3 params.SELF_PLAY_HORIZON='[1e6, 4e6]' params.sim_threads=30 params.TOTAL_BATCH_SIZE=12000"
fi

echo ""
echo "Done. Submitted jobs are parallel and independently schedulable."
