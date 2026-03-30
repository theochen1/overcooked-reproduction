#!/bin/bash
set -euo pipefail

# Merge per-seed run directories created via SPLIT_SEEDS=1 into a single
# run directory compatible with load_training_data / plot_ppo_run.
#
# Inputs (env vars):
#   BASE_RUN_NAME   e.g. ppo_sp_simple
#   SEEDS           e.g. "2229,7649,7225,9807,386" or "[2229, 7649, ...]"
#   HOST_GT_DIR     defaults to <repo>/ground_truth_runs
#
# Output:
#   ${HOST_GT_DIR}/${BASE_RUN_NAME}_combined/
#     config.pickle
#     seed<seed>/training_info.pickle
#     seed<seed>/ppo_agent/...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
HOST_GT_DIR="${HOST_GT_DIR:-${HOST_REPO_ROOT}/ground_truth_runs}"
BASE_RUN_NAME="${BASE_RUN_NAME:-}"
SEEDS="${SEEDS:-}"

if [[ -z "${BASE_RUN_NAME}" ]]; then
  echo "ERROR: BASE_RUN_NAME is required (e.g., BASE_RUN_NAME=ppo_sp_simple)"
  exit 1
fi

if [[ -z "${SEEDS}" ]]; then
  echo "ERROR: SEEDS is required (e.g., SEEDS='2229,7649,7225,9807,386')"
  exit 1
fi

parse_seed_list() {
  local seeds_raw="$1"
  local compact
  compact="$(echo "${seeds_raw}" | tr -d '[] ' )"
  if [[ -z "${compact}" ]]; then
    return 0
  fi
  echo "${compact}" | tr ',' '\n'
}

DEST_DIR="${HOST_GT_DIR}/${BASE_RUN_NAME}_combined"
mkdir -p "${DEST_DIR}"

copied_config=0
copied_seeds=0

echo "Merging seed runs:"
echo "  HOST_GT_DIR=${HOST_GT_DIR}"
echo "  BASE_RUN_NAME=${BASE_RUN_NAME}"
echo "  DEST_DIR=${DEST_DIR}"
echo "  SEEDS=${SEEDS}"
echo ""

seed=""
while IFS= read -r seed; do
  [[ -z "${seed}" ]] && continue
  SRC_RUN_DIR="${HOST_GT_DIR}/${BASE_RUN_NAME}_seed${seed}"
  SRC_SEED_DIR="${SRC_RUN_DIR}/seed${seed}"
  DEST_SEED_DIR="${DEST_DIR}/seed${seed}"

  if [[ ! -d "${SRC_SEED_DIR}" ]]; then
    echo "WARN: missing ${SRC_SEED_DIR}; skipping seed ${seed}"
    continue
  fi

  mkdir -p "${DEST_SEED_DIR}"
  rsync -a "${SRC_SEED_DIR}/" "${DEST_SEED_DIR}/"
  echo "  copied seed${seed}"
  copied_seeds=$((copied_seeds + 1))

  if [[ "${copied_config}" -eq 0 && -f "${SRC_RUN_DIR}/config.pickle" ]]; then
    cp -f "${SRC_RUN_DIR}/config.pickle" "${DEST_DIR}/config.pickle"
    copied_config=1
    echo "  copied config.pickle from ${SRC_RUN_DIR}"
  fi
done < <(parse_seed_list "${SEEDS}")

if [[ "${copied_seeds}" -eq 0 ]]; then
  echo "ERROR: no seed directories were copied."
  exit 1
fi

if [[ "${copied_config}" -eq 0 ]]; then
  echo "ERROR: could not find config.pickle in any source run."
  exit 1
fi

echo ""
echo "Done."
echo "Combined run directory: ${DEST_DIR}"
echo "Copied seeds: ${copied_seeds}"
