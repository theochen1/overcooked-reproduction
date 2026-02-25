#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_eval_figure4a.sh
#
# Runs Figure-4a evaluation for all 5 Overcooked layouts inside the
# human_aware_rl Docker container, then aggregates into results_figure4a.json.
#
# Full pipeline:
#   cd overcooked-reproduction/human_aware_rl_jax_lift/experiments
#   bash run_eval_figure4a.sh
#
# Optional env-var overrides:
#   NUM_GAMES=50      rollouts per (condition × seed)  [default: 100]
#   DOCKER_IMG=name   image tag to run                [default: human_aware_rl]
#   OUT_DIR=/path     host dir for per-layout JSONs   [default: ./eval_results]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOCKER_IMG="${DOCKER_IMG:-human_aware_rl}"
NUM_GAMES="${NUM_GAMES:-100}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/eval_results}"
LAYOUTS=(simple unident_s random0 random1 random3)

echo "=================================================="
echo "  Figure-4a Evaluation Pipeline"
echo "=================================================="
echo "  DOCKER_IMG  :  ${DOCKER_IMG}"
echo "  NUM_GAMES   :  ${NUM_GAMES}"
echo "  OUT_DIR     :  ${OUT_DIR}"
echo "  REPO_ROOT   :  ${REPO_ROOT}"
echo ""

mkdir -p "${OUT_DIR}"

# ── [1/3] Build Docker image if not already present ──────────────────────────
if ! docker image inspect "${DOCKER_IMG}" &>/dev/null; then
    echo "[1/3] Building Docker image '${DOCKER_IMG}' from ${REPO_ROOT}/human_aware_rl ..."
    docker build -t "${DOCKER_IMG}" "${REPO_ROOT}/human_aware_rl"
else
    echo "[1/3] Docker image '${DOCKER_IMG}' already present – skipping build."
fi

# ── [2/3] Stage inner eval script so the container can reach it ──────────────
#
# The container working dir is /code/human_aware_rl (mounted from the host
# human_aware_rl/ subtree).  We copy eval_figure4a_inner.py there temporarily.
INNER_SRC="${SCRIPT_DIR}/eval_figure4a_inner.py"
INNER_DST="${REPO_ROOT}/human_aware_rl/human_aware_rl/eval_figure4a_inner.py"
cp "${INNER_SRC}" "${INNER_DST}"
# Always remove the staged copy on exit (success or error).
trap 'echo "Cleaning up staged eval script ..."; rm -f "${INNER_DST}"' EXIT

echo "[2/3] Running evaluations inside Docker ..."
for layout in "${LAYOUTS[@]}"; do
    echo ""
    echo "  ── layout: ${layout} ──"

    docker run --rm \
        --name "eval_fig4a_${layout}" \
        -v "${REPO_ROOT}/human_aware_rl:/code" \
        -v "${OUT_DIR}:/results" \
        -w /code/human_aware_rl \
        "${DOCKER_IMG}" \
        python eval_figure4a_inner.py \
            --layout    "${layout}" \
            --num_games "${NUM_GAMES}" \
            --out_dir   /results

    echo "  ✓ ${layout} → ${OUT_DIR}/results_${layout}.json"
done

# ── [3/3] Aggregate per-layout fragments into one JSON ───────────────────────
echo ""
echo "[3/3] Aggregating results ..."
python "${SCRIPT_DIR}/prepare_results.py" \
    --aggregate \
    --eval_dir  "${OUT_DIR}" \
    --out       "${SCRIPT_DIR}/results_figure4a.json"

echo ""
echo "✓  Pipeline complete."
echo ""
echo "   Next step:"
echo "     cd ${SCRIPT_DIR}"
echo "     python figure4a.py \\"
echo "         --results_path results_figure4a.json \\"
echo "         --output figure_4a.png"
