#!/bin/bash
set -euo pipefail

# Build helper for TF1 human_aware_rl Apptainer image.
# Usage:
#   bash hpc_scripts/build_harl_sif.sh /path/to/harl_tf1.sif

OUT_SIF="${1:-$HOME/containers/harl_tf1.sif}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/harl_apptainer.def"

mkdir -p "$(dirname "${OUT_SIF}")"

echo "==============================================================="
echo "Building HARL Apptainer image"
echo "Definition: ${DEF_FILE}"
echo "Output SIF: ${OUT_SIF}"
echo "Start: $(date)"
echo "==============================================================="

# On many clusters, mksquashfs default threading can fail with:
#   "FATAL ERROR: Failed to create thread"
# so we force single-thread squashfs.
(cd "${SCRIPT_DIR}" && \
  apptainer build \
    --fakeroot \
    --mksquashfs-args "-processors 1" \
    "${OUT_SIF}" \
    "${DEF_FILE}")

echo "==============================================================="
echo "Build complete: ${OUT_SIF}"
echo "End: $(date)"
echo "==============================================================="
