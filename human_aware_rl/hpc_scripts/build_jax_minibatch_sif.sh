#!/bin/bash
set -euo pipefail

# Build helper for JAX minibatch comparison Apptainer image.
# Usage:
#   bash hpc_scripts/build_jax_minibatch_sif.sh /path/to/jax_minibatch.sif

OUT_SIF="${1:-$HOME/containers/jax_minibatch.sif}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/jax_minibatch_apptainer.def"

mkdir -p "$(dirname "${OUT_SIF}")"

echo "==============================================================="
echo "Building JAX minibatch Apptainer image"
echo "Definition: ${DEF_FILE}"
echo "Output SIF: ${OUT_SIF}"
echo "Start: $(date)"
echo "==============================================================="

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
