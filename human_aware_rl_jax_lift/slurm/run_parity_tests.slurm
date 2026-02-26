#!/bin/bash
#SBATCH --job-name=oc_parity_tests
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/parity_tests_%j.out
#SBATCH --error=logs/parity_tests_%j.err

set -euo pipefail

echo "Node: $(hostname) | Start: $(date)"

module load cuda/12.1 2>/dev/null || true
module load miniforge
conda activate overcooked

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="/home/tchen22/.cache/jax_compile_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

python -c "
import jax
devs = jax.devices()
print('JAX devices:', devs)
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), \
    f'No GPU found. Devices: {devs}'
print('GPU check passed.')
" || exit 1

cd "$SLURM_SUBMIT_DIR/.."

echo ""
echo "======================================================"
echo "Running parity test suite"
echo "======================================================"

python -m pytest \
    tests/test_encoding_parity.py::test_start_state_encoding_matches_legacy_reset \
    tests/test_decisive_parity.py::test_multi_episode_reset_parity \
    tests/test_decisive_parity.py::test_reward_pipeline_rollout_and_gae_parity \
    tests/test_decisive_parity.py::test_step_parity_random_sequence \
    -v -s 2>&1

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"
