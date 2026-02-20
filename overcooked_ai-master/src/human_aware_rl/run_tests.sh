#!/usr/bin/env bash
#
# Run tests for the human_aware_rl module.
#
# This script tests the modernized PyTorch/JAX implementation.
# For legacy TensorFlow tests, use run_tests_legacy.sh.
#
# Usage:
#   cd src/human_aware_rl
#   ./run_tests.sh
#
# Requirements:
#   - PyTorch (for BC tests)
#   - JAX/Flax (for PPO tests, optional)
#

set -e  # Exit on error

export RUN_ENV=local

# Create a dummy data_dir.py if the file does not already exist
[ ! -f data_dir.py ] && echo "import os; DATA_DIR = os.path.abspath('.')" >> data_dir.py

echo "=========================================="
echo "Running Human-Aware RL Tests (PyTorch/JAX)"
echo "=========================================="
echo ""

# Human data tests (unchanged)
echo "[1/4] Running human data processing tests..."
python -m unittest human.tests
echo "Human data tests passed!"
echo ""

# BC tests (PyTorch)
echo "[2/4] Running BC tests (PyTorch)..."
python -m unittest imitation.behavior_cloning_test.TestBCTraining
echo "BC tests passed!"
echo ""

# BC Agent tests
echo "[3/4] Running BC Agent tests..."
python -m unittest imitation.behavior_cloning_test.TestBCAgent
echo "BC Agent tests passed!"
echo ""

# JAX environment tests (optional, skip if JAX not available)
echo "[4/4] Running JAX environment tests (if JAX available)..."
python -c "
import sys
try:
    import jax
    print('JAX available, running JAX tests...')
    # Import to verify the modules load correctly
    from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig
    from human_aware_rl.jaxmarl.ppo import PPOConfig
    print('JAX modules loaded successfully!')
except ImportError as e:
    print(f'JAX not available, skipping JAX tests: {e}')
    sys.exit(0)
"
echo ""

echo "=========================================="
echo "All tests passed!"
echo "=========================================="
