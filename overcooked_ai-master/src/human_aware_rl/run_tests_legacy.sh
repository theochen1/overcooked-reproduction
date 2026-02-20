#!/usr/bin/env bash
#
# Run legacy tests for the human_aware_rl module (TensorFlow/RLlib).
#
# WARNING: This script uses deprecated TensorFlow/RLlib dependencies.
# For the modernized PyTorch/JAX tests, use run_tests.sh.
#
# Usage:
#   cd src/human_aware_rl
#   ./run_tests_legacy.sh
#
# Requirements:
#   - TensorFlow
#   - Ray/RLlib
#   - Install with: pip install ".[harl-legacy]"
#

set -e  # Exit on error

export RUN_ENV=local

# Create a dummy data_dir.py if the file does not already exist
[ ! -f data_dir.py ] && echo "import os; DATA_DIR = os.path.abspath('.')" >> data_dir.py

echo "=========================================="
echo "Running Legacy Tests (TensorFlow/RLlib)"
echo "WARNING: These tests use deprecated dependencies"
echo "=========================================="
echo ""

# Human data tests
echo "[1/4] Running human data processing tests..."
python -m unittest human.tests
echo "Human data tests passed!"
echo ""

# BC tests (TensorFlow - legacy)
echo "[2/4] Running BC tests (TensorFlow - legacy)..."
python -m unittest imitation.behavior_cloning_tf2_test.TestBCTraining
echo "BC tests passed!"
echo ""

# rllib tests (legacy)
echo "[3/4] Running rllib tests (legacy)..."
python -m unittest rllib.tests
echo "rllib tests passed!"
echo ""

# PPO tests (legacy)
echo "[4/4] Running PPO tests (legacy)..."
python -m unittest ppo.ppo_rllib_test
echo "PPO tests passed!"
echo ""

echo "=========================================="
echo "All legacy tests passed!"
echo "=========================================="

