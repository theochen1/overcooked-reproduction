#!/bin/bash
# ============================================================================
# HPC Training Configuration
# ============================================================================
# This file contains shared configuration for all training scripts.
# Source this file at the beginning of each training script.
#
# SETUP: Update the following before running on your cluster:
#   1. PROJECT_ROOT - path to your cloned repo
#   2. setup_conda() - how to activate your conda environment
# ============================================================================

# Project paths - UPDATE THIS to your clone location
export PROJECT_ROOT="$HOME/home/overcooked_ai"
export HUMAN_AWARE_RL_DIR="${PROJECT_ROOT}/src/human_aware_rl"
export HPC_SCRIPTS_DIR="${PROJECT_ROOT}/hpc_scripts"
export LOGS_DIR="${HPC_SCRIPTS_DIR}/logs"

# Results directories (relative to HUMAN_AWARE_RL_DIR)
export RESULTS_DIR="${HUMAN_AWARE_RL_DIR}/results"
export BC_RESULTS_DIR="${HUMAN_AWARE_RL_DIR}/bc_runs"

# Conda environment setup
# Directly prepend the conda env bin to PATH.
# This bypasses 'module load' and 'conda activate' which don't work on compute nodes.
setup_conda() {
    export PATH="$HOME/.conda/envs/overcooked/bin:$PATH"
    export CONDA_DEFAULT_ENV="overcooked"
}

# Call setup automatically when sourced
setup_conda

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Navigate to working directory
cd "${HUMAN_AWARE_RL_DIR}"

# Layouts (paper layouts)
LAYOUTS=(
    "cramped_room"
    "asymmetric_advantages"
    "coordination_ring"
    "forced_coordination"
    "counter_circuit"
)

# Seeds (paper seeds)
SEEDS=(0 10 20 30 40)

# SLURM defaults
# Use mit_normal only — older partitions (sched_mit_hill, newnodes) have
# glibc 2.17 which is too old for JAX/PyTorch dependencies.
export SLURM_PARTITION="mit_normal"
export SLURM_TIME="12:00:00"
export SLURM_MEM="32G"
export SLURM_CPUS="8"

# Thread parallelism — ensure JAX/numpy/XLA use all allocated CPUs
export OMP_NUM_THREADS="${SLURM_CPUS}"
export MKL_NUM_THREADS="${SLURM_CPUS}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS}"

# JAX-specific settings
export JAX_PLATFORM_NAME="cpu"
export JAX_ENABLE_X64="0"

# Logging
log_start() {
    echo "============================================================================"
    echo "Job: ${SLURM_JOB_NAME:-local}"
    echo "Job ID: ${SLURM_JOB_ID:-N/A}"
    echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
    echo "Start time: $(date)"
    echo "Working directory: $(pwd)"
    echo "Python: $(which python)"
    echo "============================================================================"
}

log_end() {
    echo "============================================================================"
    echo "End time: $(date)"
    echo "Exit code: $1"
    echo "============================================================================"
}

# Export functions for use in scripts
export -f setup_conda log_start log_end
