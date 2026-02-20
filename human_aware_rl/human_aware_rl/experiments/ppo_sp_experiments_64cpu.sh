#!/bin/bash
# ==============================================================================
# PPO Self-Play Experiments - Optimized for 64 vCPU (c7i.16xlarge)
# ==============================================================================
# Uses sim_threads=60 to leverage all available CPUs
# Increased TOTAL_BATCH_SIZE for better throughput
# ==============================================================================

# Change to the correct directory
cd /workspace/human_aware_rl/human_aware_rl

# Simple layout
python ppo/ppo.py with \
    EX_NAME="ppo_sp_simple" \
    layout_name="simple" \
    REW_SHAPING_HORIZON=2.5e6 \
    LR=1e-3 \
    PPO_RUN_TOT_TIMESTEPS=6e6 \
    OTHER_AGENT_TYPE="sp" \
    SEEDS="[2229, 7649, 7225, 9807, 386]" \
    VF_COEF=1 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Unident_s layout
python ppo/ppo.py with \
    EX_NAME="ppo_sp_unident_s" \
    layout_name="unident_s" \
    REW_SHAPING_HORIZON=2.5e6 \
    PPO_RUN_TOT_TIMESTEPS=7e6 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="sp" \
    SEEDS="[2229, 7649, 7225, 9807, 386]" \
    VF_COEF=0.5 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random1 layout
python ppo/ppo.py with \
    EX_NAME="ppo_sp_random1" \
    layout_name="random1" \
    REW_SHAPING_HORIZON=3.5e6 \
    PPO_RUN_TOT_TIMESTEPS=1e7 \
    LR=6e-4 \
    OTHER_AGENT_TYPE="sp" \
    SEEDS="[2229, 7649, 7225, 9807, 386]" \
    VF_COEF=0.5 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random0 layout
python ppo/ppo.py with \
    EX_NAME="ppo_sp_random0" \
    layout_name="random0" \
    REW_SHAPING_HORIZON=2.5e6 \
    PPO_RUN_TOT_TIMESTEPS=7.5e6 \
    LR=8e-4 \
    OTHER_AGENT_TYPE="sp" \
    SEEDS="[2229, 7649, 7225, 9807, 386]" \
    VF_COEF=0.5 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random3 layout
python ppo/ppo.py with \
    EX_NAME="ppo_sp_random3" \
    layout_name="random3" \
    REW_SHAPING_HORIZON=2.5e6 \
    PPO_RUN_TOT_TIMESTEPS=8e6 \
    LR=8e-4 \
    OTHER_AGENT_TYPE="sp" \
    SEEDS="[2229, 7649, 7225, 9807, 386]" \
    VF_COEF=0.5 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

echo "All PPO-SP experiments complete!"

