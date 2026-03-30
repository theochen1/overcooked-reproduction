#!/bin/bash
# ==============================================================================
# PPO with Human Model Experiments - Optimized for 64 vCPU (c7i.16xlarge)
# ==============================================================================
# Uses sim_threads=60 to leverage all available CPUs
# ==============================================================================

# Change to the correct directory
cd /workspace/human_aware_rl/human_aware_rl

# Simple layout
python ppo/ppo.py with \
    EX_NAME="ppo_hm_simple" \
    layout_name="simple" \
    REW_SHAPING_HORIZON=1e6 \
    PPO_RUN_TOT_TIMESTEPS=8e6 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="hm" \
    HM_PARAMS="[True, 1.75, True, 1.7]" \
    SEEDS="[8355, 5748, 1352, 3325, 8611]" \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[1e2, 1e4]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Unident_s layout
python ppo/ppo.py with \
    EX_NAME="ppo_hm_unident_s" \
    layout_name="unident_s" \
    REW_SHAPING_HORIZON=6e6 \
    PPO_RUN_TOT_TIMESTEPS=1e7 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="hm" \
    HM_PARAMS="[True, 1.3, True, 1.1]" \
    SEEDS="[8355, 5748, 1352, 3325, 8611]" \
    VF_COEF=0.5 \
    MINIBATCHES=12 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[1e6, 7e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random1 layout
python ppo/ppo.py with \
    EX_NAME="ppo_hm_random1" \
    layout_name="random1" \
    REW_SHAPING_HORIZON=5e6 \
    PPO_RUN_TOT_TIMESTEPS=1.6e7 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="hm" \
    HM_PARAMS="[True, 2, True, 1.8]" \
    SEEDS="[8355, 5748, 1352, 3325, 8611]" \
    VF_COEF=0.5 \
    MINIBATCHES=15 \
    LR_ANNEALING=1.5 \
    SELF_PLAY_HORIZON="[2e6, 6e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random3 layout
python ppo/ppo.py with \
    EX_NAME="ppo_hm_random3" \
    layout_name="random3" \
    REW_SHAPING_HORIZON=4e6 \
    PPO_RUN_TOT_TIMESTEPS=1.2e7 \
    LR=1.5e-3 \
    OTHER_AGENT_TYPE="hm" \
    HM_PARAMS="[True, 2.2, True, 2]" \
    SEEDS="[8355, 5748, 1352, 3325, 8611]" \
    VF_COEF=0.1 \
    MINIBATCHES=15 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[1e6, 4e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

echo "All PPO-HM experiments complete!"

