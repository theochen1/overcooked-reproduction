#!/bin/bash
# ==============================================================================
# PPO with BC Partner Experiments - Optimized for 64 vCPU (c7i.16xlarge)
# ==============================================================================
# Uses sim_threads=60 to leverage all available CPUs
# ==============================================================================

# Change to the correct directory
cd /workspace/human_aware_rl/human_aware_rl

# Simple layout - train
python ppo/ppo.py with \
    EX_NAME="ppo_bc_train_simple" \
    layout_name="simple" \
    REW_SHAPING_HORIZON=1e6 \
    PPO_RUN_TOT_TIMESTEPS=8e6 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="bc_train" \
    SEEDS="[9456, 1887, 5578, 5987, 516]" \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[5e5, 3e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Simple layout - test
python ppo/ppo.py with \
    EX_NAME="ppo_bc_test_simple" \
    layout_name="simple" \
    REW_SHAPING_HORIZON=1e6 \
    PPO_RUN_TOT_TIMESTEPS=8e6 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="bc_test" \
    SEEDS="[2888, 7424, 7360, 4467, 184]" \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[5e5, 3e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Unident_s layout - train
python ppo/ppo.py with \
    EX_NAME="ppo_bc_train_unident_s" \
    layout_name="unident_s" \
    REW_SHAPING_HORIZON=6e6 \
    PPO_RUN_TOT_TIMESTEPS=1e7 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="bc_train" \
    SEEDS="[9456, 1887, 5578, 5987, 516]" \
    VF_COEF=0.5 \
    MINIBATCHES=12 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[1e6, 7e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Unident_s layout - test
python ppo/ppo.py with \
    EX_NAME="ppo_bc_test_unident_s" \
    layout_name="unident_s" \
    REW_SHAPING_HORIZON=6e6 \
    PPO_RUN_TOT_TIMESTEPS=1e7 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="bc_test" \
    SEEDS="[2888, 7424, 7360, 4467, 184]" \
    VF_COEF=0.5 \
    MINIBATCHES=12 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[1e6, 7e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random1 layout - train
python ppo/ppo.py with \
    EX_NAME="ppo_bc_train_random1" \
    layout_name="random1" \
    REW_SHAPING_HORIZON=5e6 \
    PPO_RUN_TOT_TIMESTEPS=1.6e7 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="bc_train" \
    SEEDS="[9456, 1887, 5578, 5987, 516]" \
    VF_COEF=0.5 \
    MINIBATCHES=15 \
    LR_ANNEALING=1.5 \
    SELF_PLAY_HORIZON="[2e6, 6e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random1 layout - test
python ppo/ppo.py with \
    EX_NAME="ppo_bc_test_random1" \
    layout_name="random1" \
    REW_SHAPING_HORIZON=5e6 \
    PPO_RUN_TOT_TIMESTEPS=1.6e7 \
    LR=1e-3 \
    OTHER_AGENT_TYPE="bc_test" \
    SEEDS="[2888, 7424, 7360, 4467, 184]" \
    VF_COEF=0.5 \
    MINIBATCHES=15 \
    LR_ANNEALING=1.5 \
    SELF_PLAY_HORIZON="[2e6, 6e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random0 layout - train
python ppo/ppo.py with \
    EX_NAME="ppo_bc_train_random0" \
    layout_name="random0" \
    REW_SHAPING_HORIZON=4e6 \
    PPO_RUN_TOT_TIMESTEPS=9e6 \
    LR=1.5e-3 \
    OTHER_AGENT_TYPE="bc_train" \
    SEEDS="[9456, 1887, 5578, 5987, 516]" \
    VF_COEF=0.1 \
    MINIBATCHES=15 \
    LR_ANNEALING=2 \
    SELF_PLAY_HORIZON=None \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random0 layout - test
python ppo/ppo.py with \
    EX_NAME="ppo_bc_test_random0" \
    layout_name="random0" \
    REW_SHAPING_HORIZON=4e6 \
    PPO_RUN_TOT_TIMESTEPS=9e6 \
    LR=1.5e-3 \
    OTHER_AGENT_TYPE="bc_test" \
    SEEDS="[2888, 7424, 7360, 4467, 184]" \
    VF_COEF=0.1 \
    MINIBATCHES=15 \
    LR_ANNEALING=2 \
    SELF_PLAY_HORIZON=None \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random3 layout - train
python ppo/ppo.py with \
    EX_NAME="ppo_bc_train_random3" \
    layout_name="random3" \
    REW_SHAPING_HORIZON=4e6 \
    PPO_RUN_TOT_TIMESTEPS=1.2e7 \
    LR=1.5e-3 \
    OTHER_AGENT_TYPE="bc_train" \
    SEEDS="[9456, 1887, 5578, 5987, 516]" \
    VF_COEF=0.1 \
    MINIBATCHES=15 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[1e6, 4e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random3 layout - test
python ppo/ppo.py with \
    EX_NAME="ppo_bc_test_random3" \
    layout_name="random3" \
    REW_SHAPING_HORIZON=4e6 \
    PPO_RUN_TOT_TIMESTEPS=1.2e7 \
    LR=1.5e-3 \
    OTHER_AGENT_TYPE="bc_test" \
    SEEDS="[2888, 7424, 7360, 4467, 184]" \
    VF_COEF=0.1 \
    MINIBATCHES=15 \
    LR_ANNEALING=3 \
    SELF_PLAY_HORIZON="[1e6, 4e6]" \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

echo "All PPO-BC experiments complete!"

