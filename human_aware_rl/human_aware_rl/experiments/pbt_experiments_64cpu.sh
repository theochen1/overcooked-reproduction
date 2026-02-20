#!/bin/bash
# ==============================================================================
# PBT Experiments - Optimized for 64 vCPU (c7i.16xlarge)
# ==============================================================================
# Uses sim_threads=60 to leverage all available CPUs
# ==============================================================================

# Change to the correct directory
cd /workspace/human_aware_rl/human_aware_rl

# Simple layout
python pbt/pbt.py with \
    fixed_mdp \
    layout_name="simple" \
    EX_NAME="pbt_simple" \
    TOTAL_STEPS_PER_AGENT=8e6 \
    REW_SHAPING_HORIZON=3e6 \
    LR=2e-3 \
    POPULATION_SIZE=3 \
    SEEDS="[8015, 3554, 581, 5608, 4221]" \
    NUM_SELECTION_GAMES=6 \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Unident_s layout
python pbt/pbt.py with \
    fixed_mdp \
    layout_name="unident_s" \
    EX_NAME="pbt_unident_s" \
    TOTAL_STEPS_PER_AGENT=1.1e7 \
    REW_SHAPING_HORIZON=5e6 \
    LR=8e-4 \
    POPULATION_SIZE=3 \
    SEEDS="[8015, 3554, 581, 5608, 4221]" \
    NUM_SELECTION_GAMES=6 \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random1 layout
python pbt/pbt.py with \
    fixed_mdp \
    layout_name="random1" \
    EX_NAME="pbt_random1" \
    TOTAL_STEPS_PER_AGENT=5e6 \
    REW_SHAPING_HORIZON=4e6 \
    LR=8e-4 \
    POPULATION_SIZE=3 \
    SEEDS="[8015, 3554, 581, 5608, 4221]" \
    NUM_SELECTION_GAMES=6 \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random0 layout
python pbt/pbt.py with \
    fixed_mdp \
    layout_name="random0" \
    EX_NAME="pbt_random0" \
    TOTAL_STEPS_PER_AGENT=8e6 \
    REW_SHAPING_HORIZON=7e6 \
    LR=3e-3 \
    POPULATION_SIZE=3 \
    SEEDS="[8015, 3554, 581, 5608, 4221]" \
    NUM_SELECTION_GAMES=6 \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

# Random3 layout
python pbt/pbt.py with \
    fixed_mdp \
    layout_name="random3" \
    EX_NAME="pbt_random3" \
    TOTAL_STEPS_PER_AGENT=6e6 \
    REW_SHAPING_HORIZON=4e6 \
    LR=1e-3 \
    POPULATION_SIZE=3 \
    SEEDS="[8015, 3554, 581, 5608, 4221]" \
    NUM_SELECTION_GAMES=6 \
    VF_COEF=0.5 \
    MINIBATCHES=10 \
    TIMESTAMP_DIR=False \
    sim_threads=60 \
    TOTAL_BATCH_SIZE=24000

echo "All PBT experiments complete!"

