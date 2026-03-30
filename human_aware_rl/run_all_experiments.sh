#!/bin/bash
# ==============================================================================
# Run All Experiments Script
# ==============================================================================
# This script runs all experiments in sequence to reproduce paper results.
# Run this INSIDE the Docker container.
# 
# Usage: bash run_all_experiments.sh [experiment_type]
# 
# experiment_type can be:
#   all     - Run all experiments (default)
#   bc      - Only BC experiments
#   ppo_sp  - Only PPO self-play
#   pbt     - Only PBT
#   ppo_bc  - Only PPO with BC
#   ppo_hm  - Only PPO with human model
#   quick   - Quick test run (1 seed, 1 layout)
# ==============================================================================

set -e

EXPERIMENT_TYPE="${1:-all}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_start() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting: $1${NC}"
    echo -e "${BLUE}Time: $(date)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

log_complete() {
    echo ""
    echo -e "${GREEN}✓ Completed: $1${NC}"
    echo -e "${GREEN}Time: $(date)${NC}"
    echo ""
}

cd /workspace/human_aware_rl/human_aware_rl

case $EXPERIMENT_TYPE in
    "bc")
        log_start "Behavioral Cloning Experiments"
        python experiments/bc_experiments.py
        log_complete "Behavioral Cloning Experiments"
        ;;
        
    "ppo_sp")
        log_start "PPO Self-Play Experiments (64 CPU optimized)"
        bash experiments/ppo_sp_experiments_64cpu.sh
        log_complete "PPO Self-Play Experiments"
        ;;
        
    "pbt")
        log_start "Population Based Training Experiments (64 CPU optimized)"
        bash experiments/pbt_experiments_64cpu.sh
        log_complete "Population Based Training Experiments"
        ;;
        
    "ppo_bc")
        log_start "PPO with BC Partner Experiments (64 CPU optimized)"
        bash experiments/ppo_bc_experiments_64cpu.sh
        log_complete "PPO with BC Partner Experiments"
        ;;
        
    "ppo_hm")
        log_start "PPO with Human Model Experiments (64 CPU optimized)"
        bash experiments/ppo_hm_experiments_64cpu.sh
        log_complete "PPO with Human Model Experiments"
        ;;
        
    "quick")
        log_start "Quick Test Run"
        
        echo -e "${YELLOW}Running quick PPO test (simple layout, 1 seed, reduced steps)...${NC}"
        python ppo/ppo.py with \
            EX_NAME="quick_test" \
            layout_name="simple" \
            REW_SHAPING_HORIZON=1e5 \
            LR=1e-3 \
            PPO_RUN_TOT_TIMESTEPS=5e4 \
            OTHER_AGENT_TYPE="sp" \
            SEEDS="[2229]" \
            VF_COEF=1 \
            TIMESTAMP_DIR=True \
            LOCAL_TESTING=True
        
        log_complete "Quick Test Run"
        echo -e "${GREEN}Quick test passed! The setup is working correctly.${NC}"
        ;;
        
    "all")
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Running ALL Experiments (64 CPU optimized)${NC}"
        echo -e "${GREEN}This will take several days to complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        
        # BC must run first - other experiments depend on BC models
        log_start "Phase 1: Behavioral Cloning"
        python experiments/bc_experiments.py
        log_complete "Phase 1: Behavioral Cloning"
        
        # PPO-SP can run independently
        log_start "Phase 2: PPO Self-Play (64 CPU)"
        bash experiments/ppo_sp_experiments_64cpu.sh
        log_complete "Phase 2: PPO Self-Play"
        
        # PBT can run independently  
        log_start "Phase 3: Population Based Training (64 CPU)"
        bash experiments/pbt_experiments_64cpu.sh
        log_complete "Phase 3: Population Based Training"
        
        # PPO-BC requires BC models
        log_start "Phase 4: PPO with BC Partner (64 CPU)"
        bash experiments/ppo_bc_experiments_64cpu.sh
        log_complete "Phase 4: PPO with BC Partner"
        
        # PPO-HM can run independently
        log_start "Phase 5: PPO with Human Model (64 CPU)"
        bash experiments/ppo_hm_experiments_64cpu.sh
        log_complete "Phase 5: PPO with Human Model"
        
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}ALL EXPERIMENTS COMPLETE!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        ;;
        
    *)
        echo -e "${RED}Unknown experiment type: $EXPERIMENT_TYPE${NC}"
        echo ""
        echo "Usage: bash run_all_experiments.sh [experiment_type]"
        echo ""
        echo "experiment_type can be:"
        echo "  all     - Run all experiments (default)"
        echo "  bc      - Only BC experiments"
        echo "  ppo_sp  - Only PPO self-play"
        echo "  pbt     - Only PBT"
        echo "  ppo_bc  - Only PPO with BC"
        echo "  ppo_hm  - Only PPO with human model"
        echo "  quick   - Quick test run"
        exit 1
        ;;
esac

