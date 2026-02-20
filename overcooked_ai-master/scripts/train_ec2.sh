#!/bin/bash
# ============================================================================
# EC2 Training Script - Paper Reproduction with JAX
# ============================================================================
# This script trains PPO self-play agents using the JAX implementation
# that matches the original 2019 paper results.
#
# Layouts:
# - random0_legacy (forced_coordination)
# - random3_legacy
#
# Environment variables (set these before running):
#   PEM_KEY   - Path to your AWS PEM key file
#   EC2_HOST  - EC2 instance hostname or IP address
#   EC2_USER  - EC2 user (default: ec2-user)
# ============================================================================

set -e  # Exit on error

# EC2 Configuration (use environment variables with defaults)
PEM_KEY="${PEM_KEY:-$HOME/.ssh/overcooked.pem}"
EC2_USER="${EC2_USER:-ec2-user}"
EC2_HOST="${EC2_HOST:-}"

# Validate required variables
if [ -z "$EC2_HOST" ]; then
    echo "Error: EC2_HOST is not set"
    echo "Usage: export EC2_HOST=<your-ec2-ip> && ./train_ec2.sh"
    exit 1
fi

# Training Configuration
SEEDS="0 10 20 30 40"  # 5 seeds as per paper
RESULTS_DIR="results/paper_reproduction"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}EC2 Training Script - Paper Reproduction${NC}"
echo -e "${YELLOW}Using JAX implementation with legacy encoding${NC}"
echo -e "${YELLOW}Layouts: random0_legacy, random3_legacy${NC}"
echo -e "${YELLOW}============================================${NC}"

# Check if PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo -e "${RED}Error: PEM key not found at $PEM_KEY${NC}"
    exit 1
fi

chmod 400 "$PEM_KEY"

echo -e "\n${GREEN}Starting training on EC2...${NC}"
echo -e "This will run in the background on EC2."
echo -e "You can disconnect and training will continue."

# Create the training script on EC2
ssh -i "$PEM_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" << 'ENDSSH'
# Create training script
cat > ~/run_paper_reproduction.sh << 'TRAINING_SCRIPT'
#!/bin/bash
# ============================================================================
# Paper Reproduction Training - JAX Implementation
# ============================================================================
# Trains PPO-SP using legacy 20-channel encoding and original hyperparameters
# ============================================================================

set -e

cd ~/overcooked_ai-master
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Use python3 (standard on EC2/Amazon Linux)
PYTHON=$(which python3 || which python)
echo "Using Python: $PYTHON"
$PYTHON --version

RESULTS_DIR="results/paper_reproduction"
LOG_DIR="logs/paper_reproduction"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================"
echo "Paper Reproduction Training"
echo "============================================"
echo "Layouts: random0_legacy (forced_coordination)"
echo "         random3_legacy (counter_circuit)"
echo "Seeds: 0, 10, 20, 30, 40"
echo "Results: $RESULTS_DIR"
echo "Started: $(date)"
echo ""
echo "Key settings (matching original paper):"
echo "  - Legacy 20-channel observation encoding"
echo "  - Per-minibatch advantage normalization"
echo "  - Learning rate: 1e-3 (constant)"
echo "  - Entropy coef: 0.1 (fixed)"
echo "  - Clip epsilon: 0.05"
echo "  - VF coef: 0.1"
echo "  - Max grad norm: 0.1"
echo "  - GAE lambda: 0.98"
echo "============================================"

# Function to train a layout using JAX paper reproduction config
train_layout() {
    local layout=$1
    local seed=$2
    
    echo ""
    echo "============================================"
    echo "Training: $layout (seed=$seed)"
    echo "Time: $(date)"
    echo "============================================"
    
    $PYTHON -m human_aware_rl.jaxmarl.train_paper_reproduction \
        --layout "$layout" \
        --seed "$seed" \
        --timesteps 5000000 \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$LOG_DIR/${layout}_seed${seed}.log"
    
    echo "Completed: $layout seed $seed at $(date)"
}

# Train random0_legacy (forced_coordination) with all seeds
echo ""
echo "=========================================="
echo "PHASE 1: Training random0_legacy (forced_coordination)"
echo "=========================================="
for seed in 0 10 20 30 40; do
    train_layout "random0_legacy" $seed
done

# Train random3_legacy (counter_circuit) with all seeds
echo ""
echo "=========================================="
echo "PHASE 2: Training random3_legacy (counter_circuit)"
echo "=========================================="
for seed in 0 10 20 30 40; do
    train_layout "random3_legacy" $seed
done

echo ""
echo "============================================"
echo "PHASE 3: Evaluation Summary"
echo "============================================"

# Run evaluation
$PYTHON << 'EVAL_SCRIPT'
import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

os.chdir(os.path.expanduser("~/overcooked_ai-master"))
import sys
sys.path.insert(0, "src")

results_dir = Path("results/paper_reproduction")
eval_results = {}

layouts = {
    "random0_legacy": "forced_coordination",
    "random3_legacy": "counter_circuit"
}
seeds = [0, 10, 20, 30, 40]

print("\nCollecting training results...")
print("="*50)

for layout_name, display_name in layouts.items():
    print(f"\n{display_name} ({layout_name}):")
    layout_results = {"seeds": {}, "mean": 0, "std": 0}
    all_rewards = []
    
    for seed in seeds:
        run_dir = results_dir / f"ppo_sp_{layout_name}_seed{seed}"
        
        if not run_dir.exists():
            print(f"  Seed {seed}: Not found at {run_dir}")
            continue
        
        # Look for checkpoint directories to find final reward
        checkpoints = sorted(run_dir.glob("checkpoint_*"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            config_path = latest_checkpoint / "config.pkl"
            
            if config_path.exists():
                # Try to find metrics from training
                pass
        
        # Also check for a summary file if we saved one
        summary_path = run_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            final_reward = summary.get("final_mean_reward", 0)
            layout_results["seeds"][seed] = final_reward
            all_rewards.append(final_reward)
            print(f"  Seed {seed}: {final_reward:.2f}")
        else:
            print(f"  Seed {seed}: No summary found (training may still be in progress)")
    
    if all_rewards:
        layout_results["mean"] = float(np.mean(all_rewards))
        layout_results["std"] = float(np.std(all_rewards))
        layout_results["se"] = float(np.std(all_rewards) / np.sqrt(len(all_rewards)))
    
    eval_results[display_name] = layout_results

# Save evaluation results
eval_path = results_dir / "evaluation_results.json"
with open(eval_path, "w") as f:
    json.dump(eval_results, f, indent=2)

print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
for layout, results in eval_results.items():
    print(f"\n{layout}:")
    if results["seeds"]:
        print(f"  Mean: {results['mean']:.2f} ± {results['std']:.2f}")
        for seed, reward in results["seeds"].items():
            print(f"  Seed {seed}: {reward:.2f}")
    else:
        print("  No results yet")

print("\n" + "="*50)
print("EXPECTED RESULTS (matching original paper):")
print("="*50)
print("forced_coordination: Training reward ~120-160")
print("counter_circuit: Training reward ~120-160")
print("")
print(f"Results saved to: {eval_path}")
EVAL_SCRIPT

echo ""
echo "============================================"
echo "TRAINING COMPLETE"
echo "============================================"
echo "Finished: $(date)"
echo "Results: $RESULTS_DIR"
echo "Logs: $LOG_DIR"
echo ""
echo "To download results:"
echo "  scp -i /path/to/key -r ec2-user@host:~/overcooked_ai-master/results/paper_reproduction ."

TRAINING_SCRIPT

chmod +x ~/run_paper_reproduction.sh
echo "Training script created at ~/run_paper_reproduction.sh"
ENDSSH

echo -e "\n${GREEN}Training script created on EC2.${NC}"
echo -e "\n${CYAN}Choose how to run:${NC}"
echo -e "\n${YELLOW}Option 1: Run in foreground (will stop if you disconnect)${NC}"
echo -e "  ssh -i $PEM_KEY $EC2_USER@$EC2_HOST './run_paper_reproduction.sh'"
echo -e "\n${YELLOW}Option 2: Run in background with nohup (recommended)${NC}"
echo -e "  ssh -i $PEM_KEY $EC2_USER@$EC2_HOST 'nohup ./run_paper_reproduction.sh > paper_reproduction.log 2>&1 &'"
echo -e "\n${YELLOW}Option 3: Run in tmux (allows reattaching)${NC}"
echo -e "  ssh -i $PEM_KEY $EC2_USER@$EC2_HOST 'tmux new-session -d -s training \"./run_paper_reproduction.sh\"'"
echo -e "\n${GREEN}To monitor training:${NC}"
echo -e "  ssh -i $PEM_KEY $EC2_USER@$EC2_HOST 'tail -f paper_reproduction.log'"
echo -e "  ssh -i $PEM_KEY $EC2_USER@$EC2_HOST 'tmux attach -t training'"

# Ask user which option to use
echo -e "\n${YELLOW}Would you like to start training now? [y/N]${NC}"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "\n${GREEN}Starting training in background with nohup...${NC}"
    ssh -i "$PEM_KEY" "$EC2_USER@$EC2_HOST" 'nohup ./run_paper_reproduction.sh > paper_reproduction.log 2>&1 &'
    
    echo -e "\n${GREEN}Training started!${NC}"
    echo -e "Monitor with: ssh -i $PEM_KEY $EC2_USER@$EC2_HOST 'tail -f paper_reproduction.log'"
else
    echo -e "\n${YELLOW}Training not started. Run manually using options above.${NC}"
fi

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}To download results when complete:${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "scp -i $PEM_KEY -r $EC2_USER@$EC2_HOST:~/overcooked_ai-master/results/paper_reproduction ."
