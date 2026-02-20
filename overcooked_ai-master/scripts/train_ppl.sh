#!/bin/bash
#SBATCH --job-name=train_ppl
#SBATCH --output=ppl_train_%j.out
#SBATCH --error=ppl_train_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Train PPL-based models for Overcooked AI
# Usage: sbatch train_ppl.sh [layout] [model]
# Examples:
#   sbatch train_ppl.sh                      # Train all models on cramped_room
#   sbatch train_ppl.sh cramped_room         # Train all models on cramped_room
#   sbatch train_ppl.sh cramped_room bayesian_bc  # Train only Bayesian BC

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate environment
# source ~/envs/overcooked/bin/activate

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
LAYOUT="${1:-cramped_room}"
MODEL="${2:-all}"

echo "=========================================="
echo "PPL Model Training"
echo "=========================================="
echo "Layout: $LAYOUT"
echo "Model: $MODEL"
echo "Start time: $(date)"
echo ""

# Install PPL dependencies if needed
pip install pyro-ppl numpyro --quiet

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run training
if [ "$MODEL" == "all" ]; then
    python -m human_aware_rl.ppl.train_ppl \
        --layout "$LAYOUT" \
        --all_models \
        --num_epochs 100 \
        --batch_size 64 \
        --verbose
else
    python -m human_aware_rl.ppl.train_ppl \
        --layout "$LAYOUT" \
        --model "$MODEL" \
        --num_epochs 100 \
        --batch_size 64 \
        --verbose
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="
