#!/bin/bash
#SBATCH --job-name=ppl_all
#SBATCH --output=ppl_all_%j.out
#SBATCH --error=ppl_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Train all PPL models on all layouts
# Usage: sbatch train_ppl_all_layouts.sh

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate environment
# source ~/envs/overcooked/bin/activate

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "PPL Model Training - All Layouts"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Install PPL dependencies if needed
pip install pyro-ppl numpyro --quiet

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Run training on all layouts
python -m human_aware_rl.ppl.train_ppl \
    --all_layouts \
    --all_models \
    --num_epochs 100 \
    --batch_size 64 \
    --verbose

# Run evaluation
echo ""
echo "=========================================="
echo "Running Evaluation"
echo "=========================================="

python -m human_aware_rl.ppl.evaluate_ppl \
    --all_layouts \
    --output "$PROJECT_ROOT/eval_results/ppl_eval_results.json"

# Compare with baselines
python -m human_aware_rl.ppl.compare_with_baselines \
    --all_layouts \
    --markdown \
    --output "$PROJECT_ROOT/eval_results/ppl_comparison.json"

echo ""
echo "=========================================="
echo "All training and evaluation complete!"
echo "End time: $(date)"
echo "=========================================="
