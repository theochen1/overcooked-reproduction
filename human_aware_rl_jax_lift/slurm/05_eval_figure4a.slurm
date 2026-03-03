#!/bin/bash
#SBATCH --job-name=oc_fig4a_eval
#SBATCH --array=0-4%2
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fig4a_eval_%A_%a.out
#SBATCH --error=logs/fig4a_eval_%A_%a.err

LAYOUTS=(simple unident_s random1 random0 random3)
LAYOUT=${LAYOUTS[$SLURM_ARRAY_TASK_ID]}
NUM_GAMES=${NUM_GAMES:-100}

module load miniforge 2>/dev/null || true
conda activate overcooked 2>/dev/null || true

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR/.."
# Use same layout dir as training so PPO CNN input shape matches checkpoints
export OVERCOOKED_LAYOUT_DIR="$(pwd)/../human_aware_rl/overcooked_ai/overcooked_ai_py/data/layouts"

python -u experiments/eval_figure4a_inner.py \
  --layout "$LAYOUT" \
  --num_games "$NUM_GAMES" \
  --ppo_runs_dir data/ppo_runs \
  --bc_paths_file data/bc_runs/best_bc_model_paths.pkl \
  --out_dir experiments/eval_results
