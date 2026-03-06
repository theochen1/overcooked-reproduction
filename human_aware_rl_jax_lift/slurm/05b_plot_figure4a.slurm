#!/bin/bash
#SBATCH --job-name=oc_fig4a_plot
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fig4a_plot_%j.out
#SBATCH --error=logs/fig4a_plot_%j.err


module load miniforge 2>/dev/null || true
conda activate overcooked 2>/dev/null || true


export PYTHONUNBUFFERED=1


cd "$SLURM_SUBMIT_DIR/.."


python experiments/prepare_results.py \
  --aggregate \
  --eval_dir experiments/eval_results \
  --out experiments/results_figure4a.json


# Plot BEST seed (max over seed index) instead of mean over seeds.
python experiments/figure4a.py \
  --results_path experiments/results_figure4a.json \
  --seed_agg best \
  --output experiments/figure_4a.png
