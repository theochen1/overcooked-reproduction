#!/usr/bin/env bash
set -euo pipefail

# Slurm-friendly Figure 4a evaluation pipeline (JAX-native; no Docker).
#
# Usage:
#   cd human_aware_rl_jax_lift/experiments
#   bash run_eval_figure4a.sh
#
# Optional env-var overrides:
#   NUM_GAMES=50    rollouts per (condition × seed) [default: 100]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs experiments/eval_results

NUM_GAMES="${NUM_GAMES:-100}"

echo "Submitting Figure 4a eval array..."
jid_eval=$(sbatch --parsable \
  --export=ALL,NUM_GAMES="$NUM_GAMES" \
  slurm/05_eval_figure4a.slurm)

echo "Submitted eval job array: $jid_eval"

echo "Submitting aggregation+plot job (afterok)..."
jid_plot=$(sbatch --parsable \
  --dependency=afterok:"$jid_eval" \
  --export=ALL \
  slurm/05b_plot_figure4a.slurm)

echo "Submitted plot job: $jid_plot"

echo "Monitor with: squeue -u $USER"
