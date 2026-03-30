#!/bin/bash
# Full Overcooked reproduction suite — Engaging HPC
# Usage: bash slurm/submit_all.sh
# Run from repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "========================================================"
echo "  Overcooked Reproduction — Full SLURM Submission"
echo "  Repo:  $REPO_ROOT"
echo "  User:  $USER"
echo "  Start: $(date)"
echo "========================================================"

# Stage 1: BC training (CPU, preemptable — ~15 min, clears before any GPU dep)
echo ""
echo "[1/5] BC models — train+test splits, all 5 layouts (CPU, preemptable)"
BC_JOB=$(sbatch --parsable slurm/01_bc.slurm)
echo "      Job ID: $BC_JOB (10 array tasks)"

# Stage 1b: Figure 6 BC ablation (independent, different epoch counts)
echo "[1b]  Figure 6 BC ablation — layout-specific epochs (CPU, preemptable)"
FIG6_JOB=$(sbatch --parsable slurm/01b_bc_fig6.slurm)
echo "      Job ID: $FIG6_JOB (10 array tasks)"

# Stage 2: PPO-SP — no dependency, fills normal_gpu slots immediately
echo ""
echo "[2/5] PPO self-play — all 5 layouts, 5 seeds (normal_gpu)"
SP_JOB=$(sbatch --parsable slurm/02_ppo_sp.slurm)
echo "      Job ID: $SP_JOB (5 array tasks)"

# Stage 2b: PBT-SP — no BC dependency, fills preemptable slots immediately
echo "[2b]  PBT self-play — 5 layouts × 5 seeds (preemptable, per-seed array)"
PBT_SP_JOB=$(sbatch --parsable \
    --export=ALL,AGENT_TYPE=sp \
    slurm/04_pbt.slurm)
echo "      Job ID: $PBT_SP_JOB (25 array tasks)"

# Stage 3: PPO-BC — depends on BC completing
echo ""
echo "[3/5] PPO+BC partner — train+test splits (normal_gpu, dep on BC)"
PPOBC_JOB=$(sbatch --parsable \
    --dependency=afterok:${BC_JOB} \
    slurm/03_ppo_bc.slurm)
echo "      Job ID: $PPOBC_JOB (10 array tasks)"

# Stage 4: PBT-BC-train and PBT-BC-test — depend on BC
echo "[4/5] PBT+BC-train partner (preemptable, per-seed, dep on BC)"
PBT_BC_TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:${BC_JOB} \
    --export=ALL,AGENT_TYPE=bc_train \
    slurm/04_pbt.slurm)
echo "      Job ID: $PBT_BC_TRAIN_JOB (25 array tasks)"

echo "[5/5] PBT+BC-test partner (preemptable, per-seed, dep on BC)"
PBT_BC_TEST_JOB=$(sbatch --parsable \
    --dependency=afterok:${BC_JOB} \
    --export=ALL,AGENT_TYPE=bc_test \
    slurm/04_pbt.slurm)
echo "      Job ID: $PBT_BC_TEST_JOB (25 array tasks)"

echo ""
echo "========================================================"
echo "  Summary"
echo "  GPU tasks:  90  (PPO: 15 via normal_gpu | PBT: 75 via preemptable)"
echo "  CPU tasks:  20  (BC: 10 | Fig6-BC: 10 via preemptable)"
echo "  Critical path: PBT (75 tasks × ~1.5 hrs ÷ 4 slots) ≈ 28 hrs"
echo "========================================================"
echo ""
echo "Monitor:"
echo "  watch -n 60 'squeue -u \$USER -o \"%.10i %.25j %.8T %.10M %.6D %R\"'"
echo ""
echo "Check GPU utilization (sample a running node):"
echo "  squeue -u \$USER --state=RUNNING -o '%N' | tail -1 | xargs -I{} ssh {} nvidia-smi"
