#!/bin/bash
#SBATCH -n 16
#SBATCH -t 47:00:00
#SBATCH --mem=32G
#SBATCH --job-name=ppo_sp_paper
#SBATCH --output=ppo_sp_%j.out

# ============================================================================
# PPO Self-Play Training Script (Paper Reproduction)
# ============================================================================
# This script trains PPO agents using the CORRECTED hyperparameters that
# match the original TensorFlow implementation from the NeurIPS 2019 paper.
#
# Key hyperparameters (validated against successful paper reproduction):
#   - vf_coef: 0.5 (NOT 0.1)
#   - ent_coef: 0.01 (NOT 0.1)
#   - learning_rate: 0.0008
#   - num_envs: 60
#   - total_timesteps: 5,000,000
#   - reward_shaping_horizon: 2,500,000
#   - use_legacy_encoding: True (20-channel observations)
#   - old_dynamics: True (auto-cook when pot has 3 ingredients)
#
# Expected results (SP+SP evaluation):
#   - cramped_room: ~200
#   - asymmetric_advantages: ~200
#   - coordination_ring: ~150
#   - forced_coordination: ~160
#   - counter_circuit: ~120
# ============================================================================

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

echo "============================================================================"
echo "Starting PPO Self-Play Training (Paper Reproduction)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "============================================================================"

# Train all layouts with all seeds
# Uses corrected hyperparameters from paper_configs.py
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40

echo "============================================================================"
echo "Training complete: $(date)"
echo "============================================================================"
