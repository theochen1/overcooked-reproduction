#!/bin/bash
#SBATCH -n 16
#SBATCH -t 47:00:00
#SBATCH --mem=32G

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh

conda activate /om/scratch/Mon/mabdel03/conda_envs/MAL_env

python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0 --fast
