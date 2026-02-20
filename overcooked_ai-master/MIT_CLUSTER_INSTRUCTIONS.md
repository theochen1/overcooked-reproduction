# HPC Cluster Paper Reproduction Instructions

Complete instructions for reproducing the Overcooked AI paper results on an HPC cluster.

## Overview

This will train **80 models** total:
- **5 BC models** (1 per layout) - ~4 hours each
- **5 GAIL models** (1 per layout) - ~8 hours each
- **25 PPO Self-Play models** (5 layouts × 5 seeds) - ~6 hours each
- **25 PPO+BC models** (5 layouts × 5 seeds) - ~12 hours each
- **25 PPO+GAIL models** (5 layouts × 5 seeds) - ~12 hours each

---

## Step 1: Get the Code

```bash
# Clone the repository
git clone https://github.com/clutchnoob/overcooked_ai.git
cd overcooked_ai

# If using a specific branch
git checkout <branch-name>
```

---

## Step 2: Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n overcooked python=3.10 -y
conda activate overcooked

# Install the overcooked package in editable mode
pip install -e .

# Install JAX/Flax for PPO training (CPU)
pip install "jax>=0.4.30" "jaxlib>=0.4.30" "flax>=0.8,<0.11" "optax>=0.2"

# Install PyTorch for BC/GAIL training
pip install torch>=2.0 tensorboard>=2.14

# Install other dependencies
pip install matplotlib seaborn Pillow wandb

# Verify installation
python -c "
import overcooked_ai_py; print('overcooked_ai_py OK')
import human_aware_rl; print('human_aware_rl OK')
import jax; print(f'JAX {jax.__version__} OK')
import flax; print(f'Flax {flax.__version__} OK')
import torch; print(f'PyTorch {torch.__version__} OK')
print('All imports successful!')
"
```

---

## Step 3: Update Configuration (REQUIRED!)

Edit `hpc_scripts/config.sh` to match your cluster setup:

```bash
# Line 13: Update PROJECT_ROOT to your directory
export PROJECT_ROOT="$HOME/home/overcooked_ai"  # <-- UPDATE THIS

# Lines 24-28: Update conda setup for your cluster
setup_conda() {
    source "$HOME/miniconda3/etc/profile.d/conda.sh"  # <-- UPDATE THIS
    conda activate overcooked                           # <-- UPDATE THIS
}
```

### Find your conda path:
```bash
which conda
# e.g., /home/user/miniconda3/bin/conda
# Then use: source /home/user/miniconda3/etc/profile.d/conda.sh
```

---

## Step 4: Create Required Directories

```bash
cd ~/home/overcooked_ai  # or wherever you cloned it

# Create logs directory
mkdir -p hpc_scripts/logs

# Create results directories
mkdir -p src/human_aware_rl/results/ppo_sp
mkdir -p src/human_aware_rl/results/ppo_bc
mkdir -p src/human_aware_rl/results/ppo_gail
mkdir -p src/human_aware_rl/bc_runs
```

---

## Step 5: Verify Setup

```bash
# Source config and check it resolves properly
source hpc_scripts/config.sh
echo "PROJECT_ROOT: $PROJECT_ROOT"
which python
python -c "from human_aware_rl.jaxmarl.ppo import PPOConfig; print('PPOConfig imported OK')"
```

---

## Step 6: Run Everything (One Command)

```bash
cd hpc_scripts

# First, do a dry run to verify all scripts are found
./submit_all.sh --dry-run

# If dry run looks good, submit ALL 80 jobs with proper dependencies
./submit_all.sh
```

This will:
1. Submit 5 BC jobs (no dependencies)
2. Submit 25 PPO_SP jobs (no dependencies, runs in parallel)
3. Submit 50 PPO+partner jobs (BC, GAIL) that wait for BC to complete

### Alternative: Step-by-step Submission

```bash
# Submit BC only first
./submit_all.sh --bc-only

# Then later, submit PPO jobs (after BC finishes)
./submit_all.sh --ppo-only
```

---

## Step 7: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch jobs in real-time
watch -n 10 'squeue -u $USER'

# View job output logs
tail -f hpc_scripts/logs/ppo_sp_forced_coordination_seed0_*.out

# View job error logs
tail -f hpc_scripts/logs/ppo_sp_forced_coordination_seed0_*.err

# Cancel all jobs if needed
scancel -u $USER
```

---

## Expected Results

### PPO Self-Play Training Rewards
| Layout | Expected Training Reward |
|--------|--------------------------|
| cramped_room | ~200 |
| asymmetric_advantages | ~200 |
| coordination_ring | ~150 |
| forced_coordination | ~150 |
| counter_circuit | ~120 |

### PPO Self-Play Evaluation (SP+SP)
| Layout | Expected Eval Reward |
|--------|----------------------|
| cramped_room | ~220 |
| asymmetric_advantages | ~220 |
| coordination_ring | ~170 |
| forced_coordination | ~160-200 |
| counter_circuit | ~120-160 |

---

## Output Locations

After training completes, models are saved to:

```
src/human_aware_rl/
├── bc_runs/
│   ├── train/{layout}/    # BC training data
│   └── test/{layout}/     # BC test data
└── results/
    ├── ppo_sp/ppo_sp_{layout}_seed{N}/checkpoint_*/
    ├── ppo_bc/ppo_bc_{layout}_seed{N}/checkpoint_*/
    └── ppo_gail/ppo_gail_{layout}_seed{N}/checkpoint_*/
```

Each checkpoint contains:
- `params.pkl` - Model weights
- `config.pkl` - Training configuration

---

## Job Details

| Experiment | Jobs | Time/Job | Memory | CPUs |
|------------|------|----------|--------|------|
| BC | 5 | 4h | 16GB | 8 |
| GAIL | 5 | 8h | 32GB | 8 |
| PPO_SP | 25 | 48h (max) | 32GB | 16 |
| PPO_BC | 25 | 48h (max) | 32GB | 16 |
| PPO_GAIL | 25 | 48h (max) | 32GB | 16 |

---

## Troubleshooting

### "ModuleNotFoundError"
Check PYTHONPATH in `config.sh`:
```bash
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
```

### "conda not found"
Update the `setup_conda()` function in `config.sh` to point to your conda installation.

### Jobs stuck in PENDING
Check cluster availability:
```bash
sinfo
```

### Job failed immediately
Check the error log:
```bash
cat hpc_scripts/logs/<job_name>_<job_id>.err
```

### BC models didn't train
PPO+BC/GAIL jobs depend on BC completion. If BC fails, dependent jobs will fail.

---

## Quick Reference

```bash
# === SETUP ===
git clone https://github.com/clutchnoob/overcooked_ai.git
cd overcooked_ai
conda create -n overcooked python=3.10 -y && conda activate overcooked
pip install -e . && pip install jax jaxlib flax optax torch tensorboard matplotlib
# Edit hpc_scripts/config.sh with your paths!
mkdir -p hpc_scripts/logs

# === RUN EVERYTHING ===
cd hpc_scripts
./submit_all.sh --dry-run   # verify first
./submit_all.sh              # submit all 80 jobs

# === MONITOR ===
squeue -u $USER
tail -f logs/*.out

# === CANCEL ===
scancel -u $USER
```

---

## Important Notes

1. **All experiments use the same layouts** - Legacy versions with explicit MDP parameters (cook_time=20, num_items=3, delivery_reward=20)

2. **Corrected hyperparameters** - vf_coef=0.5, ent_coef=0.01 (NOT 0.1)

3. **Seeds** - 0, 10, 20, 30, 40 (matches paper)

4. **Layouts** - cramped_room, asymmetric_advantages, coordination_ring, forced_coordination, counter_circuit

5. **No partition hardcoded** - Scripts use the cluster's default partition. If you need a specific partition, add `#SBATCH --partition=<name>` to the scripts or pass `--partition=<name>` to sbatch.

---

## Contact

If you encounter issues, check:
1. `hpc_scripts/logs/` for job output
2. `config.sh` for path configuration
3. Cluster documentation for SLURM specifics
