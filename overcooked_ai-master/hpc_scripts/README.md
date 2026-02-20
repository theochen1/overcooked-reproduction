# HPC Training Scripts

Parallelized SLURM batch scripts for training all Overcooked AI models.

## Quick Start

```bash
# Submit all training jobs (80 total)
./submit_all.sh

# Dry run (see what would be submitted)
./submit_all.sh --dry-run

# Submit only BC models first
./submit_all.sh --bc-only

# Submit only PPO models (assumes BC already trained)
./submit_all.sh --ppo-only
```

## Directory Structure

```
hpc_scripts/
├── config.sh           # Shared configuration (conda, paths)
├── submit_all.sh       # Master submission script
├── README.md           # This file
├── bc/                 # Behavior Cloning (5 scripts)
│   ├── submit_bc.sh
│   ├── cramped_room.sh
│   ├── asymmetric_advantages.sh
│   ├── coordination_ring.sh
│   ├── forced_coordination.sh
│   └── counter_circuit.sh
├── gail/               # GAIL training (5 scripts)
│   ├── submit_gail.sh
│   └── {layout}.sh
├── ppo_sp/             # PPO Self-Play (25 scripts)
│   ├── submit_ppo_sp.sh
│   └── {layout}_seed{0,10,20,30,40}.sh
├── ppo_bc/             # PPO with BC partner (25 scripts)
│   ├── submit_ppo_bc.sh
│   └── {layout}_seed{0,10,20,30,40}.sh
├── ppo_gail/           # PPO with GAIL partner (25 scripts)
│   ├── submit_ppo_gail.sh
│   └── {layout}_seed{0,10,20,30,40}.sh
├── evaluation/         # Evaluation scripts
│   └── run_evaluation.sh
└── logs/               # SLURM output logs
```

## Training Configuration

### SLURM Resources

| Parameter | Value |
|-----------|-------|
| **Layouts** | cramped_room, asymmetric_advantages, coordination_ring, forced_coordination, counter_circuit |
| **Seeds** | 0, 10, 20, 30, 40 |
| **Time limit** | 48 hours (PPO), 4 hours (BC), 8 hours (GAIL) |
| **Memory** | 32GB (PPO), 16GB (BC), 32GB (GAIL) |
| **CPUs** | 16 (PPO), 8 (BC/GAIL) |
| **Partition** | Uses cluster default (not hardcoded) |

### PPO Hyperparameters (Paper Reproduction)

**IMPORTANT**: These hyperparameters have been **CORRECTED** based on analysis of the
original TensorFlow implementation and successful reproduction experiments.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **vf_coef** | 0.5 | Was incorrectly 0.1 - critical for value function learning |
| **ent_coef** | 0.01 | Was incorrectly 0.1 - prevents policy from staying random |
| **learning_rate** | 0.0008 | Corrected from 0.001 |
| **num_envs** | 60 | Corrected from 30 - larger batch size |
| **total_timesteps** | 5,000,000 | |
| **reward_shaping_horizon** | 2,500,000 | Reward shaping anneals to 0 by this point |
| **gamma** | 0.99 | |
| **gae_lambda** | 0.98 | |
| **clip_eps** | 0.05 | |
| **max_grad_norm** | 0.1 | |
| **use_legacy_encoding** | True | 20-channel observation encoding |
| **old_dynamics** | True | Auto-cook when pot has 3 ingredients |

### Layout Configurations

**IMPORTANT**: ALL experiments use the SAME legacy layout versions for consistency!

This matches the original paper which used the same layouts (`random0`, `random3`, etc.) 
for ALL experiments (PPO SP, PPO BC, PPO GAIL).

| Paper Name | Environment Layout | MDP Parameters |
|------------|-------------------|----------------|
| cramped_room | `cramped_room_legacy` | cook_time=20, num_items=3, delivery_reward=20 |
| asymmetric_advantages | `asymmetric_advantages_legacy` | cook_time=20, num_items=3, delivery_reward=20 |
| coordination_ring | `coordination_ring_legacy` | cook_time=20, num_items=3, delivery_reward=20 |
| forced_coordination | `random0_legacy` | cook_time=20, num_items=3, delivery_reward=20 |
| counter_circuit | `random3_legacy` | cook_time=20, num_items=3, delivery_reward=20 |

### Expected Evaluation Results (All Experiments on Legacy Layouts)

| Layout | PPO SP | PPO BC | PPO GAIL |
|--------|--------|--------|----------|
| cramped_room | ~200 | TBD | TBD |
| asymmetric_advantages | ~200 | TBD | TBD |
| coordination_ring | ~150 | TBD | TBD |
| forced_coordination | ~160 | TBD | TBD |
| counter_circuit | ~120 | TBD | TBD |

Note: PPO BC and PPO GAIL results may differ from previously reported values 
because all experiments now use legacy layouts with explicit MDP parameters.

## Job Dependencies

```
BC (5 jobs) ─────┬──> PPO_BC (25 jobs)
                 └──> PPO_GAIL (25 jobs)

PPO_SP (25 jobs) ───> (independent, no dependencies)
```

## Output Locations

| Model | Output Directory |
|-------|------------------|
| BC | `src/human_aware_rl/bc_runs/{train\|test}/{layout}/` |
| PPO_SP | `src/human_aware_rl/results/ppo_sp/ppo_sp_{layout}_seed{seed}/` |
| PPO_BC | `src/human_aware_rl/results/ppo_bc/ppo_bc_{layout}_seed{seed}/` |
| PPO_GAIL | `src/human_aware_rl/results/ppo_gail/ppo_gail_{layout}_seed{seed}/` |

## Monitoring Jobs

```bash
# View your running jobs
squeue -u $USER

# View job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Submitting Individual Jobs

```bash
# Submit a single BC job
sbatch hpc_scripts/bc/cramped_room.sh

# Submit a single PPO_SP job
sbatch hpc_scripts/ppo_sp/cramped_room_seed0.sh

# Submit PPO_BC with dependency on BC completion
sbatch --dependency=afterok:<bc_job_id> hpc_scripts/ppo_bc/cramped_room_seed0.sh
```

## Setup

1. Update `config.sh` with your cluster paths and conda environment
2. Create logs directory: `mkdir -p hpc_scripts/logs`
3. Run `./submit_all.sh --dry-run` to verify scripts are found
4. Run `./submit_all.sh` to submit everything
