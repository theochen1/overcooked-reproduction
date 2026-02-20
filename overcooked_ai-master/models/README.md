# Trained Model Weights

This directory contains trained model weights for the Overcooked-AI human-AI coordination experiments, replicating results from Carroll et al. "On the Utility of Learning about Humans for Human-AI Coordination".

## Directory Structure

```
models/
├── bc/                    # Behavioral Cloning models (PyTorch)
│   └── {layout}/
│       └── model.pt
├── gail/                  # GAIL models (PyTorch)
│   └── {layout}/
│       └── model.pt
├── ppo_sp/                # PPO Self-Play models (JAX/Flax)
│   └── {layout}/
│       └── seed{N}/
│           ├── params.pkl
│           └── config.pkl
├── ppo_bc/                # PPO trained with BC partner (JAX/Flax)
│   └── {layout}/
│       └── seed{N}/
│           ├── params.pkl
│           └── config.pkl
└── ppo_gail/              # PPO trained with GAIL partner (JAX/Flax)
    └── {layout}/
        └── seed{N}/
            ├── params.pkl
            └── config.pkl
```

## Layouts

- `cramped_room`
- `asymmetric_advantages`
- `coordination_ring`
- `forced_coordination`
- `counter_circuit`

## Seeds

PPO models are trained with 5 seeds: `0, 10, 20, 30, 40`

## Model Counts

| Model Type | Count | Format |
|------------|-------|--------|
| BC | 5 | PyTorch (`model.pt`) |
| GAIL | 5 | PyTorch (`model.pt`) |
| PPO_SP | 25 | JAX/Flax (`params.pkl`) |
| PPO_BC | 25 | JAX/Flax (`params.pkl`) |
| PPO_GAIL | 25 | JAX/Flax (`params.pkl`) |
| **Total** | **85** | |

## Loading Models

### BC/GAIL Models (PyTorch)

```python
import torch
from human_aware_rl.imitation.bc_agent import BCAgent

# Load BC model
model = BCAgent(layout_name="cramped_room")
model.load_state_dict(torch.load("models/bc/cramped_room/model.pt"))
```

### PPO Models (JAX/Flax)

```python
import pickle

# Load PPO model
with open("models/ppo_sp/cramped_room/seed0/params.pkl", "rb") as f:
    params = pickle.load(f)

with open("models/ppo_sp/cramped_room/seed0/config.pkl", "rb") as f:
    config = pickle.load(f)
```

## Evaluation

To evaluate these models with a Human Proxy partner, run:

```bash
cd hpc_scripts/evaluation
sbatch run_evaluation.sh
```

Or for local evaluation:

```bash
python -m human_aware_rl.evaluation.evaluate_hpc_models --num_games 10
```

## Training Details

- **BC/GAIL**: Trained on human gameplay data
- **PPO_SP**: Self-play PPO (agent plays with a copy of itself)
- **PPO_BC**: PPO agent trained with BC agent as partner
- **PPO_GAIL**: PPO agent trained with GAIL agent as partner

Final checkpoints used:
- `cramped_room`: PPO checkpoint 550 (SP/BC), 515 (GAIL)
- Other layouts: PPO checkpoint 650 (SP/BC), 609 (GAIL)
