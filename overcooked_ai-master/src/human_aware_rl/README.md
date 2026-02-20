# Human-Aware Reinforcement Learning

This code is based on the work in [On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789). 

## Overview

This module provides tools for training AI agents that coordinate effectively with humans in the Overcooked environment. The implementation has been modernized to use:

- **PyTorch** for Behavior Cloning (BC)
- **JAX/Flax** for Reinforcement Learning (PPO)

## Installation

```bash
# For behavior cloning only
pip install ".[bc]"

# For full training stack (PyTorch + JAX)
pip install ".[harl]"

# With CUDA support
pip install ".[harl-cuda]"

# Legacy TensorFlow/RLlib (deprecated)
pip install ".[harl-legacy]"
```

## Contents

### Imitation Learning (`imitation/`)

PyTorch-based imitation learning from human demonstrations:

**Behavior Cloning (BC):**
- `behavior_cloning.py`: Main BC module with MLP and LSTM models
- `bc_agent.py`: Agent wrapper for trained BC models
- `train_bc_models.py`: Batch training script for all layouts

**Adversarial Inverse Reinforcement Learning (AIRL):**
- `airl.py`: AIRL implementation with disentangled reward structure
- `airl_agent.py`: Agent wrapper for AIRL-trained policies
- `train_airl.py`: Batch training script for all layouts

AIRL learns a more robust human proxy than BC by:
- Training a policy via adversarial learning against a discriminator
- Reducing compounding errors by exploring during training
- Covering more states than BC's supervised learning approach

### Reinforcement Learning (`jaxmarl/`)

JAX-based multi-agent PPO training:

- `overcooked_env.py`: JAX-compatible environment wrapper
- `ppo.py`: PPO implementation with self-play and BC-schedule support

### Policy Bridge (`bridge/`)

Utilities for loading and evaluating trained policies:

- `jax_agent.py`: Wrapper for JAX policies as Overcooked agents
- `evaluate.py`: Evaluation utilities

### PPO Training (`ppo/`)

Training scripts and configurations:

- `ppo_client.py`: Main training script with CLI and Sacred/WandB support

### Human Data (`human/`)

Data processing utilities for human demonstration data:

- `process_dataframes.py`: Convert raw data to training format
- `data_processing_utils.py`: Helper functions

## Quick Start

### Train a BC Model

```python
from human_aware_rl.imitation.behavior_cloning import (
    get_bc_params,
    train_bc_model,
    evaluate_bc_model,
)

# Get default parameters
params = get_bc_params(layout_name="cramped_room", epochs=100)

# Train model
model = train_bc_model("./bc_model", params, verbose=True)

# Evaluate
reward = evaluate_bc_model(model, params)
print(f"Average reward: {reward}")
```

### Train a PPO Agent

```bash
# Self-play training
python -m human_aware_rl.ppo.ppo_client \
    --layout cramped_room \
    --total_timesteps 1000000

# Training with BC partner
python -m human_aware_rl.ppo.ppo_client \
    --layout cramped_room \
    --bc_model_dir ./bc_model
```

### Evaluate Trained Agents

```python
from human_aware_rl.bridge import load_and_evaluate

results = load_and_evaluate(
    checkpoint_path="./ppo_checkpoint",
    layout_name="cramped_room",
    agent_type="jax",
    num_games=10
)
print(f"Mean return: {results['mean_return']:.2f}")
```

## Reproducing Paper Results

To reproduce the results from "On the Utility of Learning about Humans for Human-AI Coordination":

### Step 1: Train BC Models

```bash
# Train both BC models (training data) and Human Proxy models (test data)
python -m human_aware_rl.imitation.train_bc_models --all_layouts
```

### Step 2: Train PPO Self-Play Agents

```bash
# Train all 5 layouts with 5 seeds each
python -m human_aware_rl.ppo.train_ppo_sp --all_layouts --seeds 0,10,20,30,40
```

### Step 3: Train PBT Agents (Optional)

```bash
# Population-Based Training
python -m human_aware_rl.ppo.train_pbt --all_layouts
```

### Step 4: Train PPO with BC Partner

```bash
# Train PPO_BC agents (requires BC models from Step 1)
python -m human_aware_rl.ppo.train_ppo_bc --all_layouts --seeds 0,10,20,30,40
```

### Step 4b (Alternative): Train AIRL Human Proxy

AIRL provides a more robust human proxy than BC:

```bash
# Train AIRL models (replaces BC)
python -m human_aware_rl.imitation.train_airl --all_layouts

# Fast training (500K timesteps instead of 5M)
python -m human_aware_rl.imitation.train_airl --all_layouts --fast
```

### Step 4c (Alternative): Train PPO with AIRL Partner

```bash
# Train PPO_AIRL agents (requires AIRL models from Step 4b)
python -m human_aware_rl.ppo.train_ppo_airl --all_layouts --seeds 0,10,20,30,40

# Fast training
python -m human_aware_rl.ppo.train_ppo_airl --all_layouts --fast
```

### Step 5: Evaluate All Agents

```bash
# Run paper-style evaluations
python -m human_aware_rl.evaluation.evaluate_paper \
    --ppo_sp_dir results/ppo_sp \
    --ppo_bc_dir results/ppo_bc \
    --output_file paper_results.json
```

### Step 6: Generate Figures

```bash
# Create paper-style figures
python -m human_aware_rl.visualization.plot_results \
    --results_file paper_results.json \
    --output_dir figures/
```

## Testing

Run the test suite:

```bash
cd src/human_aware_rl
./run_tests.sh
```

For legacy TensorFlow tests:

```bash
./run_tests_legacy.sh
```

## Module Structure

```
human_aware_rl/
├── __init__.py
├── data_dir.py
├── utils.py
├── run_tests.sh
├── run_tests_legacy.sh
├── imitation/              # Imitation Learning (PyTorch)
│   ├── behavior_cloning.py # BC models and training
│   ├── bc_agent.py         # BC agent wrapper
│   ├── train_bc_models.py  # BC batch training
│   ├── airl.py             # AIRL discriminator and trainer
│   ├── airl_agent.py       # AIRL agent wrapper
│   ├── train_airl.py       # AIRL batch training
│   └── behavior_cloning_test.py
├── jaxmarl/                # RL Training (JAX)
│   ├── overcooked_env.py
│   ├── ppo.py
│   └── pbt.py              # Population-Based Training
├── bridge/                 # Policy Evaluation
│   ├── jax_agent.py
│   └── evaluate.py
├── ppo/                    # Training Scripts
│   ├── ppo_client.py       # General PPO client
│   ├── train_ppo_sp.py     # Self-play training
│   ├── train_ppo_bc.py     # PPO with BC partner
│   ├── train_ppo_airl.py   # PPO with AIRL partner
│   ├── train_pbt.py        # PBT training
│   └── configs/
│       └── paper_configs.py # Paper hyperparameters
├── evaluation/             # Evaluation Scripts
│   ├── evaluate_all.py     # Batch evaluation
│   └── evaluate_paper.py   # Paper-style evaluation
├── visualization/          # Plotting
│   └── plot_results.py     # Paper figures
├── human/                  # Data Processing
│   ├── process_dataframes.py
│   └── data_processing_utils.py
├── static/                 # Human Data
│   └── human_data/
└── rllib/                  # Legacy (deprecated)
    └── rllib.py
```

## Legacy Support

The original TensorFlow/RLlib implementation is preserved in:
- `imitation/behavior_cloning_tf2.py`
- `rllib/rllib.py`
- `ppo/ppo_rllib.py`

These files are deprecated and will be removed in a future release.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{carroll2019utility,
  title={On the Utility of Learning about Humans for Human-AI Coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Thomas L and Seshia, Sanjit A and Abbeel, Pieter and Dragan, Anca},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.
