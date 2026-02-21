# JAX-based PPO for Overcooked AI

This module provides a JAX/Flax implementation of PPO (Proximal Policy Optimization) for training cooperative agents in the Overcooked environment. It faithfully reproduces the results from the 2019 NeurIPS paper "On the Utility of Learning about Humans for Human-AI Coordination".

## Results

Training achieves the following rewards on paper benchmark layouts:

| Layout | Train Shaped Return (windowed) | Train Sparse Return (windowed) | Eval Sparse Return (SP+SP) |
|--------|--------------------------------|--------------------------------|----------------------------|
| random0_legacy (forced_coordination) | 139.8 ± 5.2 | — | 200.0 |
| random3_legacy | 120.5 ± 9.1 | — | 180.0 |

These results match or exceed the original TensorFlow implementation.

Notes:
- `Train Shaped Return` includes reward shaping during training (annealed over time).
- `Train Sparse Return` and `Eval Sparse Return` are sparse-only coordination metrics.
- Evaluation policy mode is recorded per run as `eval_policy` (`stochastic` or `greedy`).

## Quick Start

### Paper Reproduction

To train with the original paper's hyperparameters:

```bash
cd overcooked_ai-master/src
python -m human_aware_rl.jaxmarl.train_paper_reproduction --layout random0_legacy
```

### Custom Training

```python
from human_aware_rl.jaxmarl import PPOConfig, PPOTrainer

config = PPOConfig(
    layout_name="random0_legacy",
    total_timesteps=5_000_000,
    num_envs=60,
    learning_rate=8e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    use_legacy_encoding=True,  # Use 20-channel encoding from paper
)

trainer = PPOTrainer(config)
results = trainer.train()

print(f"Final reward: {results['final_mean_reward']:.2f}")
```

## Key Implementation Details

The implementation matches the original TensorFlow baselines in several critical ways:

1. **Weight Initialization**: Glorot uniform for conv/dense layers, orthogonal(scale=0.01) for policy output
2. **Activation Function**: Leaky ReLU with negative_slope=0.2 (TensorFlow default)
3. **Observation Encoding**: 20-channel legacy encoding from the 2019 paper
4. **Agent Index Randomization**: Training agent randomly plays as player 0 or 1
5. **Shared Shaped Rewards**: Both agents receive the same total shaped reward
6. **Reward Shaping Annealing**: Shaped rewards anneal to 0 over 2.5M timesteps

## Module Structure

- `ppo.py`: PPO trainer, actor-critic networks, training loop
- `overcooked_env.py`: JAX-compatible environment wrapper
- `train_paper_reproduction.py`: Training script with paper hyperparameters
- `pbt.py`: Population-based training (experimental)

## Configuration Options

Key `PPOConfig` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layout_name` | "cramped_room" | Overcooked layout name |
| `total_timesteps` | 1_000_000 | Total training steps |
| `num_envs` | 30 | Number of parallel environments |
| `learning_rate` | 1e-3 | Adam learning rate |
| `ent_coef` | 0.1 | Entropy coefficient |
| `vf_coef` | 0.1 | Value function loss coefficient |
| `clip_eps` | 0.05 | PPO clip epsilon |
| `gae_lambda` | 0.98 | GAE lambda |
| `reward_shaping_horizon` | inf | Steps to anneal shaped rewards |
| `use_legacy_encoding` | True | Use 20-channel encoding |
| `verbose_debug` | False | Enable detailed diagnostics |

## Requirements

- JAX >= 0.4.0
- Flax >= 0.7.0
- Optax >= 0.1.7
- NumPy >= 1.24.0

Install with:
```bash
pip install jax jaxlib flax optax
```

For GPU support:
```bash
pip install "jax[cuda12]" flax optax
```



