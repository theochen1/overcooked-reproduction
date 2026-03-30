# human_aware_rl_jax_lift

JAX implementation of the Overcooked training stack (Carroll et al., NeurIPS 2019).
Supports BC, PPO self-play, PPO+BC, and PBT.

## Module Structure

- `env/`: JAX state containers and transition logic
- `encoding/`: BC 62/64-dim features and PPO 20-channel masks
- `agents/`: BC, PPO, and PBT implementations
- `config.py`: paper-locked hyperparameters and seeding
- `training/`: rollout, PPO loop, PBT loop, checkpoints
- `experiments/`: figure scripts and evaluation
- `scripts/`: CLI entrypoints for training
- `tests/`: parity tests (env, encoding, models)

## Quickstart

```bash
pip install -e .
pytest human_aware_rl_jax_lift/tests -q
```

See `slurm/README.md` for the full reproduction pipeline on HPC.
