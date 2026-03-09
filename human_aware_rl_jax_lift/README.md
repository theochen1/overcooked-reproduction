# human_aware_rl_jax_lift

JAX implementation of the Overcooked human-aware RL training stack (BC, PPO self-play, PPO+BC, PBT).

## Module Structure

- `env/`: JAX state containers and transition logic
- `encoding/`: BC 64-dim features and PPO 20-channel masks
- `agents/`: BC, PPO, and PBT implementations
- `planning/`: wrappers over Overcooked planning modules
- `experiments/`: figure scripts and evaluation
- `reproducibility/`: seed handling and deterministic command references
- `docs/porting_map.md`: symbol mapping for maintainers
- `tests/`: parity tests (env, encoding, models)

## Quickstart

```bash
pip install -e .
pytest human_aware_rl_jax_lift/tests -q
```

See `slurm/README.md` for the full reproduction pipeline on HPC.
