# human_aware_rl_jax_lift

Pure-JAX liftover of the legacy `human_aware_rl` stack.

## Module Structure

- `env/`: JAX state containers and transition logic
- `encoding/`: BC 64-dim features and PPO 20-channel masks
- `agents/`: BC, PPO, and PBT implementations
- `planning/`: wrappers over legacy planning modules
- `experiments/`: figure-entrypoint scripts
- `reproducibility/`: seed handling and deterministic command references
- `docs/porting_map.md`: legacy-to-JAX symbol mapping
- `tests/`: parity and smoke tests

## Quickstart

```bash
pip install -e .
pytest human_aware_rl_jax_lift/tests -q
```
