# Determinism

This module follows explicit key threading for JAX randomness:

- Every stochastic op receives an explicit `jax.random.PRNGKey`.
- Keys are split once per environment step/update and passed downstream.
- Python `random` and NumPy RNG are seeded once per process via `set_global_seed()`.

Known non-determinism risks:

- Different accelerator backends (CPU/GPU/TPU) can produce slight numeric drift.
- Parallel host execution order can affect floating-point reduction order.
- Mixed precision may introduce backend-specific differences.
