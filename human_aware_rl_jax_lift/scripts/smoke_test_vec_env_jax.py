"""Smoke test: lossless_state_encoding_20 is vmap-safe with batched env.

Run from package dir (e.g. with conda env that has JAX):
  cd human_aware_rl_jax_lift && python scripts/smoke_test_vec_env_jax.py

Expected:
  obs0.shape = (4, H, W, 20)  e.g. (4, 5, 9, 20) for unident_s
  rewards.shape = (4,)  dones.shape = (4,)
"""
import jax
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.training.vec_env import (
    batched_step,
    encode_obs,
    make_batched_state,
)

def main():
    terrain = parse_layout("unident_s")
    h, w = terrain.grid.shape
    num_envs = 4
    rng = jax.random.PRNGKey(0)
    bstate = make_batched_state(terrain, num_envs, rng)
    obs0, obs1 = encode_obs(terrain, bstate)
    print("obs0.shape:", obs0.shape)  # expect (4, H, W, 20)

    actions = jax.numpy.zeros((num_envs,), dtype=jax.numpy.int32)
    new_bstate, obs0, obs1, rewards, dones, sparse = batched_step(
        terrain, bstate, actions, actions, jax.numpy.array(1.0), 400
    )
    print("rewards.shape:", rewards.shape, "dones.shape:", dones.shape)  # expect (4,) (4,)

    assert obs0.shape == (num_envs, h, w, 20), f"obs0.shape: {obs0.shape}"
    assert rewards.shape == (num_envs,), f"rewards.shape: {rewards.shape}"
    assert dones.shape == (num_envs,), f"dones.shape: {dones.shape}"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
