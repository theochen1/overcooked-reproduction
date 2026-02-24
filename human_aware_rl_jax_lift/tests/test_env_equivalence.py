import jax.numpy as jnp

from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.overcooked_mdp import step
from human_aware_rl_jax_lift.env.state import make_initial_state


def test_step_level_parity_smoke():
    terrain = parse_layout("simple")
    state = make_initial_state(terrain)
    joint_action = jnp.array([4, 4], dtype=jnp.int32)  # STAY, STAY
    new_state, sparse_reward, shaped_reward, _ = step(terrain, state, joint_action)
    assert new_state.player_pos.shape == (2, 2)
    assert sparse_reward.shape == ()
    assert shaped_reward.shape == ()


def test_collision_parity_smoke():
    terrain = parse_layout("simple")
    state = make_initial_state(terrain)
    # Move both players toward each other in one step (layout-dependent smoke check).
    joint_action = jnp.array([2, 3], dtype=jnp.int32)  # EAST, WEST
    new_state, *_ = step(terrain, state, joint_action)
    assert new_state.player_pos.shape == (2, 2)
