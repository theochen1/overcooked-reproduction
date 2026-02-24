import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from human_aware_rl_jax_lift.env.compat import from_legacy_state
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.overcooked_mdp import step


def _assert_state_equal(a, b):
    assert np.array_equal(np.asarray(a.player_pos), np.asarray(b.player_pos))
    assert np.array_equal(np.asarray(a.player_or), np.asarray(b.player_or))
    assert np.array_equal(np.asarray(a.held_obj), np.asarray(b.held_obj))
    assert np.array_equal(np.asarray(a.held_soup), np.asarray(b.held_soup))
    assert np.array_equal(np.asarray(a.pot_state), np.asarray(b.pot_state))
    assert np.array_equal(np.asarray(a.counter_obj), np.asarray(b.counter_obj))
    assert np.array_equal(np.asarray(a.counter_soup), np.asarray(b.counter_soup))


def test_step_level_parity_against_legacy():
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")

    Action = actions_mod.Action
    OvercookedGridworld = mdp_mod.OvercookedGridworld

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)
    legacy_state = mdp.get_standard_start_state()
    jax_state = from_legacy_state(terrain, legacy_state)

    action_seq = [
        (Action.STAY, Action.STAY),
        (Action.INTERACT, Action.INTERACT),
        (Action.EAST, Action.WEST),
    ]
    for joint_action in action_seq:
        legacy_next, legacy_sparse, legacy_shaped = mdp.get_state_transition(legacy_state, joint_action)
        joint_idx = jnp.array([Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32)
        jax_next, jax_sparse, jax_shaped, _ = step(terrain, jax_state, joint_idx)

        legacy_as_jax = from_legacy_state(terrain, legacy_next)
        _assert_state_equal(jax_next, legacy_as_jax)
        assert float(jax_sparse) == float(legacy_sparse)
        assert float(jax_shaped) == float(legacy_shaped)

        legacy_state = legacy_next
        jax_state = jax_next
