import numpy as np
import pytest

from human_aware_rl_jax_lift.env.compat import from_legacy_state
from human_aware_rl_jax_lift.encoding.ppo_masks import lossless_state_encoding_20
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.state import make_initial_state


def _get_mdp_and_terrain(layout="simple"):
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    OvercookedGridworld = mdp_mod.OvercookedGridworld
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    Action = actions_mod.Action
    terrain = parse_layout(layout)
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)
    return mdp, terrain, Action, mdp_mod


def _assert_encoding_equal(terrain, legacy_state, mdp):
    jax_state = from_legacy_state(terrain, legacy_state)
    jax_o0, jax_o1 = lossless_state_encoding_20(terrain, jax_state)
    legacy_o0, legacy_o1 = mdp.lossless_state_encoding(legacy_state)
    assert np.array_equal(np.asarray(jax_o0), np.asarray(legacy_o0)), (
        f"player-0 obs mismatch\nJAX:\n{np.asarray(jax_o0)}\nLegacy:\n{np.asarray(legacy_o0)}"
    )
    assert np.array_equal(np.asarray(jax_o1), np.asarray(legacy_o1)), (
        f"player-1 obs mismatch\nJAX:\n{np.asarray(jax_o1)}\nLegacy:\n{np.asarray(legacy_o1)}"
    )


def test_encoding_with_onions_in_pot():
    mdp, terrain, Action, _ = _get_mdp_and_terrain("simple")
    state = mdp.get_standard_start_state()
    actions = [
        (Action.NORTH, Action.STAY),
        (Action.WEST, Action.STAY),
        (Action.INTERACT, Action.STAY),  # pick onion
        (Action.EAST, Action.STAY),
        (Action.EAST, Action.STAY),
        (Action.NORTH, Action.STAY),
        (Action.INTERACT, Action.STAY),  # place onion in pot
    ]
    for a in actions:
        state, _, _ = mdp.get_state_transition(state, a)
        _assert_encoding_equal(terrain, state, mdp)


def test_encoding_with_cooked_soup_in_pot():
    mdp, terrain, _, mdp_mod = _get_mdp_and_terrain("simple")
    ObjectState = mdp_mod.ObjectState
    state = mdp.get_standard_start_state().deepcopy()
    for cook_t in (0, 10, 20):
        s = state.deepcopy()
        s.objects[(2, 0)] = ObjectState("soup", (2, 0), state=("onion", 3, cook_t))
        _assert_encoding_equal(terrain, s, mdp)


def test_encoding_held_soup():
    mdp, terrain, _, mdp_mod = _get_mdp_and_terrain("simple")
    ObjectState = mdp_mod.ObjectState
    state = mdp.get_standard_start_state().deepcopy()
    p0 = state.players[0]
    p0.set_object(ObjectState("soup", p0.position, state=("onion", 3, 20)))
    _assert_encoding_equal(terrain, state, mdp)
    jax_state = from_legacy_state(terrain, state)
    jax_o0, _ = lossless_state_encoding_20(terrain, jax_state)
    p0x, p0y = np.asarray(jax_state.player_pos[0]).tolist()
    assert int(np.asarray(jax_o0)[p0y, p0x, 17]) == 1


def test_encoding_counter_object():
    mdp, terrain, _, mdp_mod = _get_mdp_and_terrain("simple")
    ObjectState = mdp_mod.ObjectState
    state = mdp.get_standard_start_state().deepcopy()
    state.objects[(0, 2)] = ObjectState("onion", (0, 2))
    _assert_encoding_equal(terrain, state, mdp)
    jax_state = from_legacy_state(terrain, state)
    jax_o0, _ = lossless_state_encoding_20(terrain, jax_state)
    assert np.any(np.asarray(jax_o0[:, :, 19]) == 1), "onion on counter not reflected in channel 20"


def test_encoding_all_layouts():
    for layout in ("simple", "unident_s", "random1", "random0", "random3"):
        mdp, terrain, _, _ = _get_mdp_and_terrain(layout)
        state = mdp.get_standard_start_state()
        _assert_encoding_equal(terrain, state, mdp)


def test_start_state_encoding_matches_legacy_reset():
    """Ensure fresh JAX resets produce identical observations to legacy TF."""
    for layout in ("simple", "unident_s", "random1", "random0", "random3"):
        mdp, terrain, _, _ = _get_mdp_and_terrain(layout)
        legacy_state = mdp.get_standard_start_state()
        jax_state = make_initial_state(terrain)

        legacy_from_jax = from_legacy_state(terrain, legacy_state)
        assert np.array_equal(
            np.asarray(jax_state.player_or), np.asarray(legacy_from_jax.player_or)
        ), f"{layout}: start orientation mismatch"

        jax_o0, jax_o1 = lossless_state_encoding_20(terrain, jax_state)
        legacy_o0, legacy_o1 = mdp.lossless_state_encoding(legacy_state)
        assert np.array_equal(np.asarray(jax_o0), np.asarray(legacy_o0)), (
            f"{layout}: player-0 start obs mismatch"
        )
        assert np.array_equal(np.asarray(jax_o1), np.asarray(legacy_o1)), (
            f"{layout}: player-1 start obs mismatch"
        )
