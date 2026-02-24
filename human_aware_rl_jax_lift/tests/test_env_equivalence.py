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


def _run_and_compare(mdp, terrain, legacy_state, jax_state, joint_action):
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    Action = actions_mod.Action
    legacy_next, legacy_sparse, legacy_shaped = mdp.get_state_transition(legacy_state, joint_action)
    joint_idx = jnp.array([Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32)
    jax_next, jax_sparse, jax_shaped, _ = step(terrain, jax_state, joint_idx)
    legacy_as_jax = from_legacy_state(terrain, legacy_next)
    _assert_state_equal(jax_next, legacy_as_jax)
    assert float(jax_sparse) == float(legacy_sparse), (
        f"sparse reward mismatch: jax={float(jax_sparse)} legacy={float(legacy_sparse)}"
    )
    assert float(jax_shaped) == float(legacy_shaped), (
        f"shaped reward mismatch: jax={float(jax_shaped)} legacy={float(legacy_shaped)}"
    )
    return legacy_next, jax_next, float(jax_sparse), float(jax_shaped)


def test_step_level_parity_against_legacy_smoke():
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
        legacy_next, jax_next, _, _ = _run_and_compare(mdp, terrain, legacy_state, jax_state, joint_action)

        legacy_state = legacy_next
        jax_state = jax_next


def test_full_soup_cycle_delivers_reward():
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    Action = actions_mod.Action
    OvercookedGridworld = mdp_mod.OvercookedGridworld
    ObjectState = mdp_mod.ObjectState
    PlayerState = mdp_mod.PlayerState
    OvercookedState = mdp_mod.OvercookedState
    Direction = actions_mod.Direction

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)

    # Construct pre-soup-pickup state: ready soup in pot, p0 holding dish next to pot.
    # Pot in simple is at (2, 0), serve is at (3, 3).
    p0 = PlayerState(position=(2, 1), orientation=Direction.NORTH, held_object=ObjectState("dish", (2, 1)))
    p1 = PlayerState(position=(4, 1), orientation=Direction.WEST, held_object=None)
    objects = {(2, 0): ObjectState("soup", (2, 0), state=("onion", 3, 20))}
    legacy_state = OvercookedState(players=[p0, p1], objects=objects, order_list=None)
    jax_state = from_legacy_state(terrain, legacy_state)

    # Pickup soup from pot.
    legacy_state, jax_state, sparse, _ = _run_and_compare(
        mdp, terrain, legacy_state, jax_state, (Action.INTERACT, Action.STAY)
    )
    assert sparse == 0.0

    # Move to serving station and serve.
    sequence = [
        (Action.EAST, Action.STAY),   # (2,1)->(3,1)
        (Action.SOUTH, Action.STAY),  # (3,1)->(3,2)
        (Action.SOUTH, Action.STAY),  # face serve tile (3,3), blocked movement
        (Action.INTERACT, Action.STAY),  # serve
    ]
    final_sparse = 0.0
    for a in sequence:
        legacy_state, jax_state, final_sparse, _ = _run_and_compare(mdp, terrain, legacy_state, jax_state, a)
    assert final_sparse == 20.0


def test_collision_both_players_stay():
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    Action = actions_mod.Action
    Direction = actions_mod.Direction
    OvercookedGridworld = mdp_mod.OvercookedGridworld
    PlayerState = mdp_mod.PlayerState
    OvercookedState = mdp_mod.OvercookedState

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)
    legacy_state = OvercookedState(
        players=[
            PlayerState((1, 1), Direction.EAST, None),
            PlayerState((3, 1), Direction.WEST, None),
        ],
        objects={},
        order_list=None,
    )
    jax_state = from_legacy_state(terrain, legacy_state)
    legacy_next, jax_next, _, _ = _run_and_compare(
        mdp, terrain, legacy_state, jax_state, (Action.EAST, Action.WEST)
    )
    assert np.array_equal(np.asarray(jax_next.player_pos), np.asarray(from_legacy_state(terrain, legacy_state).player_pos))


def test_swap_collision_both_stay():
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    Action = actions_mod.Action
    Direction = actions_mod.Direction
    OvercookedGridworld = mdp_mod.OvercookedGridworld
    PlayerState = mdp_mod.PlayerState
    OvercookedState = mdp_mod.OvercookedState

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)
    legacy_state = OvercookedState(
        players=[
            PlayerState((1, 1), Direction.EAST, None),
            PlayerState((2, 1), Direction.WEST, None),
        ],
        objects={},
        order_list=None,
    )
    jax_state = from_legacy_state(terrain, legacy_state)
    legacy_next, jax_next, _, _ = _run_and_compare(
        mdp, terrain, legacy_state, jax_state, (Action.EAST, Action.WEST)
    )
    assert np.array_equal(np.asarray(jax_next.player_pos), np.asarray(from_legacy_state(terrain, legacy_state).player_pos))


def test_counter_place_and_pick():
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    Action = actions_mod.Action
    OvercookedGridworld = mdp_mod.OvercookedGridworld

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)
    legacy_state = mdp.get_standard_start_state()
    jax_state = from_legacy_state(terrain, legacy_state)

    actions = [
        (Action.NORTH, Action.STAY),      # p0 -> (1,1)
        (Action.WEST, Action.STAY),       # orient west toward onion dispenser
        (Action.INTERACT, Action.STAY),   # pickup onion
        (Action.SOUTH, Action.STAY),      # p0 -> (1,2)
        (Action.WEST, Action.STAY),       # orient west toward counter at (0,2)
        (Action.INTERACT, Action.STAY),   # place onion
        (Action.STAY, Action.WEST),       # p1 -> (2,1)
        (Action.STAY, Action.WEST),       # p1 -> (1,1)
        (Action.STAY, Action.SOUTH),      # p1 -> (1,2)
        (Action.STAY, Action.WEST),       # orient west toward counter
        (Action.STAY, Action.INTERACT),   # p1 picks onion
    ]
    for a in actions:
        legacy_state, jax_state, _, _ = _run_and_compare(mdp, terrain, legacy_state, jax_state, a)

    # p1 (index 1) should hold onion after pickup from counter.
    assert int(jax_state.held_obj[1]) == 1
