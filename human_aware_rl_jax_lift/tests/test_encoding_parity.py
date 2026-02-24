import numpy as np
import pytest

from human_aware_rl_jax_lift.env.compat import from_legacy_state
from human_aware_rl_jax_lift.encoding.ppo_masks import lossless_state_encoding_20
from human_aware_rl_jax_lift.env.layouts import parse_layout


def test_ppo_masks_value_parity_against_legacy():
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    Action = actions_mod.Action
    OvercookedGridworld = mdp_mod.OvercookedGridworld

    layout = "simple"
    terrain = parse_layout("simple")
    mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)
    legacy_state = mdp.get_standard_start_state()
    states = [legacy_state]
    rollout_actions = [
        (Action.STAY, Action.STAY),
        (Action.INTERACT, Action.INTERACT),
        (Action.EAST, Action.WEST),
    ]
    for joint_action in rollout_actions:
        legacy_state, _, _ = mdp.get_state_transition(legacy_state, joint_action)
        states.append(legacy_state)

    for st in states:
        jax_state = from_legacy_state(terrain, st)
        jax_o0, jax_o1 = lossless_state_encoding_20(terrain, jax_state)
        legacy_o0, legacy_o1 = mdp.lossless_state_encoding(st)
        assert np.array_equal(np.asarray(jax_o0), np.asarray(legacy_o0))
        assert np.array_equal(np.asarray(jax_o1), np.asarray(legacy_o1))
