"""Diagnostic: verify mlp-path features == dist_table-path features == TF features.

This test identifies train/eval feature mismatches that cause BC performance gaps.
Run with: JAX_PLATFORMS=cpu pytest tests/test_bc_feature_parity.py -v -s
"""
import numpy as np
import pytest

from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.state import make_initial_state
from human_aware_rl_jax_lift.encoding.bc_features import (
    build_dist_table,
    featurize_state_64,
    get_mlp_for_layout,
)


def _get_tf_mdp(layout):
    """Load legacy MDP for ground-truth feature comparison."""
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    planner_mod = pytest.importorskip("overcooked_ai_py.planning.planners")
    mdp = mdp_mod.OvercookedGridworld.from_layout_name(
        layout_name=layout, start_order_list=None
    )
    mlp = planner_mod.MediumLevelPlanner.from_pickle_or_compute(
        mdp, planner_mod.NO_COUNTERS_PARAMS, force_compute=False
    )
    return mdp, mlp


def _legacy_state_to_jax(terrain, legacy_state):
    from human_aware_rl_jax_lift.env.compat import from_legacy_state

    return from_legacy_state(terrain, legacy_state)


def _step_legacy(mdp, state, actions):
    """Step the legacy MDP through a sequence of actions and return all states."""
    from overcooked_ai_py.mdp.actions import Action, Direction

    action_map = {
        "N": Direction.NORTH,
        "S": Direction.SOUTH,
        "E": Direction.EAST,
        "W": Direction.WEST,
        "X": Action.INTERACT,
        ".": Action.STAY,
    }
    states = [state]
    for a0_str, a1_str in actions:
        a0 = action_map[a0_str]
        a1 = action_map[a1_str]
        state, _, _ = mdp.get_state_transition(state, (a0, a1))
        states.append(state)
    return states


# Short action sequences that exercise interesting states (onion pick/place, cooking)
_ACTION_SEQS = {
    "start": [],
    "p0_picks_onion": [("N", "."), ("W", "."), ("X", "."), ("E", ".")],
    "p0_places_in_pot": [
        ("N", "."),
        ("W", "."),
        ("X", "."),  # pick onion
        ("E", "."),
        ("E", "."),
        ("N", "."),
        ("X", "."),  # place in pot
    ],
    "both_move": [("N", "S"), ("E", "W"), ("S", "N")],
}


@pytest.mark.parametrize("layout", ["simple", "unident_s", "random0", "random1", "random3"])
@pytest.mark.parametrize("seq_name", list(_ACTION_SEQS.keys()))
def test_mlp_vs_dist_table(layout, seq_name):
    """Verify mlp path (data-prep) and dist_table path (eval) produce identical features."""
    terrain = parse_layout(layout)
    mlp = get_mlp_for_layout(layout)
    dist_table = build_dist_table(mlp, terrain)

    mdp, _ = _get_tf_mdp(layout)
    start_state = mdp.get_standard_start_state()

    try:
        states = _step_legacy(mdp, start_state, _ACTION_SEQS[seq_name])
    except Exception:
        pytest.skip(f"Could not step {layout} with action sequence '{seq_name}'")

    for step_i, legacy_state in enumerate(states):
        jax_state = _legacy_state_to_jax(terrain, legacy_state)

        # mlp path (used in data preparation)
        f0_mlp, f1_mlp = featurize_state_64(terrain, jax_state, mlp=mlp, tf_compat=True)
        # dist_table path (used in evaluation)
        f0_dt, f1_dt = featurize_state_64(terrain, jax_state, dist_table=dist_table, tf_compat=True)

        f0_mlp_np = np.asarray(f0_mlp, dtype=np.float32)
        f1_mlp_np = np.asarray(f1_mlp, dtype=np.float32)
        f0_dt_np = np.asarray(f0_dt, dtype=np.float32)
        f1_dt_np = np.asarray(f1_dt, dtype=np.float32)

        diff0 = np.abs(f0_mlp_np - f0_dt_np)
        diff1 = np.abs(f1_mlp_np - f1_dt_np)

        if np.max(diff0) > 1e-5 or np.max(diff1) > 1e-5:
            # Print detailed diff for debugging
            for dim in range(62):
                if abs(f0_mlp_np[dim] - f0_dt_np[dim]) > 1e-5:
                    print(
                        f"  [P0 step={step_i} dim={dim}] mlp={f0_mlp_np[dim]:.6f} "
                        f"dt={f0_dt_np[dim]:.6f} diff={diff0[dim]:.6f}"
                    )
                if abs(f1_mlp_np[dim] - f1_dt_np[dim]) > 1e-5:
                    print(
                        f"  [P1 step={step_i} dim={dim}] mlp={f1_mlp_np[dim]:.6f} "
                        f"dt={f1_dt_np[dim]:.6f} diff={diff1[dim]:.6f}"
                    )

        np.testing.assert_allclose(
            f0_mlp_np,
            f0_dt_np,
            atol=1e-5,
            err_msg=f"{layout}/{seq_name} step {step_i}: P0 mlp vs dist_table mismatch",
        )
        np.testing.assert_allclose(
            f1_mlp_np,
            f1_dt_np,
            atol=1e-5,
            err_msg=f"{layout}/{seq_name} step {step_i}: P1 mlp vs dist_table mismatch",
        )


@pytest.mark.parametrize("layout", ["simple", "unident_s", "random0", "random1", "random3"])
def test_jax_vs_tf_features(layout):
    """Verify JAX features (mlp path with tf_compat=True) match TF ground truth exactly."""
    terrain = parse_layout(layout)
    mlp = get_mlp_for_layout(layout)

    mdp, tf_mlp = _get_tf_mdp(layout)
    start_state = mdp.get_standard_start_state()

    # Get TF features
    tf_f0, tf_f1 = mdp.featurize_state(start_state, tf_mlp)
    tf_f0 = np.asarray(tf_f0, dtype=np.float32)
    tf_f1 = np.asarray(tf_f1, dtype=np.float32)

    # Get JAX features (mlp path, tf_compat=True to replicate abs_pos bug)
    jax_state = _legacy_state_to_jax(terrain, start_state)
    jax_f0, jax_f1 = featurize_state_64(terrain, jax_state, mlp=mlp, tf_compat=True)
    jax_f0 = np.asarray(jax_f0, dtype=np.float32)
    jax_f1 = np.asarray(jax_f1, dtype=np.float32)

    assert tf_f0.shape[0] == jax_f0.shape[0], (
        f"{layout}: TF dim={tf_f0.shape[0]}, JAX dim={jax_f0.shape[0]}"
    )

    diff0 = np.abs(tf_f0 - jax_f0)
    diff1 = np.abs(tf_f1 - jax_f1)

    if np.max(diff0) > 1e-5 or np.max(diff1) > 1e-5:
        print(f"\n{layout} TF vs JAX feature mismatches:")
        for dim in range(min(tf_f0.shape[0], jax_f0.shape[0])):
            if abs(tf_f0[dim] - jax_f0[dim]) > 1e-5:
                print(
                    f"  [P0 dim={dim}] tf={tf_f0[dim]:.6f} jax={jax_f0[dim]:.6f} "
                    f"diff={diff0[dim]:.6f}"
                )
            if abs(tf_f1[dim] - jax_f1[dim]) > 1e-5:
                print(
                    f"  [P1 dim={dim}] tf={tf_f1[dim]:.6f} jax={jax_f1[dim]:.6f} "
                    f"diff={diff1[dim]:.6f}"
                )

    np.testing.assert_allclose(
        jax_f0,
        tf_f0,
        atol=1e-5,
        err_msg=f"{layout}: P0 JAX vs TF feature mismatch",
    )
    np.testing.assert_allclose(
        jax_f1,
        tf_f1,
        atol=1e-5,
        err_msg=f"{layout}: P1 JAX vs TF feature mismatch",
    )


@pytest.mark.parametrize("layout", ["simple", "unident_s", "random0", "random1", "random3"])
def test_jax_vs_tf_features_after_steps(layout):
    """Same as above but after a few game steps to test non-initial states."""
    terrain = parse_layout(layout)
    mlp = get_mlp_for_layout(layout)

    mdp, tf_mlp = _get_tf_mdp(layout)
    start_state = mdp.get_standard_start_state()

    # Do a few steps
    states = _step_legacy(
        mdp, start_state, [("N", "S"), ("E", "W"), ("S", "N"), ("W", "E")]
    )

    for step_i, legacy_state in enumerate(states):
        tf_f0, tf_f1 = mdp.featurize_state(legacy_state, tf_mlp)
        tf_f0 = np.asarray(tf_f0, dtype=np.float32)
        tf_f1 = np.asarray(tf_f1, dtype=np.float32)

        jax_state = _legacy_state_to_jax(terrain, legacy_state)
        jax_f0, jax_f1 = featurize_state_64(terrain, jax_state, mlp=mlp, tf_compat=True)
        jax_f0 = np.asarray(jax_f0, dtype=np.float32)
        jax_f1 = np.asarray(jax_f1, dtype=np.float32)

        if tf_f0.shape[0] != jax_f0.shape[0]:
            pytest.fail(
                f"{layout} step {step_i}: TF dim={tf_f0.shape[0]}, JAX dim={jax_f0.shape[0]}"
            )

        diff0 = np.abs(tf_f0 - jax_f0)
        diff1 = np.abs(tf_f1 - jax_f1)

        if np.max(diff0) > 1e-5 or np.max(diff1) > 1e-5:
            print(f"\n{layout} step {step_i} TF vs JAX mismatches:")
            for dim in range(min(tf_f0.shape[0], jax_f0.shape[0])):
                if abs(tf_f0[dim] - jax_f0[dim]) > 1e-5:
                    print(
                        f"  [P0 dim={dim}] tf={tf_f0[dim]:.6f} jax={jax_f0[dim]:.6f}"
                    )
                if abs(tf_f1[dim] - jax_f1[dim]) > 1e-5:
                    print(
                        f"  [P1 dim={dim}] tf={tf_f1[dim]:.6f} jax={jax_f1[dim]:.6f}"
                    )

        np.testing.assert_allclose(
            jax_f0,
            tf_f0,
            atol=1e-5,
            err_msg=f"{layout} step {step_i}: P0 JAX vs TF feature mismatch",
        )
        np.testing.assert_allclose(
            jax_f1,
            tf_f1,
            atol=1e-5,
            err_msg=f"{layout} step {step_i}: P1 JAX vs TF feature mismatch",
        )
