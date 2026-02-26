"""Decisive parity tests: JAX-lift vs legacy TF dynamics.

Three tests that isolate whether the true_eprew divergence is
(a) a real env/reward bug, or (b) just RNG stream differences.

Run with:
    pytest tests/test_decisive_parity.py -v -s
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from human_aware_rl_jax_lift.env.compat import from_legacy_state
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.overcooked_mdp import step as jax_step


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _import_legacy():
    actions_mod = pytest.importorskip("overcooked_ai_py.mdp.actions")
    mdp_mod = pytest.importorskip("overcooked_ai_py.mdp.overcooked_mdp")
    return actions_mod, mdp_mod


# Must match JAX's _BASE_*_REW constants so shaped rewards are comparable.
_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}


def _make_legacy_mdp(layout):
    """Create a legacy OvercookedGridworld with shaping params enabled."""
    _, mdp_mod = _import_legacy()
    return mdp_mod.OvercookedGridworld.from_layout_name(
        layout_name=layout,
        start_order_list=None,
        rew_shaping_params=_SHAPING_PARAMS,
    )


def _assert_state_equal(jax_s, legacy_s_as_jax, step_idx):
    for field in ("player_pos", "player_or", "held_obj", "held_soup",
                   "pot_state", "counter_obj", "counter_soup"):
        a = np.asarray(getattr(jax_s, field))
        b = np.asarray(getattr(legacy_s_as_jax, field))
        assert np.array_equal(a, b), (
            f"Step {step_idx}: state field '{field}' mismatch\n"
            f"  jax:    {a}\n  legacy: {b}"
        )


INDEX_TO_ACTION = None  # populated lazily

def _get_index_to_action():
    global INDEX_TO_ACTION
    if INDEX_TO_ACTION is None:
        actions_mod, _ = _import_legacy()
        INDEX_TO_ACTION = actions_mod.Action.INDEX_TO_ACTION
    return INDEX_TO_ACTION


# ---------------------------------------------------------------------------
# Test 1: Long random action sequence — step-by-step parity
# ---------------------------------------------------------------------------

LAYOUTS = ["simple", "unident_s", "random1", "random3"]
N_RANDOM_STEPS = 400  # full episode length


@pytest.mark.parametrize("layout", LAYOUTS)
def test_step_parity_random_sequence(layout):
    """Run N_RANDOM_STEPS with a deterministic random action sequence and
    assert full state + reward parity at every step."""
    terrain = parse_layout(layout)
    mdp = _make_legacy_mdp(layout)
    legacy_state = mdp.get_standard_start_state()
    jax_state = from_legacy_state(terrain, legacy_state)

    rng = np.random.RandomState(42)
    idx_to_act = _get_index_to_action()
    cum_sparse_jax, cum_sparse_legacy = 0.0, 0.0
    cum_shaped_jax, cum_shaped_legacy = 0.0, 0.0

    for t in range(N_RANDOM_STEPS):
        a0_idx, a1_idx = rng.randint(0, 6, size=2)
        joint_action = (idx_to_act[a0_idx], idx_to_act[a1_idx])
        joint_idx = jnp.array([a0_idx, a1_idx], dtype=jnp.int32)

        legacy_next, legacy_sparse, legacy_shaped = mdp.get_state_transition(
            legacy_state, joint_action
        )
        jax_next, jax_sparse, jax_shaped, _ = jax_step(terrain, jax_state, joint_idx)

        legacy_next_as_jax = from_legacy_state(terrain, legacy_next)
        _assert_state_equal(jax_next, legacy_next_as_jax, t)

        assert float(jax_sparse) == float(legacy_sparse), (
            f"Step {t}: sparse reward mismatch "
            f"jax={float(jax_sparse)} legacy={float(legacy_sparse)}"
        )
        assert float(jax_shaped) == float(legacy_shaped), (
            f"Step {t}: shaped reward mismatch "
            f"jax={float(jax_shaped)} legacy={float(legacy_shaped)}"
        )

        cum_sparse_jax += float(jax_sparse)
        cum_sparse_legacy += float(legacy_sparse)
        cum_shaped_jax += float(jax_shaped)
        cum_shaped_legacy += float(legacy_shaped)

        legacy_state = legacy_next
        jax_state = jax_next

    print(f"\n[{layout}] {N_RANDOM_STEPS} steps OK")
    print(f"  cumulative sparse: jax={cum_sparse_jax}  legacy={cum_sparse_legacy}")
    print(f"  cumulative shaped: jax={cum_shaped_jax}  legacy={cum_shaped_legacy}")

    assert cum_sparse_jax == cum_sparse_legacy, (
        f"Cumulative sparse mismatch over {N_RANDOM_STEPS} steps: "
        f"jax={cum_sparse_jax}  legacy={cum_sparse_legacy}"
    )
    assert cum_shaped_jax == cum_shaped_legacy, (
        f"Cumulative shaped mismatch over {N_RANDOM_STEPS} steps: "
        f"jax={cum_shaped_jax}  legacy={cum_shaped_legacy}"
    )


# ---------------------------------------------------------------------------
# Test 2: Uniform-policy rollout statistics (JAX only, for comparison)
# ---------------------------------------------------------------------------

def test_uniform_policy_statistics():
    """Run 30 envs × 400 steps with uniform-random actions on 'simple' layout.

    Prints per-episode sparse/shaped/total reward stats so you can compare
    against a TF uniform-policy run.  No assertion — purely diagnostic.
    """
    from human_aware_rl_jax_lift.env.vec_env import VectorizedEnv

    layout = "simple"
    terrain = parse_layout(layout)
    num_envs = 30
    horizon = 400
    seed = 2229

    vec_env = VectorizedEnv(
        terrain, num_envs, horizon,
        reward_shaping_params=None,
        randomize_agent_idx=False,
    )
    _, obs0, obs1, _ = vec_env.reset_all()

    rng = np.random.RandomState(seed)
    ep_sparse_returns = []
    ep_shaped_returns = []
    ep_total_returns = []

    for _ in range(horizon):
        ta = rng.randint(0, 6, size=num_envs).astype(np.int32)
        oa = rng.randint(0, 6, size=num_envs).astype(np.int32)
        step_out = vec_env.step_all(ta, oa)
        for info in step_out.infos:
            if "episode" in info:
                ep_sparse_returns.append(info["episode"]["ep_sparse_r"])
                ep_shaped_returns.append(info["episode"]["ep_shaped_r"])
                ep_total_returns.append(info["episode"]["r"])

    n_eps = len(ep_sparse_returns)
    print(f"\n[simple] Uniform-policy rollout: {n_eps} episodes completed")
    if n_eps:
        print(f"  ep_sparse_r  mean={np.mean(ep_sparse_returns):.4f}  "
              f"std={np.std(ep_sparse_returns):.4f}")
        print(f"  ep_shaped_r  mean={np.mean(ep_shaped_returns):.4f}  "
              f"std={np.std(ep_shaped_returns):.4f}")
        print(f"  ep_total_r   mean={np.mean(ep_total_returns):.4f}  "
              f"std={np.std(ep_total_returns):.4f}")
        print(f"  soups delivered total: {sum(ep_sparse_returns) / 20:.0f}")


# ---------------------------------------------------------------------------
# Test 3: Dish-pickup shaped reward parity (expected to FAIL → known delta)
# ---------------------------------------------------------------------------

def test_dish_pickup_shaped_reward_parity():
    """Dish pickup with no nearly-ready pots → both stacks give 0 shaped."""
    actions_mod, mdp_mod = _import_legacy()
    Action = actions_mod.Action
    Direction = actions_mod.Direction
    PlayerState = mdp_mod.PlayerState
    OvercookedState = mdp_mod.OvercookedState

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = _make_legacy_mdp(layout)

    # simple layout: dish dispenser at (1,3).  Walkable: (1,1),(2,1),(3,1),(1,2),(2,2),(3,2).
    # Player at (1,2) facing SOUTH interacts with (1,3) = dish dispenser.
    # No pots have items → nearly_ready_pots is empty → reward should be 0.
    p0 = PlayerState(position=(1, 2), orientation=Direction.SOUTH, held_object=None)
    p1 = PlayerState(position=(3, 2), orientation=Direction.WEST, held_object=None)
    legacy_state = OvercookedState(players=[p0, p1], objects={}, order_list=None)
    jax_state = from_legacy_state(terrain, legacy_state)

    joint_action = (Action.INTERACT, Action.STAY)
    joint_idx = jnp.array([Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32)

    legacy_next, legacy_sparse, legacy_shaped = mdp.get_state_transition(
        legacy_state, joint_action
    )
    jax_next, jax_sparse, jax_shaped, _ = jax_step(terrain, jax_state, joint_idx)

    print(f"\n[dish pickup, no nearly-ready pots]")
    print(f"  legacy shaped_r = {legacy_shaped}")
    print(f"  jax    shaped_r = {float(jax_shaped)}")

    assert legacy_sparse == 0 and float(jax_sparse) == 0, "Sparse should be 0"
    assert legacy_shaped == 0, "TF: no dish-pickup reward (no nearly-ready pots)"
    assert float(jax_shaped) == 0.0, "JAX should also give 0 (conditional check)"


def test_dish_pickup_with_ready_pot_parity():
    """When pots ARE nearly ready, both stacks should give dish-pickup reward."""
    actions_mod, mdp_mod = _import_legacy()
    Action = actions_mod.Action
    Direction = actions_mod.Direction
    PlayerState = mdp_mod.PlayerState
    OvercookedState = mdp_mod.OvercookedState
    ObjectState = mdp_mod.ObjectState

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = _make_legacy_mdp(layout)

    # Cooking pot at (2,0) with 3 onions, cook_time=0 → "cooking" bucket.
    # Player at (1,2) facing SOUTH → interact with dish dispenser at (1,3).
    p0 = PlayerState(position=(1, 2), orientation=Direction.SOUTH, held_object=None)
    p1 = PlayerState(position=(3, 2), orientation=Direction.WEST, held_object=None)
    objects = {(2, 0): ObjectState("soup", (2, 0), state=("onion", 3, 0))}
    legacy_state = OvercookedState(players=[p0, p1], objects=objects, order_list=None)
    jax_state = from_legacy_state(terrain, legacy_state)

    joint_action = (Action.INTERACT, Action.STAY)
    joint_idx = jnp.array([Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32)

    legacy_next, legacy_sparse, legacy_shaped = mdp.get_state_transition(
        legacy_state, joint_action
    )
    jax_next, jax_sparse, jax_shaped, _ = jax_step(terrain, jax_state, joint_idx)

    print(f"\n[dish pickup, with cooking pot]")
    print(f"  legacy shaped_r = {legacy_shaped}")
    print(f"  jax    shaped_r = {float(jax_shaped)}")

    assert legacy_shaped == 3, "TF: dish pickup with nearly-ready pot → +3"
    assert float(jax_shaped) == 3.0, "JAX: conditional check should also give +3"


# ---------------------------------------------------------------------------
# Test 4: Reward accounting decomposition audit
# ---------------------------------------------------------------------------

def test_reward_accounting_audit():
    """Run a scripted soup-delivery sequence on both stacks and verify the
    reward decomposition at each step.

    Prints a table showing (step, action, sparse, shaped) for manual audit.
    """
    actions_mod, mdp_mod = _import_legacy()
    Action = actions_mod.Action
    Direction = actions_mod.Direction
    PlayerState = mdp_mod.PlayerState
    OvercookedState = mdp_mod.OvercookedState
    ObjectState = mdp_mod.ObjectState

    layout = "simple"
    terrain = parse_layout(layout)
    mdp = _make_legacy_mdp(layout)

    # simple layout walkable: (1,1),(2,1),(3,1),(1,2),(2,2),(3,2).
    # p0 at (2,1) facing NORTH holding dish; p1 at (3,1) idle.
    # Pot at (2,0) has ready soup (3 onions, cook_time=20).
    p0 = PlayerState(
        position=(2, 1), orientation=Direction.NORTH,
        held_object=ObjectState("dish", (2, 1))
    )
    p1 = PlayerState(position=(3, 1), orientation=Direction.WEST, held_object=None)
    objects = {(2, 0): ObjectState("soup", (2, 0), state=("onion", 3, 20))}
    legacy_state = OvercookedState(players=[p0, p1], objects=objects, order_list=None)
    jax_state = from_legacy_state(terrain, legacy_state)

    sequence = [
        (Action.INTERACT, Action.STAY),    # pick up soup from pot
        (Direction.EAST, Action.STAY),     # move east
        (Direction.SOUTH, Action.STAY),    # move south
        (Direction.SOUTH, Action.STAY),    # face serve tile
        (Action.INTERACT, Action.STAY),    # serve soup → +20 sparse
    ]

    print(f"\n{'Step':>4}  {'Action':>20}  {'sparse_l':>9}  {'sparse_j':>9}  "
          f"{'shaped_l':>9}  {'shaped_j':>9}  {'match':>5}")
    print("-" * 85)

    all_match = True
    cum_sparse_l, cum_sparse_j = 0.0, 0.0
    cum_shaped_l, cum_shaped_j = 0.0, 0.0

    for t, joint_action in enumerate(sequence):
        joint_idx = jnp.array(
            [Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32
        )
        legacy_next, l_sparse, l_shaped = mdp.get_state_transition(
            legacy_state, joint_action
        )
        jax_next, j_sparse, j_shaped, _ = jax_step(terrain, jax_state, joint_idx)

        j_sparse, j_shaped = float(j_sparse), float(j_shaped)
        match = (l_sparse == j_sparse) and (l_shaped == j_shaped)
        all_match &= match

        cum_sparse_l += l_sparse
        cum_sparse_j += j_sparse
        cum_shaped_l += l_shaped
        cum_shaped_j += j_shaped

        act_str = f"({joint_action[0]}, {joint_action[1]})"
        print(f"{t:>4}  {act_str:>20}  {l_sparse:>9.1f}  {j_sparse:>9.1f}  "
              f"{l_shaped:>9.1f}  {j_shaped:>9.1f}  {'OK' if match else 'DIFF':>5}")

        legacy_state = legacy_next
        jax_state = jax_next

    print("-" * 85)
    print(f"{'SUM':>4}  {'':>20}  {cum_sparse_l:>9.1f}  {cum_sparse_j:>9.1f}  "
          f"{cum_shaped_l:>9.1f}  {cum_shaped_j:>9.1f}")

    assert cum_sparse_l == cum_sparse_j, f"Sparse mismatch: legacy={cum_sparse_l} jax={cum_sparse_j}"
    assert cum_shaped_l == cum_shaped_j, f"Shaped mismatch: legacy={cum_shaped_l} jax={cum_shaped_j}"
