"""Run parity tests as a simple script (avoids pytest fork issues on login nodes)."""
import sys
import numpy as np
import jax.numpy as jnp

from human_aware_rl_jax_lift.env.compat import from_legacy_state
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.overcooked_mdp import step as jax_step

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld, OvercookedState, PlayerState, ObjectState,
)

_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

def make_mdp(layout):
    return OvercookedGridworld.from_layout_name(
        layout_name=layout, start_order_list=None, rew_shaping_params=_SHAPING_PARAMS
    )

def assert_state_equal(jax_s, legacy_s_as_jax, step_idx):
    for field in ("player_pos", "player_or", "held_obj", "held_soup",
                   "pot_state", "counter_obj", "counter_soup"):
        a = np.asarray(getattr(jax_s, field))
        b = np.asarray(getattr(legacy_s_as_jax, field))
        if not np.array_equal(a, b):
            print(f"  FAIL: Step {step_idx}: state field '{field}' mismatch")
            print(f"    jax:    {a}")
            print(f"    legacy: {b}")
            return False
    return True

passed = 0
failed = 0

# -------------------------------------------------------
# Test 1: 400-step random sequence (simple layout)
# -------------------------------------------------------
print("=" * 60)
print("Test 1: 400-step random sequence [simple]")
print("=" * 60)
layout = "simple"
terrain = parse_layout(layout)
mdp = make_mdp(layout)
legacy_state = mdp.get_standard_start_state()
jax_state = from_legacy_state(terrain, legacy_state)

rng = np.random.RandomState(42)
idx_to_act = Action.INDEX_TO_ACTION
ok = True
cum_sparse_j, cum_sparse_l = 0.0, 0.0
cum_shaped_j, cum_shaped_l = 0.0, 0.0

for t in range(400):
    a0, a1 = rng.randint(0, 6, size=2)
    joint_action = (idx_to_act[a0], idx_to_act[a1])
    joint_idx = jnp.array([a0, a1], dtype=jnp.int32)

    legacy_next, l_sparse, l_shaped = mdp.get_state_transition(legacy_state, joint_action)
    jax_next, j_sparse, j_shaped, _ = jax_step(terrain, jax_state, joint_idx)

    legacy_next_as_jax = from_legacy_state(terrain, legacy_next)
    if not assert_state_equal(jax_next, legacy_next_as_jax, t):
        ok = False
        break

    j_sp, j_sh = float(j_sparse), float(j_shaped)
    if j_sp != float(l_sparse):
        print(f"  FAIL: Step {t}: sparse reward mismatch jax={j_sp} legacy={l_sparse}")
        ok = False
        break
    if j_sh != float(l_shaped):
        print(f"  FAIL: Step {t}: shaped reward mismatch jax={j_sh} legacy={l_shaped}")
        ok = False
        break

    cum_sparse_j += j_sp
    cum_sparse_l += float(l_sparse)
    cum_shaped_j += j_sh
    cum_shaped_l += float(l_shaped)

    legacy_state = legacy_next
    jax_state = jax_next

if ok:
    print(f"  PASS: 400 steps OK")
    print(f"    sparse: jax={cum_sparse_j} legacy={cum_sparse_l}")
    print(f"    shaped: jax={cum_shaped_j} legacy={cum_shaped_l}")
    passed += 1
else:
    failed += 1

# -------------------------------------------------------
# Test 2: Dish pickup, no nearly-ready pots → 0 shaped
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Test 2: Dish pickup, no nearly-ready pots")
print("=" * 60)
terrain = parse_layout("simple")
mdp = make_mdp("simple")

p0 = PlayerState(position=(1, 2), orientation=Direction.SOUTH, held_object=None)
p1 = PlayerState(position=(3, 2), orientation=Direction.WEST, held_object=None)
legacy_state = OvercookedState(players=[p0, p1], objects={}, order_list=None)
jax_state = from_legacy_state(terrain, legacy_state)

joint_action = (Action.INTERACT, Action.STAY)
joint_idx = jnp.array([Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32)

legacy_next, l_sparse, l_shaped = mdp.get_state_transition(legacy_state, joint_action)
_, j_sparse, j_shaped, _ = jax_step(terrain, jax_state, joint_idx)

print(f"  legacy: sparse={l_sparse} shaped={l_shaped}")
print(f"  jax:    sparse={float(j_sparse)} shaped={float(j_shaped)}")

if l_shaped == 0 and float(j_shaped) == 0:
    print("  PASS")
    passed += 1
else:
    print("  FAIL")
    failed += 1

# -------------------------------------------------------
# Test 3: Dish pickup WITH nearly-ready pot → +3 shaped
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Test 3: Dish pickup with cooking pot")
print("=" * 60)
terrain = parse_layout("simple")
mdp = make_mdp("simple")

p0 = PlayerState(position=(1, 2), orientation=Direction.SOUTH, held_object=None)
p1 = PlayerState(position=(3, 2), orientation=Direction.WEST, held_object=None)
objects = {(2, 0): ObjectState("soup", (2, 0), state=("onion", 3, 0))}
legacy_state = OvercookedState(players=[p0, p1], objects=objects, order_list=None)
jax_state = from_legacy_state(terrain, legacy_state)

joint_action = (Action.INTERACT, Action.STAY)
joint_idx = jnp.array([Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32)

legacy_next, l_sparse, l_shaped = mdp.get_state_transition(legacy_state, joint_action)
_, j_sparse, j_shaped, _ = jax_step(terrain, jax_state, joint_idx)

print(f"  legacy: sparse={l_sparse} shaped={l_shaped}")
print(f"  jax:    sparse={float(j_sparse)} shaped={float(j_shaped)}")

if l_shaped == 3 and float(j_shaped) == 3.0:
    print("  PASS")
    passed += 1
else:
    print("  FAIL")
    failed += 1

# -------------------------------------------------------
# Test 4: Reward accounting audit (soup delivery)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Test 4: Reward accounting audit (soup delivery)")
print("=" * 60)
terrain = parse_layout("simple")
mdp = make_mdp("simple")

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

print(f"{'Step':>4}  {'sparse_l':>9}  {'sparse_j':>9}  {'shaped_l':>9}  {'shaped_j':>9}  {'match':>5}")
print("-" * 60)

ok = True
cum_sl, cum_sj, cum_shl, cum_shj = 0.0, 0.0, 0.0, 0.0

for t, joint_action in enumerate(sequence):
    joint_idx = jnp.array([Action.ACTION_TO_INDEX[a] for a in joint_action], dtype=jnp.int32)
    legacy_next, l_sparse, l_shaped = mdp.get_state_transition(legacy_state, joint_action)
    jax_next, j_sparse, j_shaped, _ = jax_step(terrain, jax_state, joint_idx)

    js, jsh = float(j_sparse), float(j_shaped)
    match = (l_sparse == js) and (l_shaped == jsh)
    ok &= match
    cum_sl += l_sparse; cum_sj += js; cum_shl += l_shaped; cum_shj += jsh
    print(f"{t:>4}  {l_sparse:>9.1f}  {js:>9.1f}  {l_shaped:>9.1f}  {jsh:>9.1f}  {'OK' if match else 'DIFF':>5}")
    legacy_state = legacy_next
    jax_state = jax_next

print("-" * 60)
print(f"SUM   {cum_sl:>9.1f}  {cum_sj:>9.1f}  {cum_shl:>9.1f}  {cum_shj:>9.1f}")

if ok and cum_sl == 20.0:
    print("  PASS")
    passed += 1
else:
    print("  FAIL")
    failed += 1

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)
