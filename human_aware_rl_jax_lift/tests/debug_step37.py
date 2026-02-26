"""Debug the counter_obj divergence at step 37."""
import numpy as np
import jax.numpy as jnp

from human_aware_rl_jax_lift.env.compat import from_legacy_state
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.overcooked_mdp import step as jax_step

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout = "simple"
terrain = parse_layout(layout)
mdp = OvercookedGridworld.from_layout_name(layout_name=layout, start_order_list=None)
legacy_state = mdp.get_standard_start_state()
jax_state = from_legacy_state(terrain, legacy_state)

idx_to_act = Action.INDEX_TO_ACTION
rng = np.random.RandomState(42)

for t in range(40):
    a0_idx, a1_idx = rng.randint(0, 6, size=2)
    joint_action = (idx_to_act[a0_idx], idx_to_act[a1_idx])
    joint_idx = jnp.array([a0_idx, a1_idx], dtype=jnp.int32)

    if t >= 34:
        print(f"\n--- Step {t}: actions=({idx_to_act[a0_idx]}, {idx_to_act[a1_idx]}) "
              f"[idx=({a0_idx}, {a1_idx})] ---")
        print(f"  BEFORE:")
        print(f"    jax  player_pos={np.asarray(jax_state.player_pos)}")
        print(f"    jax  player_or ={np.asarray(jax_state.player_or)}")
        print(f"    jax  held_obj  ={np.asarray(jax_state.held_obj)}")
        print(f"    jax  counter_obj={np.asarray(jax_state.counter_obj)}")
        print(f"    jax  pot_state ={np.asarray(jax_state.pot_state)}")
        print(f"    legacy player_pos={[p.position for p in legacy_state.players]}")
        print(f"    legacy player_or ={[p.orientation for p in legacy_state.players]}")
        print(f"    legacy held_obj  ={[p.held_object for p in legacy_state.players]}")
        print(f"    legacy objects   ={dict(legacy_state.objects)}")

    legacy_next, legacy_sparse, legacy_shaped = mdp.get_state_transition(
        legacy_state, joint_action
    )
    jax_next, jax_sparse, jax_shaped, _ = jax_step(terrain, jax_state, joint_idx)

    if t >= 34:
        print(f"  AFTER:")
        print(f"    jax  player_pos={np.asarray(jax_next.player_pos)}")
        print(f"    jax  held_obj  ={np.asarray(jax_next.held_obj)}")
        print(f"    jax  counter_obj={np.asarray(jax_next.counter_obj)}")
        print(f"    legacy player_pos={[p.position for p in legacy_next.players]}")
        print(f"    legacy held_obj  ={[p.held_object for p in legacy_next.players]}")
        print(f"    legacy objects   ={dict(legacy_next.objects)}")
        print(f"    sparse: jax={float(jax_sparse)} legacy={legacy_sparse}")
        print(f"    shaped: jax={float(jax_shaped)} legacy={legacy_shaped}")

    legacy_next_as_jax = from_legacy_state(terrain, legacy_next)
    jax_cobj = np.asarray(jax_next.counter_obj)
    leg_cobj = np.asarray(legacy_next_as_jax.counter_obj)
    if not np.array_equal(jax_cobj, leg_cobj):
        diff_idx = np.where(jax_cobj != leg_cobj)[0]
        for di in diff_idx:
            pos = np.asarray(terrain.counter_positions[di])
            print(f"\n  *** COUNTER DIVERGENCE at counter idx={di}, pos={pos}: "
                  f"jax={jax_cobj[di]} legacy={leg_cobj[di]} ***")
        break

    # Check other fields too
    for field in ("player_pos", "player_or", "held_obj", "held_soup", "pot_state"):
        a = np.asarray(getattr(jax_next, field))
        b = np.asarray(getattr(legacy_next_as_jax, field))
        if not np.array_equal(a, b):
            print(f"\n  *** {field} DIVERGENCE at step {t}: jax={a} legacy={b} ***")
            break

    legacy_state = legacy_next
    jax_state = jax_next
