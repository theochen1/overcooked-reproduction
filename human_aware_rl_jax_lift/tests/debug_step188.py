"""Debug shaped reward divergence at step 188."""
import numpy as np
import jax.numpy as jnp

from human_aware_rl_jax_lift.env.compat import from_legacy_state
from human_aware_rl_jax_lift.env.layouts import parse_layout
from human_aware_rl_jax_lift.env.overcooked_mdp import step as jax_step

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

layout = "simple"
terrain = parse_layout(layout)
mdp = OvercookedGridworld.from_layout_name(
    layout_name=layout, start_order_list=None, rew_shaping_params=_SHAPING_PARAMS
)
legacy_state = mdp.get_standard_start_state()
jax_state = from_legacy_state(terrain, legacy_state)

idx_to_act = Action.INDEX_TO_ACTION
rng = np.random.RandomState(42)

for t in range(200):
    a0_idx, a1_idx = rng.randint(0, 6, size=2)
    joint_action = (idx_to_act[a0_idx], idx_to_act[a1_idx])
    joint_idx = jnp.array([a0_idx, a1_idx], dtype=jnp.int32)

    legacy_next, legacy_sparse, legacy_shaped = mdp.get_state_transition(
        legacy_state, joint_action
    )
    jax_next, jax_sparse, jax_shaped, _ = jax_step(terrain, jax_state, joint_idx)

    j_sh = float(jax_shaped)
    if j_sh != float(legacy_shaped):
        print(f"\n*** Step {t}: shaped mismatch jax={j_sh} legacy={legacy_shaped} ***")
        print(f"  Actions: ({idx_to_act[a0_idx]}, {idx_to_act[a1_idx]})")
        print(f"  BEFORE:")
        print(f"    player_pos={np.asarray(jax_state.player_pos)}")
        print(f"    player_or ={np.asarray(jax_state.player_or)}")
        print(f"    held_obj  ={np.asarray(jax_state.held_obj)}")
        print(f"    pot_state ={np.asarray(jax_state.pot_state)}")
        print(f"    counter_obj={np.asarray(jax_state.counter_obj)}")
        print(f"    legacy held ={[p.held_object for p in legacy_state.players]}")
        print(f"    legacy objects={dict(legacy_state.objects)}")

        # Check pot states in TF
        pot_states = mdp.get_pot_states(legacy_state)
        ready = pot_states.get("onion", {}).get("ready", []) + pot_states.get("tomato", {}).get("ready", [])
        cooking = pot_states.get("onion", {}).get("cooking", []) + pot_states.get("tomato", {}).get("cooking", [])
        partial = []
        for st_type in ["onion", "tomato"]:
            for k, v in pot_states.get(st_type, {}).items():
                if k not in ["empty", "ready", "cooking"]:
                    partial.extend(v)
        nearly_ready = ready + cooking + partial
        print(f"    TF nearly_ready_pots={nearly_ready} (len={len(nearly_ready)})")
        dishes_held = [p for p in legacy_state.players if p.held_object and p.held_object.name == 'dish']
        print(f"    TF dishes_already={len(dishes_held)}")
        counter_objs = mdp.get_counter_objects_dict(legacy_state)
        print(f"    TF counter_objects={counter_objs}")
        break

    legacy_state = legacy_next
    jax_state = jax_next
else:
    print("All 200 steps match!")
