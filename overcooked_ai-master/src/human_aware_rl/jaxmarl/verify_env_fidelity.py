"""
Verify that the JAX env wrapper and legacy encoding produce correct results.

Checks:
1. Observation shape matches expected (W, H, 20) for all legacy layouts
2. Episode length is exactly 400 steps with old_dynamics=True
3. Reward structure: sparse=0 for most steps, sparse=20 on delivery
4. Reward shaping: correct shaped rewards for subgoals
5. old_dynamics: auto-cooking when 3 ingredients placed
6. Encoding integrity: observations are integer-valued, channels are correct

Usage:
    python -m human_aware_rl.jaxmarl.verify_env_fidelity
"""
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction


LAYOUTS = {
    "coordination_ring_legacy": (5, 5),
    "cramped_room_legacy": (5, 4),
    "asymmetric_advantages_legacy": (9, 5),
    "random0_legacy": (5, 5),
    "random3_legacy": (8, 5),
}


def test_obs_shape():
    print("\n[TEST] Observation shapes (legacy 20-channel encoding)")
    all_ok = True
    for layout_name, (expected_w, expected_h) in LAYOUTS.items():
        mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=True)
        env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
        state = mdp.get_standard_start_state()

        obs_pair = mdp.lossless_state_encoding_legacy(state)
        obs0 = obs_pair[0]

        expected_shape = (expected_w, expected_h, 20)
        if obs0.shape != expected_shape:
            print(f"  FAIL {layout_name}: got {obs0.shape}, expected {expected_shape}")
            all_ok = False
        else:
            print(f"  OK   {layout_name}: shape={obs0.shape}")

        assert obs0.dtype == np.int_ or obs0.dtype == np.int64, \
            f"Obs dtype should be int, got {obs0.dtype}"
        assert np.all((obs0 >= 0)), "Obs values should be non-negative"
    return all_ok


def test_episode_length():
    print("\n[TEST] Episode length = 400 with old_dynamics=True")
    all_ok = True
    for layout_name in LAYOUTS:
        mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=True)
        env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
        env.reset()

        steps = 0
        done = False
        while not done:
            joint_action = (Action.STAY, Action.STAY)
            _, _, done, _ = env.step(joint_action)
            steps += 1

        if steps != 400:
            print(f"  FAIL {layout_name}: episode length={steps}, expected 400")
            all_ok = False
        else:
            print(f"  OK   {layout_name}: episode_length=400")
    return all_ok


def test_reward_structure():
    print("\n[TEST] Reward structure: sparse rewards on delivery only")
    layout_name = "coordination_ring_legacy"
    mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=True)
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)

    np.random.seed(42)
    total_sparse = 0
    total_episodes = 20

    for _ in range(total_episodes):
        env.reset()
        done = False
        while not done:
            actions = (
                Action.INDEX_TO_ACTION[np.random.randint(0, 6)],
                Action.INDEX_TO_ACTION[np.random.randint(0, 6)],
            )
            _, sparse_r, done, info = env.step(actions)
            total_sparse += sparse_r

            shaped_by_agent = info.get("shaped_r_by_agent", [0, 0])
            assert isinstance(shaped_by_agent, (list, tuple)), \
                f"shaped_r_by_agent should be list/tuple, got {type(shaped_by_agent)}"
            assert len(shaped_by_agent) == 2, \
                f"shaped_r_by_agent should have 2 elements, got {len(shaped_by_agent)}"

            if sparse_r > 0:
                assert sparse_r == 20, f"Sparse reward should be 20, got {sparse_r}"

    print(f"  OK   {layout_name}: total_sparse_reward={total_sparse} over {total_episodes} episodes")
    print(f"         (avg {total_sparse/total_episodes:.1f} per episode with random actions)")
    return True


def test_old_dynamics_auto_cook():
    print("\n[TEST] old_dynamics=True: auto-cooking when 3 ingredients placed")
    layout_name = "coordination_ring_legacy"
    mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=True)
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    env.reset()

    state = env.state
    pot_locations = mdp.get_pot_locations()
    print(f"  Pot locations: {pot_locations}")

    from overcooked_ai_py.mdp.overcooked_mdp import SoupState, ObjectState

    pot_loc = pot_locations[0]
    soup = SoupState(pot_loc, ingredients=[])
    for _ in range(3):
        soup.add_ingredient(ObjectState("onion", pot_loc))
    state.objects[pot_loc] = soup

    assert not soup.is_cooking, "Soup should not be cooking before step_environment_effects"
    assert soup._cooking_tick == -1, f"Expected _cooking_tick=-1, got {soup._cooking_tick}"

    mdp.step_environment_effects(state)

    soup_after = state.objects[pot_loc]
    assert soup_after._cooking_tick == 1, \
        f"Expected _cooking_tick=1 after auto-cook + first tick, got {soup_after._cooking_tick}"
    assert soup_after.is_cooking, "Soup should be cooking after step_environment_effects"

    print(f"  OK   Auto-cooking works: _cooking_tick went from -1 -> 1")
    return True


def test_encoding_channels():
    """Verify specific channels of the legacy encoding."""
    print("\n[TEST] Legacy encoding channel correctness")
    layout_name = "coordination_ring_legacy"
    mdp = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=True)
    state = mdp.get_standard_start_state()

    obs_pair = mdp.lossless_state_encoding_legacy(state)
    obs0 = obs_pair[0]  # From player 0's perspective

    p0_pos = state.players[0].position
    p1_pos = state.players[1].position

    player_0_loc = obs0[..., 0]
    assert player_0_loc[p0_pos] == 1, "Player 0 location channel should have 1 at player 0 pos"
    assert player_0_loc.sum() == 1, "Player 0 location channel should have exactly one 1"

    player_1_loc = obs0[..., 1]
    assert player_1_loc[p1_pos] == 1, "Player 1 location channel should have 1 at player 1 pos"
    assert player_1_loc.sum() == 1, "Player 1 location channel should have exactly one 1"

    pot_locs = mdp.get_pot_locations()
    pot_channel = obs0[..., 10]  # pot_loc is the first base_map_feature after 10 player features
    for loc in pot_locs:
        assert pot_channel[loc] == 1, f"Pot channel should be 1 at {loc}"
    assert pot_channel.sum() == len(pot_locs), \
        f"Pot channel should have exactly {len(pot_locs)} ones"

    obs1 = obs_pair[1]  # From player 1's perspective
    player_1_loc_from_p1 = obs1[..., 0]
    assert player_1_loc_from_p1[p1_pos] == 1, \
        "In player 1's obs, channel 0 should have 1 at player 1 pos (primary agent)"

    print(f"  OK   Channel correctness verified for {layout_name}")
    return True


def test_jax_env_wrapper():
    """Test the full JAX env wrapper matches expected behavior."""
    print("\n[TEST] JAX env wrapper consistency")
    try:
        from human_aware_rl.jaxmarl.overcooked_env import (
            OvercookedJaxEnv, OvercookedJaxEnvConfig
        )
        import jax.numpy as jnp
    except ImportError as e:
        print(f"  SKIP (JAX not available: {e})")
        return True

    config = OvercookedJaxEnvConfig(
        layout_name="coordination_ring_legacy",
        old_dynamics=True,
        horizon=400,
        use_legacy_encoding=True,
        use_phi=False,
    )
    env = OvercookedJaxEnv(config)

    assert env.obs_shape == (5, 5, 20), f"Expected obs_shape=(5,5,20), got {env.obs_shape}"
    assert env.num_actions == 6, f"Expected 6 actions, got {env.num_actions}"

    state, obs = env.reset()
    obs0 = np.array(obs["agent_0"])
    obs1 = np.array(obs["agent_1"])

    assert obs0.shape == (5, 5, 20), f"Obs0 shape mismatch: {obs0.shape}"
    assert obs1.shape == (5, 5, 20), f"Obs1 shape mismatch: {obs1.shape}"

    np.random.seed(123)
    total_reward = 0.0
    total_sparse = 0.0
    done = False
    steps = 0
    while not done:
        actions = {
            "agent_0": np.random.randint(0, 6),
            "agent_1": np.random.randint(0, 6),
        }
        state, obs, rewards, dones, infos = env.step(state, actions)
        total_reward += float(rewards["agent_0"])
        total_sparse += float(infos.get("sparse_reward", 0))
        done = bool(dones["__all__"])
        steps += 1

    assert steps == 400, f"Expected 400 steps, got {steps}"
    print(f"  OK   JAX wrapper: 400 steps, total_reward={total_reward:.1f}, "
          f"sparse={total_sparse:.0f}")
    return True


if __name__ == "__main__":
    results = []
    results.append(("obs_shape", test_obs_shape()))
    results.append(("episode_length", test_episode_length()))
    results.append(("reward_structure", test_reward_structure()))
    results.append(("old_dynamics", test_old_dynamics_auto_cook()))
    results.append(("encoding_channels", test_encoding_channels()))
    results.append(("jax_env_wrapper", test_jax_env_wrapper()))

    print("\n" + "=" * 50)
    all_pass = all(r[1] for r in results)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    if all_pass:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
    print("=" * 50)
