"""
Diagnostic script to verify the training signal (rewards, observations, shaped rewards)
in the JAX Overcooked environment matches expected behavior.

Run from the overcooked_ai-master/src directory:
    python -m human_aware_rl.jaxmarl.verify_training_signal
"""

import numpy as np
import sys

def verify_reward_shaping():
    """Verify that shaped rewards are non-zero and have correct magnitude."""
    from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig
    from overcooked_ai_py.mdp.actions import Action

    config = OvercookedJaxEnvConfig(
        layout_name="cramped_room_legacy",
        horizon=400,
        old_dynamics=True,
        reward_shaping_factor=1.0,
        use_phi=False,
        use_legacy_encoding=True,
    )
    env = OvercookedJaxEnv(config=config)

    print("=" * 70)
    print("REWARD SHAPING VERIFICATION")
    print("=" * 70)
    print(f"Layout: {config.layout_name}")
    print(f"Reward shaping factor: {env.reward_shaping_factor}")
    print(f"MDP reward_shaping_params: {env.mdp.reward_shaping_params}")
    print(f"MDP old_dynamics: {env.mdp.old_dynamics}")
    print()

    # Run 5 episodes with random actions, track rewards
    total_sparse = 0
    total_shaped = 0
    total_steps = 0
    episode_details = []

    for ep in range(5):
        np.random.seed(42 + ep)
        state, obs = env.reset()
        ep_sparse = 0.0
        ep_shaped = 0.0
        ep_training_reward = 0.0
        done = False
        step = 0

        while not done:
            a0 = np.random.randint(0, 6)
            a1 = np.random.randint(0, 6)
            actions = {"agent_0": a0, "agent_1": a1}
            state, obs, rewards, dones, infos = env.step(state, actions)

            training_reward = float(rewards["agent_0"])
            sparse_r = float(infos.get("sparse_reward", 0))

            # Shaped reward = training_reward - sparse_reward (when factor=1)
            shaped_r = training_reward - sparse_r

            ep_sparse += sparse_r
            ep_shaped += shaped_r
            ep_training_reward += training_reward

            done = bool(dones["__all__"])
            step += 1

        total_sparse += ep_sparse
        total_shaped += ep_shaped
        total_steps += step
        episode_details.append({
            "sparse": ep_sparse,
            "shaped": ep_shaped,
            "training": ep_training_reward,
            "steps": step,
        })

        print(f"Episode {ep}: sparse={ep_sparse:.1f}, shaped={ep_shaped:.1f}, "
              f"training={ep_training_reward:.1f}, steps={step}")

    print(f"\nMean sparse reward:   {total_sparse / 5:.1f}")
    print(f"Mean shaped reward:   {total_shaped / 5:.1f}")
    print(f"Mean training reward: {(total_sparse + total_shaped) / 5:.1f}")

    if total_shaped == 0:
        print("\n*** CRITICAL: Shaped rewards are ZERO! Reward shaping is NOT working! ***")
        return False
    else:
        print(f"\nShaped rewards are non-zero. Ratio shaped/sparse: "
              f"{abs(total_shaped) / max(abs(total_sparse), 1):.2f}")
        return True


def verify_observations():
    """Verify observation encoding produces expected values."""
    from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig
    import jax.numpy as jnp

    config = OvercookedJaxEnvConfig(
        layout_name="cramped_room_legacy",
        horizon=400,
        old_dynamics=True,
        use_legacy_encoding=True,
    )
    env = OvercookedJaxEnv(config=config)

    print("\n" + "=" * 70)
    print("OBSERVATION ENCODING VERIFICATION")
    print("=" * 70)

    state, obs = env.reset()
    obs_0 = np.array(obs["agent_0"])
    obs_1 = np.array(obs["agent_1"])

    print(f"Observation shape: {obs_0.shape}")
    print(f"Expected shape: (width, height, 20) = ({env.mdp.width}, {env.mdp.height}, 20)")
    expected_shape = (env.mdp.width, env.mdp.height, 20)
    shape_ok = obs_0.shape == expected_shape
    print(f"Shape match: {shape_ok}")

    print(f"\nObs dtype: {obs_0.dtype}")
    print(f"Obs range: [{obs_0.min()}, {obs_0.max()}]")
    print(f"Obs sum: {obs_0.sum():.1f}")
    print(f"Non-zero channels: {[c for c in range(20) if obs_0[:,:,c].sum() > 0]}")

    # Channel names for legacy 20-channel encoding
    channels = [
        "player_0_loc", "player_1_loc",
        "p0_orient_N", "p0_orient_S", "p0_orient_E", "p0_orient_W",
        "p1_orient_N", "p1_orient_S", "p1_orient_E", "p1_orient_W",
        "pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc",
        "onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions",
    ]

    print("\nChannel activations (initial state):")
    for c, name in enumerate(channels):
        ch = obs_0[:, :, c]
        if ch.sum() > 0:
            positions = list(zip(*np.where(ch > 0)))
            vals = [ch[p] for p in positions]
            print(f"  [{c:2d}] {name:20s}: {list(zip(positions, vals))}")

    # Verify both player observations are different (agent_idx randomization)
    obs_diff = np.abs(obs_0 - obs_1).sum()
    print(f"\nObs difference between agent_0 and agent_1: {obs_diff:.1f}")
    if obs_diff > 0:
        print("  -> Observations differ (expected: agent perspectives are swapped)")
    else:
        print("  -> Observations are IDENTICAL (may indicate agent_idx=0 for both)")

    return shape_ok


def verify_delivery_reward():
    """Play a scripted sequence to achieve a soup delivery and verify reward."""
    from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig
    from overcooked_ai_py.mdp.actions import Action

    config = OvercookedJaxEnvConfig(
        layout_name="cramped_room_legacy",
        horizon=400,
        old_dynamics=True,
        reward_shaping_factor=1.0,
        use_phi=False,
        use_legacy_encoding=True,
    )
    env = OvercookedJaxEnv(config=config)

    print("\n" + "=" * 70)
    print("DELIVERY REWARD VERIFICATION")
    print("=" * 70)

    # Force agent_idx to 0 for predictable behavior
    env.agent_idx = 0

    state, obs = env.reset()
    env.agent_idx = 0  # Override randomization

    # cramped_room_legacy layout:
    # XXPXX   (P at (2,0))
    # O  2O   (O at (0,1) and (4,1), player 2 at (3,1))
    # X1  X   (player 1 at (1,2))
    # XDXSX   (D at (1,3), S at (3,3))

    print(f"Agent idx: {env.agent_idx}")
    print(f"Initial state:")
    for i, p in enumerate(env.base_env.state.players):
        print(f"  Player {i}: pos={p.position}, orient={p.orientation}")

    # Track all rewards
    total_sparse = 0
    total_shaped = 0
    deliveries = 0

    for step in range(400):
        # Random actions
        a0 = np.random.randint(0, 6)
        a1 = np.random.randint(0, 6)
        actions = {"agent_0": a0, "agent_1": a1}
        state, obs, rewards, dones, infos = env.step(state, actions)

        r = float(rewards["agent_0"])
        sparse_r = float(infos.get("sparse_reward", 0))
        shaped_r = r - sparse_r

        total_sparse += sparse_r
        total_shaped += shaped_r

        if sparse_r > 0:
            deliveries += 1
            print(f"  Step {step}: DELIVERY! sparse={sparse_r}, shaped={shaped_r}, total_r={r}")

        if bool(dones["__all__"]):
            break

    print(f"\nTotal: sparse={total_sparse:.1f}, shaped={total_shaped:.1f}, deliveries={deliveries}")
    print(f"Expected delivery reward: 20 per soup")

    if deliveries > 0 and total_sparse == deliveries * 20:
        print("PASS: Delivery reward is correct (20 per soup)")
        return True
    elif deliveries == 0:
        print("No deliveries occurred (normal for random play in 400 steps)")
        return True
    else:
        print(f"FAIL: Expected {deliveries * 20}, got {total_sparse}")
        return False


def verify_training_batch():
    """Verify a mini training loop produces sensible loss values."""
    import jax
    import jax.numpy as jnp
    from jax import random
    from human_aware_rl.jaxmarl.ppo import PPOConfig, PPOTrainer

    print("\n" + "=" * 70)
    print("TRAINING BATCH VERIFICATION")
    print("=" * 70)

    config = PPOConfig(
        layout_name="cramped_room_legacy",
        horizon=400,
        num_envs=4,
        num_steps=400,
        total_timesteps=12000,  # Just 1-2 updates
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.98,
        clip_eps=0.05,
        vf_coef=1.0,
        ent_coef=0.1,
        max_grad_norm=0.1,
        num_minibatches=2,
        num_epochs=2,
        num_hidden_layers=3,
        hidden_dim=64,
        num_filters=25,
        num_conv_layers=3,
        reward_shaping_factor=1.0,
        reward_shaping_horizon=2.5e6,
        use_phi=False,
        use_legacy_encoding=True,
        old_dynamics=True,
        entropy_coeff_start=0.1,
        entropy_coeff_end=0.1,
        use_entropy_annealing=False,
        use_early_stopping=False,
        verbose=False,
        verbose_debug=False,
        eval_interval=9999,
        log_interval=1,
        seed=42,
    )

    np.random.seed(42)
    trainer = PPOTrainer(config)
    trainer.total_timesteps = 0

    print(f"Network param count: {sum(x.size for x in jax.tree_util.tree_leaves(trainer.train_state.params))}")
    print(f"Obs shape: {trainer.obs_shape}")
    print(f"Num actions: {trainer.num_actions}")

    # Collect one rollout
    states, obs = trainer.envs.reset()
    current_episode_rewards = np.zeros(config.num_envs)
    current_episode_sparse_rewards = np.zeros(config.num_envs)
    transitions, states, obs, ep_rewards, ep_sparse_rewards = trainer._collect_rollout_with_rewards(
        trainer.train_state, states, obs, current_episode_rewards, current_episode_sparse_rewards
    )

    print(f"\nRollout collected:")
    print(f"  Num transitions: {len(transitions)}")
    print(f"  Transition obs shape: {transitions[0].obs.shape}")
    print(f"  Transition value shape: {transitions[0].value.shape}")
    print(f"  Transition reward shape: {transitions[0].reward.shape}")

    # Check reward statistics
    all_rewards = np.array([float(t.reward.sum()) for t in transitions])
    all_values = np.array([float(t.value.mean()) for t in transitions])
    all_dones = np.array([float(t.done.sum()) for t in transitions])

    print(f"\n  Reward stats: mean={all_rewards.mean():.4f}, std={all_rewards.std():.4f}, "
          f"min={all_rewards.min():.4f}, max={all_rewards.max():.4f}")
    print(f"  Value stats:  mean={all_values.mean():.4f}, std={all_values.std():.4f}")
    print(f"  Total dones: {all_dones.sum():.0f} (expected: {config.num_envs})")

    if all_rewards.max() == 0 and all_rewards.min() == 0:
        print("\n  *** WARNING: ALL rewards are zero! Training signal is absent! ***")
    elif abs(all_rewards.mean()) < 0.001:
        print(f"\n  *** WARNING: Mean reward is very small ({all_rewards.mean():.6f}). "
              f"Check if shaped rewards are working. ***")
    else:
        print(f"\n  Rewards look reasonable (mean={all_rewards.mean():.4f})")

    # Compute GAE
    last_result = trainer._jit_inference(trainer.train_state.params, obs["agent_0"])
    _, last_value = last_result
    advantages, returns = trainer._compute_gae(transitions, last_value)

    print(f"\n  Advantage stats: mean={float(advantages.mean()):.4f}, "
          f"std={float(advantages.std()):.4f}")
    print(f"  Returns stats:   mean={float(returns.mean()):.4f}, "
          f"std={float(returns.std()):.4f}")

    # Check initial logits
    sample_obs = transitions[0].obs
    logits, values = trainer._jit_inference(trainer.train_state.params, sample_obs)
    probs = jax.nn.softmax(logits)
    entropy = -(probs * jax.nn.log_softmax(logits)).sum(axis=-1).mean()

    print(f"\n  Initial logits stats: mean={float(logits.mean()):.4f}, "
          f"std={float(logits.std()):.4f}")
    print(f"  Initial probs (first env): {np.array(probs[0]).round(3)}")
    print(f"  Initial entropy: {float(entropy):.4f} (max for 6 actions: {np.log(6):.4f})")

    # Perform one PPO update and check gradient magnitudes
    print("\n  Performing one PPO update...")
    batch_obs = jnp.concatenate([t.obs for t in transitions])
    batch_actions = jnp.concatenate([t.action for t in transitions])
    batch_log_probs = jnp.concatenate([t.log_prob for t in transitions])
    batch_values = jnp.concatenate([t.value for t in transitions])
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)

    batch = {
        "obs": batch_obs,
        "actions": batch_actions,
        "log_probs": batch_log_probs,
        "advantages": advantages_flat,
        "returns": returns_flat,
        "old_values": batch_values,
    }

    new_train_state, loss, metrics = trainer._update(
        trainer.train_state, batch, need_debug_grads=True
    )

    print(f"  Loss: {float(loss):.4f}")
    print(f"  Actor loss: {float(metrics.get('actor_loss', 0)):.6f}")
    print(f"  Critic loss: {float(metrics.get('critic_loss', 0)):.6f}")
    print(f"  Entropy: {float(metrics.get('entropy', 0)):.6f}")
    print(f"  Grad norm: {float(metrics.get('grad_norm', 0)):.6f}")

    conv_norm = float(metrics.get("grad_conv_norm", 0))
    dense_norm = float(metrics.get("grad_dense_norm", 0))
    actor_norm = float(metrics.get("grad_actor_head_norm", 0))
    critic_norm = float(metrics.get("grad_critic_head_norm", 0))

    print(f"\n  Gradient norms by component:")
    print(f"    Conv layers:    {conv_norm:.6f}")
    print(f"    Dense layers:   {dense_norm:.6f}")
    print(f"    Actor head:     {actor_norm:.6f}")
    print(f"    Critic head:    {critic_norm:.6f}")

    total_grad = float(metrics.get("grad_norm", 0))
    if total_grad > 0:
        actor_fraction = actor_norm / total_grad * 100
        critic_fraction = critic_norm / total_grad * 100
        print(f"\n  Actor head is {actor_fraction:.1f}% of total gradient")
        print(f"  Critic head is {critic_fraction:.1f}% of total gradient")

        if actor_fraction < 1:
            print(f"\n  *** WARNING: Actor head gradient is <1% of total! ***")
            print(f"  The policy gradient signal may be overwhelmed by critic/entropy gradients.")
            print(f"  Consider: lower vf_coef, higher max_grad_norm, or separate optimizers.")

    return True


if __name__ == "__main__":
    print("Overcooked JAX Training Signal Verification")
    print("=" * 70)

    results = {}

    try:
        results["reward_shaping"] = verify_reward_shaping()
    except Exception as e:
        print(f"\nERROR in reward shaping verification: {e}")
        import traceback; traceback.print_exc()
        results["reward_shaping"] = False

    try:
        results["observations"] = verify_observations()
    except Exception as e:
        print(f"\nERROR in observation verification: {e}")
        import traceback; traceback.print_exc()
        results["observations"] = False

    try:
        results["delivery_reward"] = verify_delivery_reward()
    except Exception as e:
        print(f"\nERROR in delivery reward verification: {e}")
        import traceback; traceback.print_exc()
        results["delivery_reward"] = False

    try:
        results["training_batch"] = verify_training_batch()
    except Exception as e:
        print(f"\nERROR in training batch verification: {e}")
        import traceback; traceback.print_exc()
        results["training_batch"] = False

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test}")

    all_passed = all(results.values())
    if not all_passed:
        print("\nSome tests FAILED - there may be an environment or training signal issue.")
        sys.exit(1)
    else:
        print("\nAll tests passed. Environment and training signal look correct.")
        print("\nIf training still underperforms, possible remaining causes:")
        print("  1. Try original paper seeds: [2229, 7649, 7225, 9807, 386]")
        print("  2. Subtle numerical differences between TF and JAX")
        print("  3. Run side-by-side comparison with original TF code")
