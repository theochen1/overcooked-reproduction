"""
Quick diagnostic test for AIRL with BC warm-start.
Runs 50 iterations to verify BC initialization is working and rewards are non-zero.
"""

import os
import sys
import torch
import numpy as np

from human_aware_rl.imitation.airl import AIRLConfig, AIRLTrainer, AIRL_SAVE_DIR
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN
from overcooked_ai_py.mdp.actions import Action


def test_bc_standalone(layout="cramped_room"):
    """Test that BC model itself gets rewards when run in the environment."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model, evaluate_bc_model
    from overcooked_ai_py.agents.benchmarking import AgentEvaluator
    from collections import Counter
    
    bc_dir = os.path.join(BC_SAVE_DIR, "train", layout)
    if not os.path.exists(os.path.join(bc_dir, "model.pt")):
        print(f"ERROR: No BC model found at {bc_dir}")
        print("Run: python -m human_aware_rl.imitation.train_bc_models --layout", layout)
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing BC model standalone: {layout}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bc_model, bc_params = load_bc_model(bc_dir, device=device)
    bc_model.eval()
    
    # Create environment
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400}
    )
    base_env = ae.env
    mdp = base_env.mdp
    
    # Test 1: Deterministic (argmax) - may deadlock
    state = mdp.get_standard_start_state()
    total_reward_det = 0
    action_counts_0 = Counter()
    action_counts_1 = Counter()
    
    for step in range(400):
        obs = base_env.featurize_state_mdp(state)
        obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits_0 = bc_model(obs_0)
            logits_1 = bc_model(obs_1)
            action_0 = torch.argmax(logits_0, dim=-1).item()
            action_1 = torch.argmax(logits_1, dim=-1).item()
        
        a0 = Action.INDEX_TO_ACTION[action_0]
        a1 = Action.INDEX_TO_ACTION[action_1]
        action_counts_0[str(a0)] += 1
        action_counts_1[str(a1)] += 1
        
        if step < 5:
            probs_0 = torch.softmax(logits_0, dim=-1).squeeze().tolist()
            print(f"  Step {step}: Agent0={a0}, Agent1={a1}, Probs={[f'{p:.2f}' for p in probs_0]}")
        
        joint_action = (a0, a1)
        next_state, info = mdp.get_state_transition(state, joint_action)
        env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
        total_reward_det += env_reward
        state = next_state
    
    print(f"\nAction counts (Agent 0): {dict(action_counts_0)}")
    print(f"Action counts (Agent 1): {dict(action_counts_1)}")
    print(f"BC self-play (DETERMINISTIC): Total reward = {total_reward_det}")
    
    # Test 2: Stochastic (how BC is actually evaluated!)
    state = mdp.get_standard_start_state()
    total_reward_stoch = 0
    
    for step in range(400):
        obs = base_env.featurize_state_mdp(state)
        obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits_0 = bc_model(obs_0)
            logits_1 = bc_model(obs_1)
            # STOCHASTIC sampling - how BC is actually used!
            probs_0 = torch.softmax(logits_0, dim=-1)
            probs_1 = torch.softmax(logits_1, dim=-1)
            action_0 = torch.multinomial(probs_0, 1).item()
            action_1 = torch.multinomial(probs_1, 1).item()
        
        joint_action = (Action.INDEX_TO_ACTION[action_0], Action.INDEX_TO_ACTION[action_1])
        next_state, info = mdp.get_state_transition(state, joint_action)
        env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
        total_reward_stoch += env_reward
        state = next_state
    
    print(f"BC self-play (STOCHASTIC): Total reward = {total_reward_stoch}")
    print(f"\n** NOTE: BC evaluation uses STOCHASTIC sampling, not deterministic! **")
    
    return total_reward_stoch


def test_airl_init(layout="cramped_room"):
    """Test AIRL policy right after BC initialization (no training)."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model
    from overcooked_ai_py.agents.benchmarking import AgentEvaluator
    
    bc_dir = os.path.join(BC_SAVE_DIR, "train", layout)
    if not os.path.exists(os.path.join(bc_dir, "model.pt")):
        print(f"ERROR: No BC model found at {bc_dir}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing AIRL after BC initialization (no training): {layout}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create minimal config
    config = AIRLConfig(
        layout_name=layout,
        horizon=400,
        old_dynamics=True,
        data_path=CLEAN_2019_HUMAN_DATA_TRAIN,
        bc_model_dir=bc_dir,
        total_timesteps=1000,  # Minimal
        steps_per_iter=100,
        verbose=True,
    )
    
    # Create trainer (this loads BC weights)
    trainer = AIRLTrainer(config)
    
    # Now test the AIRL policy (should behave like BC)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400}
    )
    base_env = ae.env
    mdp = base_env.mdp
    
    state = mdp.get_standard_start_state()
    total_reward_det = 0
    total_reward_stoch = 0
    
    # Test deterministic actions
    for step in range(400):
        obs = base_env.featurize_state_mdp(state)
        obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits_0, _ = trainer.policy(obs_0)
            logits_1, _ = trainer.policy(obs_1)
            action_0 = torch.argmax(logits_0, dim=-1).item()
            action_1 = torch.argmax(logits_1, dim=-1).item()
        
        joint_action = (Action.INDEX_TO_ACTION[action_0], Action.INDEX_TO_ACTION[action_1])
        next_state, info = mdp.get_state_transition(state, joint_action)
        
        env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
        total_reward_det += env_reward
        state = next_state
    
    print(f"AIRL policy (400 steps, DETERMINISTIC): Total reward = {total_reward_det}")
    
    # Test stochastic actions (like during training)
    state = mdp.get_standard_start_state()
    for step in range(400):
        obs = base_env.featurize_state_mdp(state)
        obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # This is what happens during training
            action_0, _, _ = trainer.policy.get_action(obs_0)
            action_1, _, _ = trainer.policy.get_action(obs_1)
            action_0 = action_0.item()
            action_1 = action_1.item()
        
        joint_action = (Action.INDEX_TO_ACTION[action_0], Action.INDEX_TO_ACTION[action_1])
        next_state, info = mdp.get_state_transition(state, joint_action)
        
        env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
        total_reward_stoch += env_reward
        state = next_state
    
    print(f"AIRL policy (400 steps, STOCHASTIC): Total reward = {total_reward_stoch}")
    
    return total_reward_det, total_reward_stoch


def run_mini_airl_training(layout="cramped_room", num_iters=50):
    """Run AIRL for just 50 iterations and track rewards."""
    bc_dir = os.path.join(BC_SAVE_DIR, "train", layout)
    if not os.path.exists(os.path.join(bc_dir, "model.pt")):
        print(f"ERROR: No BC model found at {bc_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Running mini AIRL training ({num_iters} iterations): {layout}")
    print(f"{'='*60}")
    
    # Very minimal training - just 50 iterations
    steps_per_iter = 400  # One episode per iter
    total_timesteps = num_iters * steps_per_iter
    
    config = AIRLConfig(
        layout_name=layout,
        horizon=400,
        old_dynamics=True,
        data_path=CLEAN_2019_HUMAN_DATA_TRAIN,
        bc_model_dir=bc_dir,
        # Minimal discriminator updates
        disc_updates_per_iter=1,
        policy_epochs=2,
        batch_size=64,
        # Start with very low entropy (BC-like)
        ent_coef=0.01,  # Almost deterministic
        ent_coef_final=0.01,  # Don't ramp up
        ent_warmup_iters=1000,  # Never ramp
        # Very conservative learning rates
        discriminator_lr=1e-5,
        policy_lr=1e-5,
        # Training length
        total_timesteps=total_timesteps,
        steps_per_iter=steps_per_iter,
        sample_buffer_size=10,
        verbose=True,
        log_interval=1,
        save_interval=100,  # Don't save
        results_dir="/tmp/airl_test",
        experiment_name="mini_test",
    )
    
    trainer = AIRLTrainer(config)
    
    print("\nStarting mini training...\n")
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("AIRL BC WARM-START DIAGNOSTIC TEST")
    print("="*70)
    
    # Test 1: BC standalone
    bc_reward = test_bc_standalone(args.layout)
    
    # Test 2: AIRL after BC init (no training)
    if bc_reward is not None:
        airl_rewards = test_airl_init(args.layout)
    
    # Test 3: Mini AIRL training
    if bc_reward is not None:
        run_mini_airl_training(args.layout, args.iters)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nExpected behavior:")
    print("  - BC standalone should get non-zero reward")
    print("  - AIRL deterministic should get similar reward to BC")
    print("  - AIRL stochastic may be slightly lower (due to sampling)")
    print("  - Mini training should show non-zero Env_R throughout")

