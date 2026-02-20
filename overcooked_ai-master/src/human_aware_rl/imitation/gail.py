"""
GAIL (Generative Adversarial Imitation Learning) with KL Regularization to BC

Simpler than AIRL:
- Discriminator just classifies "expert" vs "policy" (no reward disentanglement)
- Policy trained with PPO to fool discriminator
- KL penalty keeps policy close to BC anchor

Goal: Better human proxy for PPO training
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR, load_bc_model


GAIL_SAVE_DIR = os.path.join(DATA_DIR, "gail_runs")

# Layout name mapping (paper name -> data name)
LAYOUT_TO_DATA = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "random0",  # Different name in data!
    "counter_circuit": "random3",  # Different name in data!
}

# Layout name mapping (paper name -> environment name)
LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",  # Different env name!
}


@dataclass
class GAILConfig:
    """Configuration for GAIL training."""
    
    # Environment
    layout_name: str = "cramped_room"
    horizon: int = 400
    old_dynamics: bool = True
    
    # Data
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    
    # BC anchor
    bc_model_dir: Optional[str] = None
    
    # Discriminator (SIMPLE - just classifies expert vs policy)
    disc_hidden_dim: int = 64
    disc_lr: float = 3e-4
    
    # Policy
    policy_hidden_dim: int = 64
    policy_lr: float = 3e-4
    
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    
    # KL regularization to BC (KEY for stability)
    kl_coef: float = 0.5
    kl_target: float = 0.02
    adaptive_kl: bool = True
    
    # Training
    total_timesteps: int = 500_000
    steps_per_iter: int = 400
    disc_updates_per_iter: int = 3
    
    # Logging
    log_interval: int = 1
    save_interval: int = 50
    verbose: bool = True
    
    # Output
    results_dir: str = GAIL_SAVE_DIR
    seed: int = 0


class GAILDiscriminator(nn.Module):
    """
    Simple GAIL discriminator.
    
    Just classifies (state, action) pairs as expert or policy.
    No reward disentanglement like AIRL.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Input: state + one-hot action
        input_dim = state_dim + action_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
            action: (batch,) action indices
            
        Returns:
            D(s, a) in (0, 1) - probability that (s, a) is from expert
        """
        # One-hot encode action
        action_onehot = F.one_hot(action, num_classes=6).float()
        
        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=-1)
        
        return torch.sigmoid(self.network(x))


class GAILPolicy(nn.Module):
    """Policy network for GAIL (same architecture as BC for compatibility)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return action, log_prob, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        logits, value = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        action_log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_prob, value, entropy


class GAILTrainer:
    """GAIL trainer with KL regularization to BC."""
    
    def __init__(self, config: GAILConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup
        self._setup_environment()
        self._load_expert_data()
        self._setup_networks()
        
        # KL coefficient (adaptive)
        self.current_kl_coef = config.kl_coef
        
        if config.verbose:
            print(f"GAIL Trainer initialized")
            print(f"  Device: {self.device}")
            print(f"  Expert transitions: {len(self.expert_states)}")
            print(f"  BC anchor: {'Yes' if self.bc_model is not None else 'No'}")
    
    def _setup_environment(self):
        """Setup Overcooked environment."""
        # Map layout name to environment name (some layouts have different env names)
        env_layout = LAYOUT_TO_ENV.get(self.config.layout_name, self.config.layout_name)
        
        mdp_params = {
            "layout_name": env_layout,  # Use environment layout name
            "old_dynamics": self.config.old_dynamics,
        }
        
        self.agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params=mdp_params,
            env_params=DEFAULT_ENV_PARAMS,
        )
        self.base_env = self.agent_evaluator.env
        self.mdp = self.base_env.mdp
        
        dummy_state = self.mdp.get_standard_start_state()
        self.obs_shape = self.base_env.featurize_state_mdp(dummy_state)[0].shape
        self.state_dim = int(np.prod(self.obs_shape))
        self.action_dim = len(Action.ALL_ACTIONS)
    
    def _load_expert_data(self):
        """Load human demonstration data."""
        # Map layout name to data name (some layouts have different names in data)
        data_layout = LAYOUT_TO_DATA.get(self.config.layout_name, self.config.layout_name)
        
        data_params = {
            "layouts": [data_layout],  # Use data layout name
            "check_trajectories": False,
            "featurize_states": True,
            "data_path": self.config.data_path,
        }
        
        if self.config.verbose:
            print(f"Loading expert data for {self.config.layout_name} (data: {data_layout})")
        
        processed_trajs = get_human_human_trajectories(**data_params, silent=not self.config.verbose)
        
        # Extract state-action pairs
        expert_states = []
        expert_actions = []
        
        ep_states = processed_trajs["ep_states"]
        ep_actions = processed_trajs["ep_actions"]
        
        for ep_idx in range(len(ep_states)):
            states = ep_states[ep_idx]
            actions = ep_actions[ep_idx]
            
            for t in range(len(states)):
                expert_states.append(states[t].flatten())
                expert_actions.append(int(actions[t].item()))
        
        self.expert_states = torch.tensor(np.array(expert_states), dtype=torch.float32, device=self.device)
        self.expert_actions = torch.tensor(expert_actions, dtype=torch.long, device=self.device)
        
        if self.config.verbose:
            print(f"Loaded {len(self.expert_states)} expert transitions")
    
    def _setup_networks(self):
        """Setup discriminator, policy, and BC anchor."""
        # Discriminator
        self.discriminator = GAILDiscriminator(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.disc_hidden_dim,
        ).to(self.device)
        
        # Policy
        self.policy = GAILPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.policy_hidden_dim,
        ).to(self.device)
        
        # Optimizers
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config.disc_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.policy_lr
        )
        
        # Load BC model as anchor
        self.bc_model = None
        if self.config.bc_model_dir and os.path.exists(os.path.join(self.config.bc_model_dir, "model.pt")):
            self._load_bc_anchor()
    
    def _load_bc_anchor(self):
        """Load BC model and initialize policy from it."""
        if self.config.verbose:
            print(f"Loading BC anchor from {self.config.bc_model_dir}")
        
        self.bc_model, _ = load_bc_model(self.config.bc_model_dir, device=self.device)
        self.bc_model.eval()
        for param in self.bc_model.parameters():
            param.requires_grad = False
        
        # Initialize policy from BC
        bc_state_dict = self.bc_model.state_dict()
        policy_state_dict = self.policy.state_dict()
        
        copied = 0
        for bc_key in bc_state_dict.keys():
            if 'network' in bc_key and 'network.4' not in bc_key:
                policy_key = bc_key.replace('network', 'feature_extractor')
                if policy_key in policy_state_dict:
                    if bc_state_dict[bc_key].shape == policy_state_dict[policy_key].shape:
                        policy_state_dict[policy_key] = bc_state_dict[bc_key].clone()
                        copied += 1
        
        if 'network.4.weight' in bc_state_dict:
            policy_state_dict['actor.weight'] = bc_state_dict['network.4.weight'].clone()
            policy_state_dict['actor.bias'] = bc_state_dict['network.4.bias'].clone()
            copied += 2
        
        self.policy.load_state_dict(policy_state_dict)
        
        if self.config.verbose:
            print(f"  Initialized policy from BC ({copied} params copied)")
    
    def _compute_kl_from_bc(self, states: torch.Tensor) -> torch.Tensor:
        """Compute KL(policy || BC)."""
        if self.bc_model is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            bc_logits = self.bc_model(states)
            bc_probs = F.softmax(bc_logits, dim=-1)
        
        policy_logits, _ = self.policy(states)
        policy_probs = F.softmax(policy_logits, dim=-1)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        
        kl = (policy_probs * (policy_log_probs - torch.log(bc_probs + 1e-8))).sum(dim=-1)
        return kl
    
    def _collect_rollout(self, num_steps: int) -> Dict[str, Any]:
        """Collect rollout with current policy."""
        states, actions, log_probs, values = [], [], [], []
        dones = []
        episode_rewards = []
        episode_reward = 0
        
        state = self.mdp.get_standard_start_state()
        obs = self.base_env.featurize_state_mdp(state)
        
        for step in range(num_steps):
            obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                action_0, log_prob_0, value_0 = self.policy.get_action(obs_0)
                action_1, _, _ = self.policy.get_action(obs_1)
            
            a0, a1 = action_0.item(), action_1.item()
            
            joint_action = (Action.INDEX_TO_ACTION[a0], Action.INDEX_TO_ACTION[a1])
            next_state, info = self.mdp.get_state_transition(state, joint_action)
            
            env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
            episode_reward += env_reward
            
            done = step >= self.config.horizon - 1
            
            states.append(obs_0.squeeze(0))
            actions.append(a0)
            log_probs.append(log_prob_0.item())
            values.append(value_0.item())
            dones.append(float(done))
            
            next_obs = self.base_env.featurize_state_mdp(next_state)
            state = next_state
            obs = next_obs
            
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state = self.mdp.get_standard_start_state()
                obs = self.base_env.featurize_state_mdp(state)
        
        return {
            "states": torch.stack(states),
            "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
            "log_probs": torch.tensor(log_probs, dtype=torch.float32, device=self.device),
            "values": torch.tensor(values, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
            "episode_rewards": episode_rewards,
        }
    
    def _update_discriminator(self, policy_states: torch.Tensor, policy_actions: torch.Tensor) -> Tuple[float, float]:
        """Update discriminator to classify expert vs policy."""
        batch_size = self.config.batch_size
        
        # Sample expert batch
        expert_idx = torch.randint(0, len(self.expert_states), (batch_size,))
        expert_s = self.expert_states[expert_idx]
        expert_a = self.expert_actions[expert_idx]
        
        # Sample policy batch
        policy_idx = torch.randint(0, len(policy_states), (batch_size,))
        policy_s = policy_states[policy_idx]
        policy_a = policy_actions[policy_idx]
        
        # Expert should be classified as 1, policy as 0
        expert_d = self.discriminator(expert_s, expert_a)
        policy_d = self.discriminator(policy_s, policy_a)
        
        # Binary cross-entropy loss with label smoothing
        expert_loss = -torch.log(expert_d + 1e-8).mean() * 0.9 - torch.log(1 - expert_d + 1e-8).mean() * 0.1
        policy_loss = -torch.log(1 - policy_d + 1e-8).mean() * 0.9 - torch.log(policy_d + 1e-8).mean() * 0.1
        
        disc_loss = expert_loss + policy_loss
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            expert_correct = (expert_d > 0.5).float().mean().item()
            policy_correct = (policy_d < 0.5).float().mean().item()
            accuracy = (expert_correct + policy_correct) / 2
        
        return disc_loss.item(), accuracy
    
    def _compute_gail_rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute GAIL reward: -log(1 - D(s, a))."""
        with torch.no_grad():
            d = self.discriminator(states, actions)
            # Reward for fooling discriminator
            rewards = -torch.log(1 - d + 1e-8).squeeze(-1)
        return rewards
    
    def _compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def _update_policy(self, rollout: Dict[str, Any]) -> Tuple[float, float]:
        """Update policy with PPO + GAIL reward + KL to BC."""
        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        dones = rollout["dones"]
        
        # Compute GAIL rewards
        gail_rewards = self._compute_gail_rewards(states, actions)
        
        # Compute advantages
        with torch.no_grad():
            _, last_value = self.policy(states[-1].unsqueeze(0))
            last_value = last_value.item()
        
        # Use GAIL rewards for advantages
        advantages, returns = self._compute_gae(
            gail_rewards, rollout["values"], dones, last_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        total_kl = 0
        num_updates = 0
        
        batch_size = len(states)
        minibatch_size = self.config.batch_size
        
        for epoch in range(self.config.ppo_epochs):
            perm = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                
                log_probs, values, entropy = self.policy.evaluate_actions(states[idx], actions[idx])
                
                # PPO loss
                ratio = torch.exp(log_probs - old_log_probs[idx])
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns[idx])
                
                # Entropy
                entropy_loss = -entropy.mean()
                
                # KL to BC (KEY for staying human-like)
                kl = self._compute_kl_from_bc(states[idx])
                kl_loss = kl.mean()
                total_kl += kl_loss.item()
                
                # Total loss
                loss = (
                    actor_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                    + self.current_kl_coef * kl_loss
                )
                
                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        mean_kl = total_kl / max(num_updates, 1)
        
        # Adaptive KL
        if self.config.adaptive_kl:
            if mean_kl > self.config.kl_target * 1.5:
                self.current_kl_coef = min(self.current_kl_coef * 1.5, 10.0)
            elif mean_kl < self.config.kl_target * 0.5:
                self.current_kl_coef = max(self.current_kl_coef / 1.5, 0.1)
        
        return total_loss / max(num_updates, 1), mean_kl
    
    def train(self) -> Dict[str, Any]:
        """Run GAIL training."""
        num_iters = self.config.total_timesteps // self.config.steps_per_iter
        total_timesteps = 0
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"GAIL Training with KL Regularization to BC")
            print(f"{'='*60}")
            print(f"Layout: {self.config.layout_name}")
            print(f"Total timesteps: {self.config.total_timesteps:,}")
            print(f"KL coef: {self.config.kl_coef} (target: {self.config.kl_target})")
            print()
        
        start_time = time.time()
        episode_rewards_history = []
        
        for iteration in range(num_iters):
            # Collect rollout
            rollout = self._collect_rollout(self.config.steps_per_iter)
            total_timesteps += self.config.steps_per_iter
            
            # Update discriminator
            disc_losses, disc_accs = [], []
            for _ in range(self.config.disc_updates_per_iter):
                d_loss, d_acc = self._update_discriminator(rollout["states"], rollout["actions"])
                disc_losses.append(d_loss)
                disc_accs.append(d_acc)
            
            # Update policy
            policy_loss, mean_kl = self._update_policy(rollout)
            
            # Track rewards
            episode_rewards = rollout["episode_rewards"]
            if episode_rewards:
                mean_reward = np.mean(episode_rewards)
                episode_rewards_history.extend(episode_rewards)
            else:
                mean_reward = 0
            
            # Logging
            if iteration % self.config.log_interval == 0 and self.config.verbose:
                elapsed = time.time() - start_time
                fps = total_timesteps / elapsed
                avg10 = np.mean(episode_rewards_history[-10:]) if episode_rewards_history else 0
                
                print(f"Iter {iteration}/{num_iters} | "
                      f"FPS: {fps:.0f} | "
                      f"D_acc: {np.mean(disc_accs):.2f} | "
                      f"KL: {mean_kl:.4f} (c={self.current_kl_coef:.2f}) | "
                      f"Env_R: {mean_reward:.0f} | "
                      f"Avg10: {avg10:.0f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self._save_checkpoint(iteration)
        
        # Final save
        self._save_checkpoint(num_iters)
        
        if self.config.verbose:
            print(f"\nTraining completed in {time.time() - start_time:.1f}s")
            if episode_rewards_history:
                print(f"Final avg reward (last 50): {np.mean(episode_rewards_history[-50:]):.1f}")
        
        return {"episode_rewards": episode_rewards_history}
    
    def _save_checkpoint(self, step: int):
        """Save checkpoint."""
        save_dir = os.path.join(self.config.results_dir, self.config.layout_name)
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "step": step,
        }, os.path.join(save_dir, "model.pt"))
        
        if self.config.verbose and step % (self.config.save_interval * 5) == 0:
            print(f"  Saved checkpoint to {save_dir}")


def train_gail(layout: str, bc_model_dir: str = None, verbose: bool = True, **kwargs):
    """Train GAIL for a layout."""
    if bc_model_dir is None:
        bc_model_dir = os.path.join(BC_SAVE_DIR, "train", layout)
    
    if not os.path.exists(os.path.join(bc_model_dir, "model.pt")):
        print(f"WARNING: No BC model found at {bc_model_dir}")
        print(f"GAIL will train without BC anchor (not recommended)")
        bc_model_dir = None
    
    config = GAILConfig(
        layout_name=layout,
        bc_model_dir=bc_model_dir,
        verbose=verbose,
        **kwargs
    )
    
    trainer = GAILTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train GAIL models for Overcooked AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layout", default="cramped_room",
                        help="Layout to train on")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--kl_coef", type=float, default=0.5,
                        help="KL regularization coefficient")
    parser.add_argument("--all_layouts", action="store_true",
                        help="Train all 5 paper layouts")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Output directory for GAIL models (default: DATA_DIR/gail_runs)")
    args = parser.parse_args()
    
    # Set results_dir in kwargs if specified
    kwargs = {"total_timesteps": args.timesteps, "kl_coef": args.kl_coef}
    if args.results_dir:
        kwargs["results_dir"] = args.results_dir
    
    if args.all_layouts:
        layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring", 
                   "forced_coordination", "counter_circuit"]
        for layout in layouts:
            print(f"\n{'='*60}")
            print(f"Training GAIL for {layout}")
            print(f"{'='*60}")
            train_gail(layout, **kwargs)
    else:
        train_gail(args.layout, **kwargs)

