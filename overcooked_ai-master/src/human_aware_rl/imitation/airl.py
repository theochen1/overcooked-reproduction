"""
Adversarial Inverse Reinforcement Learning (AIRL) for Overcooked AI.

This module implements AIRL from the paper:
"Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"
(Fu et al., 2018)

AIRL learns a disentangled reward function from demonstrations that can
produce more robust human proxy agents compared to Behavior Cloning.

Architecture:
    D(s, a, s') = exp(f(s, a, s')) / (exp(f(s, a, s')) + π(a|s))
    f(s, a, s') = g(s) + γ·h(s') - h(s)
    
Where:
    - g(s): Learned reward function (state-only for disentanglement)
    - h(s): Learned shaping potential (approximates value function)
    - π(a|s): Current policy
"""

import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN, CLEAN_2019_HUMAN_DATA_TEST
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator


#################
# Configuration #
#################

AIRL_SAVE_DIR = os.path.join(DATA_DIR, "airl_runs")

@dataclass
class AIRLConfig:
    """Configuration for AIRL training."""
    
    # Environment
    layout_name: str = "cramped_room"
    horizon: int = 400
    old_dynamics: bool = True
    
    # Data
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    featurize_states: bool = True
    
    # Discriminator architecture (SMALLER to prevent overfitting on scarce data)
    disc_hidden_dim: int = 32  # Reduced from 64 (AIRL paper uses 32)
    disc_num_layers: int = 2
    disc_g_linear: bool = True  # Linear g(s) for reward (paper recommendation)
    
    # Policy architecture
    policy_hidden_dim: int = 64
    policy_num_layers: int = 2
    use_lstm: bool = False
    cell_size: int = 256
    
    # BC warm-start and KL regularization (CRITICAL for scarce data)
    bc_model_dir: Optional[str] = None  # Path to pretrained BC model to initialize policy
    kl_coef: float = 0.5  # KL divergence penalty to stay close to BC
    kl_target: float = 0.01  # Target KL divergence (adaptive coefficient)
    use_adaptive_kl: bool = True  # Adaptively adjust kl_coef to hit kl_target
    
    # Training hyperparameters (TESTED: very conservative to preserve BC behavior)
    discriminator_lr: float = 1e-5  # Very slow discriminator
    policy_lr: float = 1e-5  # Very slow policy (fine-tuning from BC)
    gamma: float = 0.99
    batch_size: int = 64  # Small batch
    disc_updates_per_iter: int = 1  # Minimal discriminator updates
    policy_epochs: int = 2  # Minimal policy updates (preserves BC behavior)
    
    # PPO hyperparameters (LOW entropy - stochastic rollouts provide exploration)
    clip_eps: float = 0.1  # Small clipping for stability
    vf_coef: float = 0.5
    ent_coef: float = 0.01  # LOW entropy (tested: works with stochastic rollouts)
    ent_coef_final: float = 0.05  # Slight ramp for later exploration
    ent_warmup_iters: int = 200  # Slow ramp
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    
    # Regularization
    label_smoothing: float = 0.2  # Higher smoothing (expert=0.8, policy=0.2)
    grad_penalty_weight: float = 10.0
    weight_decay: float = 1e-4  # L2 regularization
    
    # Sample mixing
    sample_buffer_size: int = 50  # Keep history
    
    # Training length
    total_timesteps: int = 500_000  # 500K timesteps
    steps_per_iter: int = 400  # One episode per iteration (tested: works)
    
    # Logging and saving
    log_interval: int = 1
    save_interval: int = 20
    verbose: bool = True
    
    # Output
    results_dir: str = "results/airl"
    experiment_name: str = "airl_overcooked"
    seed: int = 0


##############
# Discriminator #
##############


class RewardNetwork(nn.Module):
    """
    Reward function g(s) - learns the disentangled reward from state only.
    Can be linear (as recommended in paper) or MLP.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_linear: bool = True,
    ):
        super().__init__()
        
        self.use_linear = use_linear
        
        if use_linear:
            # Linear g(s) as recommended in paper for disentanglement
            self.network = nn.Linear(input_dim, 1)
        else:
            # MLP for more expressive reward
            layers = []
            prev_dim = input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for states.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Reward tensor of shape (batch_size,)
        """
        return self.network(state).squeeze(-1)


class ShapingNetwork(nn.Module):
    """
    Shaping potential h(s) - approximates the value function.
    Used to compute the shaping term γ·h(s') - h(s).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute shaping potential for states.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Potential tensor of shape (batch_size,)
        """
        return self.network(state).squeeze(-1)


class AIRLDiscriminator(nn.Module):
    """
    AIRL Discriminator with disentangled reward structure.
    
    D(s, a, s') = exp(f(s, a, s')) / (exp(f(s, a, s')) + π(a|s))
    f(s, a, s') = g(s) + γ·h(s') - h(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        num_layers: int = 2,
        g_linear: bool = True,
    ):
        super().__init__()
        
        self.gamma = gamma
        
        # Reward function g(s)
        self.g_network = RewardNetwork(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_linear=g_linear,
        )
        
        # Shaping potential h(s)
        self.h_network = ShapingNetwork(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    
    def compute_f(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute f(s, a, s') = g(s) + γ·h(s') - h(s).
        
        Args:
            state: Current state tensor
            next_state: Next state tensor
            done: Done mask (1 if terminal, 0 otherwise)
            
        Returns:
            f values of shape (batch_size,)
        """
        g = self.g_network(state)
        h_s = self.h_network(state)
        h_s_prime = self.h_network(next_state)
        
        # Shaping: γ·h(s') - h(s), with h(s')=0 if terminal
        shaping = self.gamma * h_s_prime * (1 - done) - h_s
        
        return g + shaping
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator output D(s, a, s').
        
        Args:
            state: Current state tensor
            next_state: Next state tensor
            done: Done mask
            log_pi: Log probability of action under policy π(a|s)
            
        Returns:
            Discriminator output D in [0, 1]
        """
        f = self.compute_f(state, next_state, done)
        
        # D = exp(f) / (exp(f) + π(a|s))
        # Using log-sum-exp trick for numerical stability:
        # D = sigmoid(f - log_pi)
        return torch.sigmoid(f - log_pi)
    
    def get_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute AIRL reward for policy training.
        
        reward = log(D) - log(1-D) = f - log_pi
        
        This is equivalent to the advantage of the optimal policy.
        """
        f = self.compute_f(state, next_state, done)
        return f - log_pi
    
    def get_learned_reward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the learned reward g(s) for a state.
        This is the disentangled reward that transfers across dynamics.
        """
        return self.g_network(state)


##############
# Policy Network #
##############


class AIRLPolicy(nn.Module):
    """
    Policy network for AIRL.
    Maps states to action probabilities.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (for PPO updates)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.feature_extractor(state)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, return argmax action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.log(probs[torch.arange(len(action)), action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under the policy.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Log probability of taken action
        action_log_prob = log_probs[torch.arange(len(action)), action]
        
        # Entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_prob, value, entropy


class AIRLPolicyLSTM(nn.Module):
    """
    LSTM-based policy network for AIRL.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cell_size: int = 256,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.cell_size = cell_size
        
        # Feature extractor before LSTM
        layers = []
        prev_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=cell_size,
            batch_first=True,
        )
        
        # Actor and critic heads
        self.actor = nn.Linear(cell_size, action_dim)
        self.critic = nn.Linear(cell_size, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM policy.
        """
        # Feature extraction
        features = self.feature_extractor(state)
        
        # Add sequence dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        # LSTM
        if hidden is None:
            batch_size = features.size(0)
            hidden = self.get_initial_state(batch_size, features.device)
        
        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out.squeeze(1)
        
        # Heads
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out).squeeze(-1)
        
        return logits, value, new_hidden
    
    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial LSTM hidden state."""
        return (
            torch.zeros(1, batch_size, self.cell_size, device=device),
            torch.zeros(1, batch_size, self.cell_size, device=device),
        )
    
    def get_action(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample action from LSTM policy."""
        logits, value, new_hidden = self.forward(state, hidden)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.log(probs[torch.arange(len(action)), action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, new_hidden


##############
# AIRL Trainer #
##############


class AIRLTrainer:
    """
    AIRL Training loop.
    
    Alternates between:
    1. Discriminator update (distinguish expert vs policy)
    2. Policy update (PPO with AIRL reward)
    """
    
    def __init__(
        self,
        config: AIRLConfig,
        device: Optional[str] = None,
    ):
        self.config = config
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Setup environment
        self._setup_environment()
        
        # Load expert demonstrations
        self._load_expert_data()
        
        # Create networks
        self._create_networks()
        
        # Sample buffer for mixing
        self.policy_sample_buffer = deque(maxlen=config.sample_buffer_size)
        
        # BC model reference (for KL regularization)
        self.bc_model = None
        
        # Logging
        self.train_info = {
            "disc_losses": [],
            "policy_losses": [],
            "episode_rewards": [],
            "disc_accuracy": [],
        }
        
        # Create output directory
        os.makedirs(config.results_dir, exist_ok=True)
    
    def _setup_environment(self):
        """Setup the Overcooked environment."""
        mdp_params = {
            "layout_name": self.config.layout_name,
            "old_dynamics": self.config.old_dynamics,
        }
        
        self.agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params=mdp_params,
            env_params=DEFAULT_ENV_PARAMS,
        )
        self.base_env = self.agent_evaluator.env
        self.mdp = self.base_env.mdp
        
        # Get observation shape
        dummy_state = self.mdp.get_standard_start_state()
        self.obs_shape = self.base_env.featurize_state_mdp(dummy_state)[0].shape
        self.state_dim = int(np.prod(self.obs_shape))
        self.action_dim = len(Action.ALL_ACTIONS)
    
    def _load_expert_data(self):
        """Load and preprocess expert demonstrations."""
        data_params = {
            "layouts": [self.config.layout_name],
            "check_trajectories": False,
            "featurize_states": self.config.featurize_states,
            "data_path": self.config.data_path,
        }
        
        processed_trajs = get_human_human_trajectories(**data_params, silent=not self.config.verbose)
        
        # Flatten all episodes into (s, a, s', done) tuples
        expert_states = []
        expert_actions = []
        expert_next_states = []
        expert_dones = []
        
        ep_states = processed_trajs["ep_states"]
        ep_actions = processed_trajs["ep_actions"]
        
        for ep_idx in range(len(ep_states)):
            states = ep_states[ep_idx]
            actions = ep_actions[ep_idx]
            
            for t in range(len(states) - 1):
                expert_states.append(states[t].flatten())
                expert_actions.append(int(actions[t]))
                expert_next_states.append(states[t + 1].flatten())
                expert_dones.append(0.0)
            
            # Last transition (done=1)
            if len(states) > 0:
                expert_states.append(states[-1].flatten())
                expert_actions.append(int(actions[-1]) if len(actions) > 0 else 0)
                expert_next_states.append(states[-1].flatten())  # Terminal
                expert_dones.append(1.0)
        
        self.expert_states = torch.tensor(
            np.array(expert_states), dtype=torch.float32, device=self.device
        )
        self.expert_actions = torch.tensor(
            expert_actions, dtype=torch.long, device=self.device
        )
        self.expert_next_states = torch.tensor(
            np.array(expert_next_states), dtype=torch.float32, device=self.device
        )
        self.expert_dones = torch.tensor(
            expert_dones, dtype=torch.float32, device=self.device
        )
        
        if self.config.verbose:
            print(f"Loaded {len(self.expert_states)} expert transitions")
    
    def _create_networks(self):
        """Create discriminator and policy networks."""
        # Discriminator
        self.discriminator = AIRLDiscriminator(
            state_dim=self.state_dim,
            gamma=self.config.gamma,
            hidden_dim=self.config.disc_hidden_dim,
            num_layers=self.config.disc_num_layers,
            g_linear=self.config.disc_g_linear,
        ).to(self.device)
        
        # Policy
        if self.config.use_lstm:
            self.policy = AIRLPolicyLSTM(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config.policy_hidden_dim,
                num_layers=self.config.policy_num_layers,
                cell_size=self.config.cell_size,
            ).to(self.device)
        else:
            self.policy = AIRLPolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config.policy_hidden_dim,
                num_layers=self.config.policy_num_layers,
            ).to(self.device)
        
        # BC WARM-START: Initialize policy from pretrained BC model
        if self.config.bc_model_dir is not None:
            self._load_bc_weights()
        
        # Optimizers with weight decay for regularization
        weight_decay = getattr(self.config, 'weight_decay', 1e-4)
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.discriminator_lr,
            weight_decay=weight_decay,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.policy_lr,
            weight_decay=weight_decay,
        )
    
    def _load_bc_weights(self):
        """Load weights from a pretrained BC model to initialize the AIRL policy and keep as anchor."""
        from human_aware_rl.imitation.behavior_cloning import load_bc_model
        
        if self.config.verbose:
            print(f"Loading BC model from {self.config.bc_model_dir} for warm-start and KL anchor...")
        
        try:
            bc_model, bc_params = load_bc_model(self.config.bc_model_dir, device=self.device)
            
            # STORE BC MODEL AS FROZEN REFERENCE FOR KL REGULARIZATION
            self.bc_model = bc_model
            self.bc_model.eval()
            for param in self.bc_model.parameters():
                param.requires_grad = False
            
            if self.config.verbose:
                print(f"  BC model stored as frozen anchor for KL regularization")
            
            # Copy weights from BC model to AIRL policy's actor
            bc_state_dict = bc_model.state_dict()
            airl_state_dict = self.policy.state_dict()
            
            if self.config.verbose:
                print(f"  BC model keys: {list(bc_state_dict.keys())}")
                print(f"  AIRL policy keys: {list(airl_state_dict.keys())}")
            
            copied = 0
            
            # Copy hidden layers: network.0 -> feature_extractor.0, network.2 -> feature_extractor.2
            for bc_key in bc_state_dict.keys():
                if 'network' in bc_key and 'network.4' not in bc_key:
                    airl_key = bc_key.replace('network', 'feature_extractor')
                    if airl_key in airl_state_dict:
                        if bc_state_dict[bc_key].shape == airl_state_dict[airl_key].shape:
                            airl_state_dict[airl_key] = bc_state_dict[bc_key].clone()
                            copied += 1
                            if self.config.verbose:
                                print(f"  Copied {bc_key} -> {airl_key}")
                        else:
                            print(f"  Shape mismatch: {bc_key} {bc_state_dict[bc_key].shape} vs {airl_key} {airl_state_dict[airl_key].shape}")
            
            # Copy output layer: network.4 -> actor
            if 'network.4.weight' in bc_state_dict and 'actor.weight' in airl_state_dict:
                if bc_state_dict['network.4.weight'].shape == airl_state_dict['actor.weight'].shape:
                    airl_state_dict['actor.weight'] = bc_state_dict['network.4.weight'].clone()
                    airl_state_dict['actor.bias'] = bc_state_dict['network.4.bias'].clone()
                    copied += 2
                    if self.config.verbose:
                        print(f"  Copied network.4 -> actor")
                else:
                    print(f"  Shape mismatch: network.4 {bc_state_dict['network.4.weight'].shape} vs actor {airl_state_dict['actor.weight'].shape}")
            
            self.policy.load_state_dict(airl_state_dict)
            
            if self.config.verbose:
                print(f"Successfully initialized AIRL policy from BC ({copied} parameters copied)")
                self._verify_bc_initialization()
                
        except Exception as e:
            import traceback
            print(f"Warning: Could not load BC model for warm-start: {e}")
            traceback.print_exc()
            print("Continuing with random initialization (NO KL regularization)...")
            self.bc_model = None
    
    def _verify_bc_initialization(self):
        """Run a quick test to verify BC initialization works."""
        print("\n  Verifying BC initialization with test rollout...")
        
        # Do a short deterministic rollout
        state = self.mdp.get_standard_start_state()
        obs = self.base_env.featurize_state_mdp(state)
        
        total_reward = 0
        for step in range(100):  # Short test
            obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                # Use DETERMINISTIC actions for verification
                logits_0, _ = self.policy(obs_0)
                logits_1, _ = self.policy(obs_1)
                action_0 = torch.argmax(logits_0, dim=-1).item()
                action_1 = torch.argmax(logits_1, dim=-1).item()
            
            joint_action = (Action.INDEX_TO_ACTION[action_0], Action.INDEX_TO_ACTION[action_1])
            next_state, info = self.base_env.mdp.get_state_transition(state, joint_action)
            
            env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
            total_reward += env_reward
            
            state = next_state
            obs = self.base_env.featurize_state_mdp(state)
        
        print(f"  Test rollout (100 steps, deterministic): reward = {total_reward}")
        if total_reward == 0:
            print("  WARNING: BC-initialized policy got 0 reward in test. Check if BC model is trained.")
    
    def _collect_rollout(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Collect a rollout from the environment using current policy."""
        states_list = []
        actions_list = []
        next_states_list = []
        dones_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []  # Environment rewards (for monitoring)
        
        # Reset environment
        state = self.mdp.get_standard_start_state()
        obs = self.base_env.featurize_state_mdp(state)
        
        episode_reward = 0
        episode_rewards = []
        
        hidden = None
        if self.config.use_lstm:
            hidden = self.policy.get_initial_state(1, torch.device(self.device))
        
        for step in range(num_steps):
            # Get observations for both agents
            obs_0 = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_1 = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Get actions from policy (we train agent 0, agent 1 uses same policy for self-play)
            with torch.no_grad():
                if self.config.use_lstm:
                    action_0, log_prob_0, value_0, hidden = self.policy.get_action(obs_0, hidden)
                    action_1, _, _, _ = self.policy.get_action(obs_1, hidden)
                else:
                    action_0, log_prob_0, value_0 = self.policy.get_action(obs_0)
                    action_1, _, _ = self.policy.get_action(obs_1)
            
            action_0 = action_0.item()
            action_1 = action_1.item()
            
            # Step environment
            joint_action = (Action.INDEX_TO_ACTION[action_0], Action.INDEX_TO_ACTION[action_1])
            next_state, info = self.base_env.mdp.get_state_transition(state, joint_action)
            
            # Get reward (sparse environment reward)
            env_reward = sum(info.get("sparse_reward_by_agent", [0, 0]))
            episode_reward += env_reward
            
            # Check if done
            done = step >= self.config.horizon - 1
            
            # Get next observation
            next_obs = self.base_env.featurize_state_mdp(next_state)
            next_obs_0 = torch.tensor(next_obs[0].flatten(), dtype=torch.float32, device=self.device)
            
            # Store transition
            states_list.append(obs_0.squeeze(0))
            actions_list.append(action_0)
            next_states_list.append(next_obs_0)
            dones_list.append(float(done))
            log_probs_list.append(log_prob_0.item())
            values_list.append(value_0.item())
            rewards_list.append(env_reward)
            
            # Update state
            state = next_state
            obs = next_obs
            
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                
                # Reset
                state = self.mdp.get_standard_start_state()
                obs = self.base_env.featurize_state_mdp(state)
                
                if self.config.use_lstm:
                    hidden = self.policy.get_initial_state(1, torch.device(self.device))
        
        # Record final partial episode
        if episode_reward > 0:
            episode_rewards.append(episode_reward)
        
        rollout = {
            "states": torch.stack(states_list),
            "actions": torch.tensor(actions_list, dtype=torch.long, device=self.device),
            "next_states": torch.stack(next_states_list),
            "dones": torch.tensor(dones_list, dtype=torch.float32, device=self.device),
            "log_probs": torch.tensor(log_probs_list, dtype=torch.float32, device=self.device),
            "values": torch.tensor(values_list, dtype=torch.float32, device=self.device),
            "env_rewards": torch.tensor(rewards_list, dtype=torch.float32, device=self.device),
            "episode_rewards": episode_rewards,
        }
        
        return rollout
    
    def _update_discriminator(
        self,
        expert_batch: Dict[str, torch.Tensor],
        policy_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Update discriminator to distinguish expert from policy samples.
        
        Returns:
            Discriminator loss
        """
        # Get log probs under current policy for both batches
        with torch.no_grad():
            # Expert actions
            expert_logits, _ = self.policy(expert_batch["states"])
            expert_log_probs = F.log_softmax(expert_logits, dim=-1)
            expert_action_log_probs = expert_log_probs[
                torch.arange(len(expert_batch["actions"])), expert_batch["actions"]
            ]
            
            # Policy actions
            policy_logits, _ = self.policy(policy_batch["states"])
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_action_log_probs = policy_log_probs[
                torch.arange(len(policy_batch["actions"])), policy_batch["actions"]
            ]
        
        # Discriminator forward pass
        expert_d = self.discriminator(
            expert_batch["states"],
            expert_batch["next_states"],
            expert_batch["dones"],
            expert_action_log_probs,
        )
        
        policy_d = self.discriminator(
            policy_batch["states"],
            policy_batch["next_states"],
            policy_batch["dones"],
            policy_action_log_probs,
        )
        
        # Binary cross-entropy loss with LABEL SMOOTHING
        # This prevents the discriminator from being overconfident
        smooth = self.config.label_smoothing
        expert_target = 1.0 - smooth  # 0.9 instead of 1.0
        policy_target = smooth  # 0.1 instead of 0.0
        
        # Smoothed BCE loss
        expert_loss = -(expert_target * torch.log(expert_d + 1e-8) + 
                        (1 - expert_target) * torch.log(1 - expert_d + 1e-8)).mean()
        policy_loss = -(policy_target * torch.log(policy_d + 1e-8) + 
                        (1 - policy_target) * torch.log(1 - policy_d + 1e-8)).mean()
        disc_loss = expert_loss + policy_loss
        
        # GRADIENT PENALTY for stability (WGAN-GP style)
        if self.config.grad_penalty_weight > 0:
            # Interpolate between expert and policy states
            alpha = torch.rand(len(expert_batch["states"]), 1, device=self.device)
            interp_states = alpha * expert_batch["states"] + (1 - alpha) * policy_batch["states"]
            interp_states.requires_grad_(True)
            
            # Get discriminator output for interpolated states
            interp_next = alpha * expert_batch["next_states"] + (1 - alpha) * policy_batch["next_states"]
            interp_dones = alpha.squeeze() * expert_batch["dones"] + (1 - alpha.squeeze()) * policy_batch["dones"]
            interp_log_probs = alpha.squeeze() * expert_action_log_probs + (1 - alpha.squeeze()) * policy_action_log_probs
            
            interp_d = self.discriminator(interp_states, interp_next, interp_dones, interp_log_probs)
            
            # Compute gradient penalty
            gradients = torch.autograd.grad(
                outputs=interp_d.sum(),
                inputs=interp_states,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_norm = gradients.norm(2, dim=1)
            grad_penalty = ((grad_norm - 1) ** 2).mean()
            disc_loss = disc_loss + self.config.grad_penalty_weight * grad_penalty
        
        # Gradient step
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.max_grad_norm)
        self.disc_optimizer.step()
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            accuracy = (
                (expert_d > 0.5).float().mean() + (policy_d < 0.5).float().mean()
            ) / 2
        
        return disc_loss.item(), accuracy.item()
    
    def _compute_airl_rewards(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute AIRL rewards for policy update."""
        with torch.no_grad():
            # Get action log probs
            logits, _ = self.policy(rollout["states"])
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs[
                torch.arange(len(rollout["actions"])), rollout["actions"]
            ]
            
            # AIRL reward
            rewards = self.discriminator.get_reward(
                rollout["states"],
                rollout["next_states"],
                rollout["dones"],
                action_log_probs,
            )
        
        return rewards
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
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
    
    def _compute_mean_kl(self, states: torch.Tensor) -> float:
        """Compute mean KL divergence between AIRL policy and BC anchor."""
        if not hasattr(self, 'bc_model') or self.bc_model is None:
            return 0.0
        
        with torch.no_grad():
            # Get BC policy distribution
            bc_logits = self.bc_model(states)
            bc_probs = F.softmax(bc_logits, dim=-1)
            
            # Get AIRL policy distribution
            airl_logits, _ = self.policy(states)
            airl_probs = F.softmax(airl_logits, dim=-1)
            airl_log_probs = F.log_softmax(airl_logits, dim=-1)
            
            # KL(π_AIRL || π_BC)
            kl_div = (airl_probs * (airl_log_probs - torch.log(bc_probs + 1e-8))).sum(dim=-1)
            return kl_div.mean().item()
    
    def _update_policy(self, rollout: Dict[str, torch.Tensor]) -> float:
        """
        Update policy with PPO using AIRL rewards.
        
        Returns:
            Policy loss
        """
        # Compute AIRL rewards
        airl_rewards = self._compute_airl_rewards(rollout)
        
        # Compute advantages
        with torch.no_grad():
            # Get final value estimate
            last_obs = rollout["states"][-1].unsqueeze(0)
            _, last_value = self.policy(last_obs)
            last_value = last_value.item()
        
        advantages, returns = self._compute_gae(
            airl_rewards,
            rollout["values"],
            rollout["dones"],
            last_value,
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Old log probs
        old_log_probs = rollout["log_probs"]
        
        # PPO update
        batch_size = len(rollout["states"])
        minibatch_size = self.config.batch_size
        total_loss = 0
        num_updates = 0
        
        for epoch in range(self.config.policy_epochs):
            perm = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                
                # Get current policy outputs
                log_probs, values, entropy = self.policy.evaluate_actions(
                    rollout["states"][idx],
                    rollout["actions"][idx],
                )
                
                # Ratio
                ratio = torch.exp(log_probs - old_log_probs[idx])
                
                # Clipped surrogate loss
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns[idx])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # KL REGULARIZATION: Keep policy close to BC anchor
                kl_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self, 'bc_model') and self.bc_model is not None:
                    with torch.no_grad():
                        bc_logits = self.bc_model(rollout["states"][idx])
                        bc_probs = F.softmax(bc_logits, dim=-1)
                    
                    # Get AIRL policy probabilities
                    airl_logits, _ = self.policy(rollout["states"][idx])
                    airl_probs = F.softmax(airl_logits, dim=-1)
                    airl_log_probs = F.log_softmax(airl_logits, dim=-1)
                    
                    # KL(π_AIRL || π_BC) = sum(π_AIRL * log(π_AIRL / π_BC))
                    kl_div = (airl_probs * (airl_log_probs - torch.log(bc_probs + 1e-8))).sum(dim=-1)
                    kl_loss = kl_div.mean()
                
                # Total loss (use current entropy and KL coefficients)
                current_ent_coef = getattr(self, 'current_ent_coef', self.config.ent_coef)
                current_kl_coef = getattr(self, 'current_kl_coef', self.config.kl_coef)
                
                loss = (
                    actor_loss
                    + self.config.vf_coef * value_loss
                    + current_ent_coef * entropy_loss
                    + current_kl_coef * kl_loss  # KL penalty to stay close to BC
                )
                
                # Gradient step
                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        return total_loss / max(num_updates, 1)
    
    def train(self) -> Dict[str, Any]:
        """
        Run AIRL training.
        
        Returns:
            Training results dictionary
        """
        total_timesteps = 0
        num_iters = self.config.total_timesteps // self.config.steps_per_iter
        
        # For AIRL, we track discriminator accuracy as the convergence metric
        # Goal: D_acc should stay around 0.5-0.7 (balanced adversarial game)
        # We DON'T use environment reward for early stopping since AIRL
        # optimizes for imitation, not task reward
        recent_disc_accs = []
        
        # Initialize entropy coefficient (will be ramped up during training)
        self.current_ent_coef = self.config.ent_coef
        ent_warmup_iters = getattr(self.config, 'ent_warmup_iters', 50)
        ent_coef_final = getattr(self.config, 'ent_coef_final', self.config.ent_coef)
        
        # Initialize KL coefficient (adaptive)
        self.current_kl_coef = self.config.kl_coef
        kl_target = getattr(self.config, 'kl_target', 0.01)
        use_adaptive_kl = getattr(self.config, 'use_adaptive_kl', True)
        
        # Track KL divergence for logging and adaptive adjustment
        self.recent_kl = []
        
        if self.config.verbose:
            print(f"\nStarting AIRL training with KL regularization")
            print(f"Layout: {self.config.layout_name}")
            print(f"Total timesteps: {self.config.total_timesteps:,}")
            print(f"Steps per iteration: {self.config.steps_per_iter:,}")
            print(f"Device: {self.device}")
            print(f"Entropy coef: {self.config.ent_coef} -> {ent_coef_final} over {ent_warmup_iters} iters")
            print(f"KL coef: {self.config.kl_coef} (target KL: {kl_target}, adaptive: {use_adaptive_kl})")
            print(f"BC anchor: {'Yes' if hasattr(self, 'bc_model') and self.bc_model is not None else 'No'}")
            print()
        
        start_time = time.time()
        
        for iteration in range(num_iters):
            # Entropy coefficient warmup: start low (like BC), ramp up for exploration
            if iteration < ent_warmup_iters:
                self.current_ent_coef = self.config.ent_coef + (ent_coef_final - self.config.ent_coef) * (iteration / ent_warmup_iters)
            else:
                self.current_ent_coef = ent_coef_final
            # Collect rollout
            rollout = self._collect_rollout(self.config.steps_per_iter)
            total_timesteps += self.config.steps_per_iter
            
            # Add to sample buffer
            self.policy_sample_buffer.append(rollout)
            
            # Combine samples from buffer for discriminator training
            combined_states = torch.cat([r["states"] for r in self.policy_sample_buffer])
            combined_actions = torch.cat([r["actions"] for r in self.policy_sample_buffer])
            combined_next_states = torch.cat([r["next_states"] for r in self.policy_sample_buffer])
            combined_dones = torch.cat([r["dones"] for r in self.policy_sample_buffer])
            
            policy_batch = {
                "states": combined_states,
                "actions": combined_actions,
                "next_states": combined_next_states,
                "dones": combined_dones,
            }
            
            # Discriminator updates
            disc_losses = []
            disc_accs = []
            
            for _ in range(self.config.disc_updates_per_iter):
                # Sample expert batch
                expert_idx = torch.randint(0, len(self.expert_states), (self.config.batch_size,))
                expert_batch = {
                    "states": self.expert_states[expert_idx],
                    "actions": self.expert_actions[expert_idx],
                    "next_states": self.expert_next_states[expert_idx],
                    "dones": self.expert_dones[expert_idx],
                }
                
                # Sample policy batch
                policy_idx = torch.randint(0, len(combined_states), (self.config.batch_size,))
                policy_mini_batch = {k: v[policy_idx] for k, v in policy_batch.items()}
                
                disc_loss, disc_acc = self._update_discriminator(expert_batch, policy_mini_batch)
                disc_losses.append(disc_loss)
                disc_accs.append(disc_acc)
            
            # Policy update
            policy_loss = self._update_policy(rollout)
            
            # Compute KL divergence from BC anchor
            mean_kl = 0.0
            if hasattr(self, 'bc_model') and self.bc_model is not None:
                mean_kl = self._compute_mean_kl(rollout["states"])
                self.recent_kl.append(mean_kl)
                if len(self.recent_kl) > 50:
                    self.recent_kl.pop(0)
                
                # Adaptive KL coefficient adjustment
                if use_adaptive_kl and len(self.recent_kl) >= 10:
                    avg_kl = np.mean(self.recent_kl[-10:])
                    if avg_kl > kl_target * 1.5:
                        # KL too high, increase penalty
                        self.current_kl_coef = min(self.current_kl_coef * 1.5, 10.0)
                    elif avg_kl < kl_target * 0.5:
                        # KL too low, decrease penalty
                        self.current_kl_coef = max(self.current_kl_coef / 1.5, 0.01)
            
            # Compute mean AIRL reward for this rollout
            airl_rewards = self._compute_airl_rewards(rollout)
            mean_airl_reward = airl_rewards.mean().item()
            
            # Track episode rewards (environment reward, for reference only)
            episode_rewards = rollout["episode_rewards"]
            if episode_rewards:
                mean_env_reward = np.mean(episode_rewards)
                self.train_info["episode_rewards"].extend(episode_rewards)
            else:
                mean_env_reward = 0
            
            # Track discriminator accuracy for convergence monitoring
            mean_disc_acc = np.mean(disc_accs)
            recent_disc_accs.append(mean_disc_acc)
            if len(recent_disc_accs) > 50:
                recent_disc_accs.pop(0)
            
            # Logging
            self.train_info["disc_losses"].append(np.mean(disc_losses))
            self.train_info["policy_losses"].append(policy_loss)
            self.train_info["disc_accuracy"].append(mean_disc_acc)
            
            if iteration % self.config.log_interval == 0 and self.config.verbose:
                elapsed = time.time() - start_time
                fps = total_timesteps / elapsed if elapsed > 0 else 0
                
                # Show KL divergence and all metrics
                print(f"Iter {iteration}/{num_iters} | "
                      f"FPS: {fps:.0f} | "
                      f"KL: {mean_kl:.3f} (c={self.current_kl_coef:.2f}) | "
                      f"D_acc: {mean_disc_acc:.2f} | "
                      f"Env_R: {mean_env_reward:.0f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(iteration)
        
        # Final save
        self.save_checkpoint(num_iters)
        
        if self.config.verbose:
            total_time = time.time() - start_time
            avg_disc_acc = np.mean(recent_disc_accs) if recent_disc_accs else 0.5
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Final D_acc: {avg_disc_acc:.2f} (target: 0.5-0.7)")
            print(f"Total iterations: {iteration + 1}")
        
        return {
            "total_timesteps": total_timesteps,
            "train_info": self.train_info,
            "final_disc_acc": np.mean(recent_disc_accs) if recent_disc_accs else 0.5,
        }
    
    def save_checkpoint(self, step: int):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.results_dir,
            self.config.experiment_name,
            f"checkpoint_{step:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save policy
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }, os.path.join(checkpoint_dir, "model.pt"))
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        if self.config.verbose:
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            map_location=self.device,
        )
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
        if self.config.verbose:
            print(f"Loaded checkpoint from {checkpoint_dir}")
    
    def get_policy(self) -> nn.Module:
        """Return the trained policy."""
        return self.policy


##############
# Helper Functions #
##############


def save_airl_model(
    save_dir: str,
    policy: nn.Module,
    discriminator: nn.Module,
    config: AIRLConfig,
    verbose: bool = False,
):
    """Save AIRL model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
    }, os.path.join(save_dir, "model.pt"))
    
    # Save config
    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    if verbose:
        print(f"Saved AIRL model to {save_dir}")


def load_airl_model(
    model_dir: str,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, nn.Module, AIRLConfig]:
    """
    Load AIRL model from disk.
    
    Returns:
        Tuple of (policy, discriminator, config)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    # Get state dim from config
    mdp_params = {"layout_name": config.layout_name, "old_dynamics": config.old_dynamics}
    agent_evaluator = AgentEvaluator.from_layout_name(
        mdp_params=mdp_params,
        env_params=DEFAULT_ENV_PARAMS,
    )
    base_env = agent_evaluator.env
    dummy_state = base_env.mdp.get_standard_start_state()
    obs_shape = base_env.featurize_state_mdp(dummy_state)[0].shape
    state_dim = int(np.prod(obs_shape))
    action_dim = len(Action.ALL_ACTIONS)
    
    # Create networks
    if config.use_lstm:
        policy = AIRLPolicyLSTM(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.policy_hidden_dim,
            num_layers=config.policy_num_layers,
            cell_size=config.cell_size,
        )
    else:
        policy = AIRLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.policy_hidden_dim,
            num_layers=config.policy_num_layers,
        )
    
    discriminator = AIRLDiscriminator(
        state_dim=state_dim,
        gamma=config.gamma,
        hidden_dim=config.disc_hidden_dim,
        num_layers=config.disc_num_layers,
        g_linear=config.disc_g_linear,
    )
    
    # Load weights
    checkpoint = torch.load(
        os.path.join(model_dir, "model.pt"),
        map_location=device,
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    
    policy = policy.to(device)
    discriminator = discriminator.to(device)
    policy.eval()
    discriminator.eval()
    
    if verbose:
        print(f"Loaded AIRL model from {model_dir}")
    
    return policy, discriminator, config


if __name__ == "__main__":
    # Simple test
    config = AIRLConfig(
        layout_name="cramped_room",
        total_timesteps=50000,
        steps_per_iter=1000,
        verbose=True,
    )
    
    trainer = AIRLTrainer(config)
    results = trainer.train()
    print(f"Training complete: {results}")

