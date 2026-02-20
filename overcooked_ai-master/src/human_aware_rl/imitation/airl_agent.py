"""
AIRL Agent for Overcooked AI.

This module provides an Agent class that wraps AIRL-trained policies
for use in the Overcooked environment.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the last axis."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class AIRLAgent(Agent):
    """
    An agent that uses an AIRL-trained policy to select actions.
    
    This agent wraps a trained AIRL policy and provides the standard Agent 
    interface for use in Overcooked environments.
    """

    def __init__(
        self,
        policy: nn.Module,
        config: Any,
        featurize_fn: Callable,
        agent_index: int = 0,
        stochastic: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize an AIRL agent.
        
        Args:
            policy: Trained AIRL policy (AIRLPolicy or AIRLPolicyLSTM)
            config: AIRLConfig used to train the model
            featurize_fn: Function that converts OvercookedState to feature vector
            agent_index: Index of this agent (0 or 1)
            stochastic: If True, sample actions from predicted distribution; if False, use argmax
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        # Initialize these before calling super().__init__() because it calls reset()
        self.use_lstm = getattr(config, "use_lstm", False)
        self.cell_size = getattr(config, "cell_size", 256)
        self._agent_index = agent_index  # Store temporarily
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Now call parent init (which calls reset() - this sets agent_index to None)
        super(AIRLAgent, self).__init__()
        
        self.policy = policy
        self.config = config
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        
        # Set agent_index AFTER parent init (parent's reset() sets it to None)
        self.agent_index = self._agent_index

    def _init_lstm_state(self):
        """Initialize or reset LSTM hidden state."""
        if self.use_lstm:
            self.hidden_state = (
                torch.zeros(1, 1, self.cell_size, device=self.device),
                torch.zeros(1, 1, self.cell_size, device=self.device),
            )
        else:
            self.hidden_state = None

    def reset(self):
        """Reset the agent, including LSTM hidden state."""
        # Preserve agent_index as parent's reset() sets it to None
        current_agent_index = getattr(self, 'agent_index', None) or getattr(self, '_agent_index', 0)
        super().reset()
        self.agent_index = current_agent_index
        self._init_lstm_state()

    def action(self, state) -> Tuple[Any, Dict]:
        """
        Compute an action for the given state.
        
        Args:
            state: OvercookedState object
            
        Returns:
            Tuple of (action, action_info_dict)
        """
        # Featurize the state
        obs = self.featurize_fn(state)
        my_obs = obs[self.agent_index]
        
        # Flatten and convert to tensor
        obs_flat = my_obs.flatten()
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            if self.use_lstm:
                logits, _, self.hidden_state = self.policy(obs_tensor, self.hidden_state)
            else:
                logits, _ = self.policy(obs_tensor)
            
            logits = logits.squeeze(0)
        
        # Convert to numpy
        logits_np = logits.cpu().numpy()
        
        # Compute action probabilities
        action_probs = softmax(logits_np)
        
        # Select action
        if self.stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)
        
        action = Action.INDEX_TO_ACTION[action_idx]
        
        return action, {"action_probs": action_probs}

    def actions(self, states, agent_indices) -> List[Tuple[Any, Dict]]:
        """
        Compute actions for multiple states.
        
        Args:
            states: List of OvercookedState objects
            agent_indices: List of agent indices for each state
            
        Returns:
            List of (action, action_info_dict) tuples
        """
        results = []
        for state, agent_idx in zip(states, agent_indices):
            self.agent_index = agent_idx
            results.append(self.action(state))
        return results

    def set_agent_index(self, agent_index: int):
        """Set the agent index."""
        self.agent_index = agent_index

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        featurize_fn: Callable,
        agent_index: int = 0,
        stochastic: bool = True,
        device: Optional[str] = None,
    ) -> "AIRLAgent":
        """
        Create an AIRLAgent from a saved model directory.
        
        Args:
            model_dir: Directory containing saved model
            featurize_fn: Function that converts OvercookedState to feature vector
            agent_index: Index of this agent (0 or 1)
            stochastic: If True, sample actions from predicted distribution
            device: Device to run inference on
            
        Returns:
            AIRLAgent instance
        """
        from human_aware_rl.imitation.airl import load_airl_model
        
        policy, _, config = load_airl_model(model_dir, device=device)
        return cls(policy, config, featurize_fn, agent_index, stochastic, device)


class AIRLPolicy:
    """
    Policy wrapper for AIRL models that provides a consistent interface
    for evaluation and integration with other components.
    
    This class wraps an AIRL policy and provides methods compatible
    with the existing evaluation infrastructure.
    """

    def __init__(
        self,
        policy: nn.Module,
        config: Any,
        stochastic: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize an AIRL policy wrapper.
        
        Args:
            policy: Trained AIRL policy
            config: AIRLConfig used to train the model
            stochastic: If True, sample actions from predicted distribution
            device: Device to run inference on
        """
        self.policy = policy
        self.config = config
        self.stochastic = stochastic
        self.use_lstm = getattr(config, "use_lstm", False)
        self.cell_size = getattr(config, "cell_size", 256)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.policy = self.policy.to(self.device)
        self.policy.eval()

    def compute_actions(
        self,
        obs_batch: np.ndarray,
        state_batches: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """
        Compute actions for a batch of observations.
        
        Args:
            obs_batch: Batch of observations, shape (batch_size, *obs_shape)
            state_batches: Optional LSTM states [h, c] each of shape (batch_size, cell_size)
            
        Returns:
            Tuple of (actions, state_outs, info_dict)
        """
        obs_batch = np.array(obs_batch, dtype=np.float32)
        batch_size = len(obs_batch)
        
        # Flatten observations if needed
        if obs_batch.ndim > 2:
            obs_batch = obs_batch.reshape(batch_size, -1)
        
        # Convert to tensor
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            if self.use_lstm:
                # Prepare hidden state
                if state_batches is not None and len(state_batches) == 2:
                    h = torch.tensor(state_batches[0], dtype=torch.float32, device=self.device)
                    c = torch.tensor(state_batches[1], dtype=torch.float32, device=self.device)
                    # Add layer dimension if needed
                    if h.dim() == 2:
                        h = h.unsqueeze(0)
                        c = c.unsqueeze(0)
                    hidden = (h, c)
                else:
                    hidden = (
                        torch.zeros(1, batch_size, self.cell_size, device=self.device),
                        torch.zeros(1, batch_size, self.cell_size, device=self.device),
                    )
                
                logits, _, (h_out, c_out) = self.policy(obs_tensor, hidden)
                
                # Convert states back to numpy
                state_outs = [
                    h_out.squeeze(0).cpu().numpy(),
                    c_out.squeeze(0).cpu().numpy(),
                ]
            else:
                logits, _ = self.policy(obs_tensor)
                state_outs = []
        
        # Convert logits to numpy
        logits_np = logits.cpu().numpy()
        
        # Compute action probabilities
        action_probs = softmax(logits_np)
        
        # Select actions
        if self.stochastic:
            actions = np.array([
                np.random.choice(action_probs.shape[1], p=action_probs[i])
                for i in range(batch_size)
            ])
        else:
            actions = np.argmax(logits_np, axis=1)
        
        return actions, state_outs, {"action_dist_inputs": logits_np}

    def get_initial_state(self) -> List[np.ndarray]:
        """
        Get initial LSTM state.
        
        Returns:
            List of [h_0, c_0] each of shape (cell_size,) or empty list if not LSTM
        """
        if self.use_lstm:
            return [
                np.zeros(self.cell_size, dtype=np.float32),
                np.zeros(self.cell_size, dtype=np.float32),
            ]
        return []

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        stochastic: bool = True,
        device: Optional[str] = None,
    ) -> "AIRLPolicy":
        """
        Create a policy from a saved model directory.
        
        Args:
            model_dir: Directory containing saved model
            stochastic: If True, sample actions from predicted distribution
            device: Device to run inference on
            
        Returns:
            AIRLPolicy instance
        """
        from human_aware_rl.imitation.airl import load_airl_model
        
        policy, _, config = load_airl_model(model_dir, device=device)
        return cls(policy, config, stochastic, device)

