"""
Behavior Cloning Agent for Overcooked AI

This module provides an Agent class that wraps PyTorch BC models
for use in the Overcooked environment.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the last axis."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class BCAgent(Agent):
    """
    An agent that uses a PyTorch Behavior Cloning model to select actions.
    
    This agent wraps a trained BC model and provides the standard Agent interface
    for use in Overcooked environments.
    """

    def __init__(
        self,
        model: nn.Module,
        bc_params: Dict,
        featurize_fn: Callable,
        agent_index: int = 0,
        stochastic: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize a BC agent.
        
        Args:
            model: Trained PyTorch BC model (BCModel or BCLSTMModel)
            bc_params: Dictionary of BC parameters used to train the model
            featurize_fn: Function that converts OvercookedState to feature vector
            agent_index: Index of this agent (0 or 1)
            stochastic: If True, sample actions from predicted distribution; if False, use argmax
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        # Initialize these before calling super().__init__() because it calls reset()
        self.use_lstm = bc_params.get("use_lstm", False)
        self.cell_size = bc_params.get("cell_size", 256)
        self._agent_index = agent_index  # Store temporarily
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Now call parent init (which calls reset() - this sets agent_index to None)
        super(BCAgent, self).__init__()
        
        self.model = model
        self.bc_params = bc_params
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set agent_index AFTER parent init (parent's reset() sets it to None)
        self.agent_index = self._agent_index

    def _init_lstm_state(self):
        """Initialize or reset LSTM hidden state."""
        if self.use_lstm:
            self.hidden_state = (
                torch.zeros(1, 1, self.cell_size, device=self.device),
                torch.zeros(1, 1, self.cell_size, device=self.device)
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
        
        # Convert to tensor
        obs_tensor = torch.tensor(my_obs, dtype=torch.float32, device=self.device)
        
        # Get action logits from model
        with torch.no_grad():
            if self.use_lstm:
                # Add batch and sequence dimensions
                obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
                logits, self.hidden_state = self.model(obs_tensor, self.hidden_state)
                logits = logits.squeeze(0).squeeze(0)  # (num_actions,)
            else:
                # Add batch dimension
                obs_tensor = obs_tensor.unsqueeze(0)  # (1, obs_dim)
                logits = self.model(obs_tensor)
                logits = logits.squeeze(0)  # (num_actions,)
        
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
        device: Optional[str] = None
    ) -> "BCAgent":
        """
        Create a BCAgent from a saved model directory.
        
        Args:
            model_dir: Directory containing saved model
            featurize_fn: Function that converts OvercookedState to feature vector
            agent_index: Index of this agent (0 or 1)
            stochastic: If True, sample actions from predicted distribution
            device: Device to run inference on
            
        Returns:
            BCAgent instance
        """
        from human_aware_rl.imitation.behavior_cloning import load_bc_model
        
        model, bc_params = load_bc_model(model_dir, device=device)
        return cls(model, bc_params, featurize_fn, agent_index, stochastic, device)


class BehaviorCloningPolicy:
    """
    Policy wrapper for BC models that provides a consistent interface
    for evaluation and integration with other components.
    
    This class wraps a PyTorch BC model and provides methods compatible
    with the existing evaluation infrastructure.
    """

    def __init__(
        self,
        model: nn.Module,
        bc_params: Dict,
        stochastic: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize a BC policy.
        
        Args:
            model: Trained PyTorch BC model
            bc_params: Dictionary of BC parameters
            stochastic: If True, sample actions from predicted distribution
            device: Device to run inference on
        """
        self.model = model
        self.bc_params = bc_params
        self.stochastic = stochastic
        self.use_lstm = bc_params.get("use_lstm", False)
        self.cell_size = bc_params.get("cell_size", 256)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store shapes
        self.observation_shape = bc_params["observation_shape"]
        self.action_shape = bc_params["action_shape"]

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
        
        # Convert to tensor
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            if self.use_lstm:
                # Add sequence dimension
                obs_tensor = obs_tensor.unsqueeze(1)  # (batch, 1, obs_dim)
                
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
                        torch.zeros(1, batch_size, self.cell_size, device=self.device)
                    )
                
                logits, (h_out, c_out) = self.model(obs_tensor, hidden)
                logits = logits.squeeze(1)  # (batch, num_actions)
                
                # Convert states back to numpy
                state_outs = [
                    h_out.squeeze(0).cpu().numpy(),
                    c_out.squeeze(0).cpu().numpy()
                ]
            else:
                logits = self.model(obs_tensor)
                state_outs = []
        
        # Convert logits to numpy
        logits_np = logits.cpu().numpy()
        
        # Compute action probabilities
        action_probs = softmax(logits_np)
        
        # Select actions
        if self.stochastic:
            actions = np.array([
                np.random.choice(self.action_shape[0], p=action_probs[i])
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
                np.zeros(self.cell_size, dtype=np.float32)
            ]
        return []

    @classmethod
    def from_model_dir(cls, model_dir: str, stochastic: bool = True, device: Optional[str] = None) -> "BehaviorCloningPolicy":
        """
        Create a policy from a saved model directory.
        
        Args:
            model_dir: Directory containing saved model
            stochastic: If True, sample actions from predicted distribution
            device: Device to run inference on
            
        Returns:
            BehaviorCloningPolicy instance
        """
        from human_aware_rl.imitation.behavior_cloning import load_bc_model
        
        model, bc_params = load_bc_model(model_dir, device=device)
        return cls(model, bc_params, stochastic, device)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        bc_params: Dict,
        stochastic: bool = True,
        device: Optional[str] = None
    ) -> "BehaviorCloningPolicy":
        """
        Create a policy from a model instance.
        
        Args:
            model: PyTorch BC model
            bc_params: BC parameters
            stochastic: If True, sample actions from predicted distribution
            device: Device to run inference on
            
        Returns:
            BehaviorCloningPolicy instance
        """
        return cls(model, bc_params, stochastic, device)

