"""
JAX Policy Agent for Overcooked.

This module provides an Agent class that wraps JAX-based policies
for use in the Overcooked environment.
"""

import os
import pickle
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the last axis."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class JaxPolicyAgent(Agent):
    """
    An agent that uses a JAX-trained policy to select actions.
    
    This agent wraps trained JAX neural network parameters and provides
    the standard Agent interface for use in Overcooked environments.
    """

    def __init__(
        self,
        network: Any,
        params: Dict[str, Any],
        featurize_fn: Callable,
        agent_index: int = 0,
        stochastic: bool = True,
        use_lossless_encoding: bool = True
    ):
        """
        Initialize a JAX policy agent.
        
        Args:
            network: JAX/Flax neural network module
            params: Trained network parameters
            featurize_fn: Function to convert state to observation
            agent_index: Index of this agent (0 or 1)
            stochastic: Whether to sample or use argmax
            use_lossless_encoding: Whether observation uses lossless encoding
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for JaxPolicyAgent. "
                "Install with: pip install jax jaxlib flax"
            )
        
        super(JaxPolicyAgent, self).__init__()
        
        self.network = network
        self.params = params
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        self.use_lossless_encoding = use_lossless_encoding
        self.agent_index = agent_index
        
        # LSTM state (if applicable)
        self.lstm_state = None
        self._has_lstm = hasattr(network, 'initialize_carry')
        
        # Initialize random key
        self._key = random.PRNGKey(0)

    def reset(self):
        """Reset the agent, including LSTM state."""
        super().reset()
        if self._has_lstm:
            self.lstm_state = self.network.initialize_carry(1)
        else:
            self.lstm_state = None
        self._key = random.PRNGKey(0)

    def action(self, state) -> Tuple[Any, Dict]:
        """
        Compute an action for the given state.
        
        Args:
            state: OvercookedState object
            
        Returns:
            Tuple of (action, action_info_dict)
        """
        # Featurize the state
        if self.use_lossless_encoding:
            obs = self.featurize_fn(state)
        else:
            obs = self.featurize_fn(state)
        
        my_obs = obs[self.agent_index]
        
        # Convert to JAX array
        obs_jax = jnp.array(my_obs, dtype=jnp.float32)
        obs_jax = jnp.expand_dims(obs_jax, axis=0)  # Add batch dimension
        
        # Get action logits from network
        if self._has_lstm:
            if self.lstm_state is None:
                self.lstm_state = self.network.initialize_carry(1)
            logits, _, self.lstm_state = self.network.apply(
                self.params, obs_jax, self.lstm_state
            )
        else:
            logits, _ = self.network.apply(self.params, obs_jax)
        
        # Convert to numpy
        logits_np = np.array(logits[0])
        
        # Compute action probabilities
        action_probs = softmax(logits_np)
        
        # Select action
        if self.stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)
        
        action = Action.INDEX_TO_ACTION[action_idx]
        
        return action, {"action_probs": action_probs}

    def set_agent_index(self, agent_index: int):
        """Set the agent index."""
        self.agent_index = agent_index

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        featurize_fn: Callable,
        agent_index: int = 0,
        stochastic: bool = True,
        use_lossless_encoding: bool = True
    ) -> "JaxPolicyAgent":
        """
        Create a JaxPolicyAgent from a saved checkpoint.
        
        Args:
            checkpoint_dir: Directory containing saved checkpoint
            featurize_fn: Function to convert state to observation
            agent_index: Index of this agent (0 or 1)
            stochastic: Whether to sample or use argmax
            use_lossless_encoding: Whether observation uses lossless encoding
            
        Returns:
            JaxPolicyAgent instance
        """
        from human_aware_rl.jaxmarl.ppo import ActorCritic, ActorCriticLSTM, PPOConfig
        
        # Load config
        config_path = os.path.join(checkpoint_dir, "config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        # Load params
        params_path = os.path.join(checkpoint_dir, "params.pkl")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        
        # Recreate network
        from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig
        
        env_config = OvercookedJaxEnvConfig(
            layout_name=config.layout_name,
            use_lossless_encoding=use_lossless_encoding
        )
        dummy_env = OvercookedJaxEnv(env_config)
        num_actions = dummy_env.num_actions
        
        if config.use_lstm:
            network = ActorCriticLSTM(
                action_dim=num_actions,
                hidden_dim=config.hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                cell_size=config.cell_size,
            )
        else:
            network = ActorCritic(
                action_dim=num_actions,
                hidden_dim=config.hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                num_filters=config.num_filters,
                num_conv_layers=config.num_conv_layers,
            )
        
        return cls(
            network=network,
            params=params,
            featurize_fn=featurize_fn,
            agent_index=agent_index,
            stochastic=stochastic,
            use_lossless_encoding=use_lossless_encoding
        )


def load_jax_agent(
    checkpoint_dir: str,
    layout_name: str,
    agent_index: int = 0,
    stochastic: bool = True,
    use_lossless_encoding: bool = True
) -> JaxPolicyAgent:
    """
    Load a JAX-trained agent from a checkpoint.
    
    Args:
        checkpoint_dir: Directory containing saved checkpoint
        layout_name: Layout name (for creating featurize function)
        agent_index: Index of this agent (0 or 1)
        stochastic: Whether to sample or use argmax
        use_lossless_encoding: Whether to use lossless state encoding
        
    Returns:
        JaxPolicyAgent instance
    """
    from overcooked_ai_py.agents.benchmarking import AgentEvaluator
    
    # Create evaluator to get featurize function
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={"horizon": 400}
    )
    
    if use_lossless_encoding:
        featurize_fn = lambda state: ae.env.lossless_state_encoding_mdp(state)
    else:
        featurize_fn = lambda state: ae.env.featurize_state_mdp(state)
    
    return JaxPolicyAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        featurize_fn=featurize_fn,
        agent_index=agent_index,
        stochastic=stochastic,
        use_lossless_encoding=use_lossless_encoding
    )

