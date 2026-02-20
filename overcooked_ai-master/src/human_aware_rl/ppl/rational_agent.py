"""
Rational Agent Model using Probabilistic Programming

This module implements a "softmax-rational" agent model inspired by
cognitive science and computational rationality. The key idea:

    P(action | state) ∝ exp(β * Q(state, action))

Where:
- Q(state, action) is the expected value of taking action in state
- β (inverse temperature) controls rationality:
  - β → 0: random actions
  - β → ∞: optimal actions
  - β ≈ 1-10: bounded rationality (human-like)

This is related to:
- Softmax policies in RL
- Boltzmann exploration
- Luce choice axiom
- Rational Speech Acts (RSA) models in pragmatics

Key References:
- Bounded Rationality (Herbert Simon)
- Softmax Q-learning (Bridle, 1990)
- Inverse Reinforcement Learning (Ng & Russell, 2000)
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN


PPL_SAVE_DIR = os.path.join(DATA_DIR, "ppl_runs")


@dataclass
class RationalAgentConfig:
    """Configuration for Rational Agent model."""
    
    # Environment
    layout_name: str = "cramped_room"
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    
    # Model
    # Q-function architecture
    q_hidden_dims: Tuple[int, ...] = (64, 64)
    
    # Rationality parameter (inverse temperature)
    # If None, will be inferred from data
    beta_init: float = 1.0
    learn_beta: bool = True
    beta_prior_mean: float = 1.0
    beta_prior_scale: float = 2.0
    
    # Noise model for action selection
    # "softmax": standard softmax rationality
    # "epsilon_greedy": epsilon-greedy with learned epsilon
    # "sticky": adds stickiness to previous actions
    noise_model: str = "softmax"
    
    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    
    # Output
    results_dir: str = PPL_SAVE_DIR
    seed: int = 0
    verbose: bool = True


class QNetwork(nn.Module):
    """Neural network approximating Q-values."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        
        dims = [state_dim] + list(hidden_dims) + [action_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RationalAgentModel(PyroModule):
    """
    Softmax-Rational Agent Model.
    
    Models human behavior as approximately rational decision-making:
    
        P(action | state) = softmax(β * Q(state, ·))
    
    Where Q is a learned value function and β is the rationality parameter.
    
    This model has an interesting interpretation:
    - The Q-network learns what the human "values" in the task
    - β captures how consistently they pursue those values
    - Low β indicates noisy/exploratory behavior
    - High β indicates near-optimal pursuit of goals
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        q_hidden_dims: Tuple[int, ...] = (64, 64),
        beta_init: float = 1.0,
        learn_beta: bool = True,
        beta_prior_mean: float = 1.0,
        beta_prior_scale: float = 2.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learn_beta = learn_beta
        
        # Q-network (deterministic, learned via backprop)
        self.q_network = QNetwork(state_dim, action_dim, q_hidden_dims)
        
        # Rationality parameter
        if learn_beta:
            # Place a prior on beta (should be positive)
            # Use LogNormal to ensure positivity
            self.beta = PyroSample(
                dist.LogNormal(
                    torch.tensor(np.log(beta_prior_mean)),
                    torch.tensor(beta_prior_scale),
                )
            )
        else:
            self.register_buffer("beta", torch.tensor(beta_init))
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            states: (batch, state_dim)
            actions: (batch,) target actions for training
            
        Returns:
            Q-values, (batch, action_dim)
        """
        # Compute Q-values
        q_values = self.q_network(states)
        
        # Scale by rationality parameter
        if self.learn_beta:
            beta = self.beta
        else:
            beta = self.beta
        
        logits = beta * q_values
        
        # Likelihood
        if actions is not None:
            with pyro.plate("data", states.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=actions)
        
        return q_values
    
    def get_action_probs(self, states: torch.Tensor, beta: float = None) -> torch.Tensor:
        """Get action probabilities."""
        q_values = self.q_network(states)
        
        if beta is None:
            if self.learn_beta and hasattr(self, '_current_beta'):
                beta = self._current_beta
            else:
                beta = 1.0
        
        logits = beta * q_values
        return F.softmax(logits, dim=-1)


class RationalAgentTrainer:
    """Trainer for Rational Agent models."""
    
    def __init__(self, config: RationalAgentConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)
        
        self._setup_environment()
        self._load_data()
        self._setup_model()
        
        if config.verbose:
            print(f"Rational Agent Trainer initialized")
            print(f"  Device: {self.device}")
            print(f"  Learn beta: {config.learn_beta}")
    
    def _setup_environment(self):
        """Setup environment."""
        mdp_params = {"layout_name": self.config.layout_name, "old_dynamics": True}
        
        self.agent_evaluator = AgentEvaluator.from_layout_name(
            mdp_params=mdp_params,
            env_params=DEFAULT_ENV_PARAMS,
        )
        self.base_env = self.agent_evaluator.env
        self.mdp = self.base_env.mdp
        
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.base_env.featurize_state_mdp(dummy_state)[0].shape
        self.state_dim = int(np.prod(obs_shape))
        self.action_dim = len(Action.ALL_ACTIONS)
    
    def _load_data(self):
        """Load data."""
        data_params = {
            "layouts": [self.config.layout_name],
            "check_trajectories": False,
            "featurize_states": True,
            "data_path": self.config.data_path,
        }
        
        processed_trajs = get_human_human_trajectories(**data_params, silent=True)
        
        states, actions = [], []
        for ep_idx in range(len(processed_trajs["ep_states"])):
            for t in range(len(processed_trajs["ep_states"][ep_idx])):
                states.append(processed_trajs["ep_states"][ep_idx][t].flatten())
                actions.append(int(processed_trajs["ep_actions"][ep_idx][t]))
        
        self.train_states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        self.train_actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        
        if self.config.verbose:
            print(f"Loaded {len(self.train_states)} transitions")
    
    def _setup_model(self):
        """Setup model."""
        pyro.clear_param_store()
        
        self.model = RationalAgentModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            q_hidden_dims=self.config.q_hidden_dims,
            beta_init=self.config.beta_init,
            learn_beta=self.config.learn_beta,
            beta_prior_mean=self.config.beta_prior_mean,
            beta_prior_scale=self.config.beta_prior_scale,
        ).to(self.device)
        
        # Use MAP estimation for Q-network, variational for beta
        self.guide = AutoDiagonalNormal(self.model)
        
        self.optimizer = ClippedAdam({"lr": self.config.learning_rate})
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
    
    def train(self) -> Dict[str, Any]:
        """Train model."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Training Rational Agent Model")
            print(f"{'='*60}")
        
        num_samples = len(self.train_states)
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        losses = []
        betas = []
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            perm = torch.randperm(num_samples)
            
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                
                loss = self.svi.step(self.train_states[idx], self.train_actions[idx])
                epoch_loss += loss
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Get current beta estimate
            if self.config.learn_beta:
                # Sample beta from guide
                self.guide()
                beta = self.model.beta.item() if hasattr(self.model.beta, 'item') else float(self.model.beta)
                betas.append(beta)
            
            if self.config.verbose and (epoch + 1) % 10 == 0:
                acc = self._compute_accuracy()
                beta_str = f" | β: {betas[-1]:.3f}" if betas else ""
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}{beta_str}")
        
        self._save()
        
        return {"losses": losses, "betas": betas}
    
    def _compute_accuracy(self) -> float:
        """Compute accuracy."""
        with torch.no_grad():
            self.guide()
            q_values = self.model(self.train_states)
            preds = torch.argmax(q_values, dim=-1)
            return (preds == self.train_actions).float().mean().item()
    
    def _save(self):
        """Save model."""
        save_dir = os.path.join(self.config.results_dir, "rational_agent", self.config.layout_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Q-network weights
        torch.save(self.model.q_network.state_dict(), os.path.join(save_dir, "q_network.pt"))
        
        # Save Pyro params (for beta)
        pyro.get_param_store().save(os.path.join(save_dir, "params.pt"))
        
        # Save config
        config_dict = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "q_hidden_dims": self.config.q_hidden_dims,
            "learn_beta": self.config.learn_beta,
            "layout_name": self.config.layout_name,
        }
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(config_dict, f)
        
        if self.config.verbose:
            print(f"Saved to {save_dir}")


class RationalAgent(Agent):
    """Agent using trained Rational Agent model."""
    
    def __init__(
        self,
        model: RationalAgentModel,
        featurize_fn,
        agent_index: int = 0,
        beta: float = 1.0,
        stochastic: bool = True,
        device: str = None,
    ):
        super().__init__()
        
        self.model = model
        self.featurize_fn = featurize_fn
        self.agent_index = agent_index
        self.beta = beta
        self.stochastic = stochastic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def action(self, state) -> Tuple[Any, Dict]:
        """Select action."""
        obs = self.featurize_fn(state)
        my_obs = obs[self.agent_index]
        
        obs_tensor = torch.tensor(
            my_obs.flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model.q_network(obs_tensor)
            probs = F.softmax(self.beta * q_values, dim=-1).squeeze().cpu().numpy()
        
        if self.stochastic:
            action_idx = np.random.choice(len(probs), p=probs)
        else:
            action_idx = np.argmax(probs)
        
        action = Action.INDEX_TO_ACTION[action_idx]
        
        return action, {
            "action_probs": probs,
            "q_values": q_values.squeeze().cpu().numpy(),
        }
    
    def reset(self):
        pass


def train_rational_agent(layout: str, verbose: bool = True, **kwargs) -> Dict[str, Any]:
    """Train Rational Agent for a layout."""
    config = RationalAgentConfig(layout_name=layout, verbose=verbose, **kwargs)
    trainer = RationalAgentTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--all_layouts", action="store_true")
    args = parser.parse_args()
    
    if args.all_layouts:
        layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring",
                   "forced_coordination", "counter_circuit"]
        for layout in layouts:
            print(f"\n{'='*60}")
            print(f"Training Rational Agent for {layout}")
            train_rational_agent(layout, num_epochs=args.epochs)
    else:
        train_rational_agent(args.layout, num_epochs=args.epochs)
