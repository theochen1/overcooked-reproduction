"""
Bayesian Behavior Cloning using Pyro

This module implements a Bayesian neural network for behavior cloning.
Unlike standard BC, this model:
1. Maintains uncertainty over weights (epistemic uncertainty)
2. Can express "I don't know" via high entropy predictions
3. Provides calibrated confidence estimates

Key References:
- Weight Uncertainty in Neural Networks (Blundell et al., 2015)
- Practical Variational Inference for Neural Networks (Graves, 2011)
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pyro imports
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal
from pyro.optim import Adam, ClippedAdam

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN


PPL_SAVE_DIR = os.path.join(DATA_DIR, "ppl_runs")


@dataclass
class BayesianBCConfig:
    """Configuration for Bayesian BC training."""
    
    # Environment
    layout_name: str = "cramped_room"
    
    # Data
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    
    # Model architecture
    hidden_dims: Tuple[int, ...] = (64, 64)
    
    # Prior specification (controls regularization)
    prior_scale: float = 1.0  # Scale of weight prior (larger = more uncertainty)
    
    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_particles: int = 1  # Number of samples for ELBO estimation
    
    # Inference
    guide_type: str = "diagonal"  # "diagonal" or "lowrank"
    
    # Prediction
    num_posterior_samples: int = 100  # Samples for uncertainty estimation
    
    # Output
    results_dir: str = PPL_SAVE_DIR
    seed: int = 0
    verbose: bool = True


class BayesianBCModel(PyroModule):
    """
    Bayesian Neural Network for Behavior Cloning.
    
    Instead of point estimates for weights, this model places priors over
    weights and learns a variational posterior. This allows:
    1. Uncertainty quantification (how confident is the model?)
    2. Better generalization (Bayesian regularization)
    3. Detecting out-of-distribution states
    
    The model is a simple MLP: state -> hidden layers -> action logits
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        prior_scale: float = 1.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.prior_scale = prior_scale
        
        # Build layers with Pyro-style weight priors
        dims = [state_dim] + list(hidden_dims) + [action_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            layer = PyroModule[nn.Linear](dims[i], dims[i + 1])
            
            # Place priors on weights and biases
            # Normal(0, prior_scale) prior encourages small weights
            layer.weight = PyroSample(
                dist.Normal(0., prior_scale).expand([dims[i + 1], dims[i]]).to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(0., prior_scale).expand([dims[i + 1]]).to_event(1)
            )
            
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the Bayesian network.
        
        Args:
            x: Input states, shape (batch, state_dim)
            y: Target actions (optional, for training), shape (batch,)
            
        Returns:
            Action logits, shape (batch, action_dim)
        """
        # Pass through hidden layers with ReLU
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        
        # Output layer (no activation)
        logits = self.layers[-1](x)
        
        # If targets provided, score against Categorical likelihood
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        
        return logits
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        guide,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input states
            guide: Trained variational guide
            num_samples: Number of posterior samples
            
        Returns:
            Tuple of (mean_probs, std_probs, entropy)
        """
        predictive = Predictive(self, guide=guide, num_samples=num_samples)
        
        # Get samples from posterior predictive
        with torch.no_grad():
            samples = predictive(x)
        
        # samples contains logits for each posterior sample
        # We need to extract them and compute statistics
        # Note: The way Pyro handles this depends on the model structure
        
        # Alternative: manually sample and compute
        probs_samples = []
        for _ in range(num_samples):
            guide()  # Sample from posterior
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
            probs_samples.append(probs.detach().cpu().numpy())
        
        probs_samples = np.stack(probs_samples, axis=0)  # (num_samples, batch, action_dim)
        
        mean_probs = np.mean(probs_samples, axis=0)
        std_probs = np.std(probs_samples, axis=0)
        
        # Entropy of the mean prediction
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
        
        return mean_probs, std_probs, entropy


class BayesianBCTrainer:
    """Trainer for Bayesian BC models using Stochastic Variational Inference."""
    
    def __init__(self, config: BayesianBCConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Setup
        self._setup_environment()
        self._load_data()
        self._setup_model()
        
        if config.verbose:
            print(f"Bayesian BC Trainer initialized")
            print(f"  Device: {self.device}")
            print(f"  State dim: {self.state_dim}")
            print(f"  Action dim: {self.action_dim}")
            print(f"  Training samples: {len(self.train_states)}")
    
    def _setup_environment(self):
        """Setup environment to get state/action dimensions."""
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
        """Load human demonstration data."""
        data_params = {
            "layouts": [self.config.layout_name],
            "check_trajectories": False,
            "featurize_states": True,
            "data_path": self.config.data_path,
        }
        
        if self.config.verbose:
            print(f"Loading data for {self.config.layout_name}...")
        
        processed_trajs = get_human_human_trajectories(**data_params, silent=not self.config.verbose)
        
        # Extract state-action pairs
        states, actions = [], []
        for ep_idx in range(len(processed_trajs["ep_states"])):
            ep_states = processed_trajs["ep_states"][ep_idx]
            ep_actions = processed_trajs["ep_actions"][ep_idx]
            
            for t in range(len(ep_states)):
                states.append(ep_states[t].flatten())
                actions.append(int(ep_actions[t]))
        
        self.train_states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        self.train_actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        
        if self.config.verbose:
            print(f"Loaded {len(self.train_states)} transitions")
    
    def _setup_model(self):
        """Setup Bayesian model and variational guide."""
        # Clear Pyro param store
        pyro.clear_param_store()
        
        # Create model
        self.model = BayesianBCModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims,
            prior_scale=self.config.prior_scale,
        ).to(self.device)
        
        # Create variational guide (approximate posterior)
        if self.config.guide_type == "diagonal":
            # Mean-field approximation (independent Gaussians)
            self.guide = AutoDiagonalNormal(self.model)
        elif self.config.guide_type == "lowrank":
            # Low-rank + diagonal (captures some correlations)
            self.guide = AutoLowRankMultivariateNormal(self.model, rank=10)
        else:
            raise ValueError(f"Unknown guide type: {self.config.guide_type}")
        
        # Setup optimizer and SVI
        self.optimizer = ClippedAdam({"lr": self.config.learning_rate})
        self.svi = SVI(
            self.model,
            self.guide,
            self.optimizer,
            loss=Trace_ELBO(num_particles=self.config.num_particles),
        )
    
    def train(self) -> Dict[str, Any]:
        """Run SVI training."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Training Bayesian BC")
            print(f"{'='*60}")
        
        num_samples = len(self.train_states)
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        losses = []
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(num_samples)
            
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                
                batch_states = self.train_states[idx]
                batch_actions = self.train_actions[idx]
                
                # SVI step
                loss = self.svi.step(batch_states, batch_actions)
                epoch_loss += loss
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if self.config.verbose and (epoch + 1) % 10 == 0:
                # Compute accuracy on training data
                acc = self._compute_accuracy()
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")
        
        # Save model
        self._save()
        
        return {"losses": losses}
    
    def _compute_accuracy(self) -> float:
        """Compute training accuracy."""
        # Sample from posterior and predict
        self.model.eval()
        
        with torch.no_grad():
            # Get MAP-like prediction by sampling once from guide
            self.guide()
            logits = self.model(self.train_states)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == self.train_actions).float().mean().item()
        
        return acc
    
    def _save(self):
        """Save model and guide."""
        save_dir = os.path.join(self.config.results_dir, "bayesian_bc", self.config.layout_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Pyro param store (contains guide parameters)
        pyro.get_param_store().save(os.path.join(save_dir, "params.pt"))
        
        # Save model architecture info
        config_dict = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.config.hidden_dims,
            "prior_scale": self.config.prior_scale,
            "guide_type": self.config.guide_type,
            "layout_name": self.config.layout_name,
        }
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(config_dict, f)
        
        if self.config.verbose:
            print(f"Saved model to {save_dir}")


class BayesianBCAgent(Agent):
    """
    Agent that uses a trained Bayesian BC model.
    
    This agent can:
    1. Make stochastic predictions by sampling from the posterior
    2. Provide uncertainty estimates for each action
    3. Detect out-of-distribution states via high entropy
    """
    
    def __init__(
        self,
        model: BayesianBCModel,
        guide,
        featurize_fn,
        agent_index: int = 0,
        stochastic: bool = True,
        num_posterior_samples: int = 10,
        device: str = None,
    ):
        super().__init__()
        
        self.model = model
        self.guide = guide
        self.featurize_fn = featurize_fn
        self.agent_index = agent_index
        self.stochastic = stochastic
        self.num_posterior_samples = num_posterior_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def action(self, state) -> Tuple[Any, Dict]:
        """Select action with uncertainty."""
        obs = self.featurize_fn(state)
        my_obs = obs[self.agent_index]
        
        obs_tensor = torch.tensor(
            my_obs.flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        # Sample from posterior multiple times for uncertainty
        probs_samples = []
        with torch.no_grad():
            for _ in range(self.num_posterior_samples):
                self.guide()  # Sample weights from posterior
                logits = self.model(obs_tensor)
                probs = F.softmax(logits, dim=-1)
                probs_samples.append(probs.cpu().numpy())
        
        probs_samples = np.stack(probs_samples, axis=0)  # (num_samples, 1, action_dim)
        mean_probs = np.mean(probs_samples, axis=0).squeeze()  # (action_dim,)
        std_probs = np.std(probs_samples, axis=0).squeeze()
        
        # Entropy as uncertainty measure
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
        
        if self.stochastic:
            action_idx = np.random.choice(len(mean_probs), p=mean_probs)
        else:
            action_idx = np.argmax(mean_probs)
        
        action = Action.INDEX_TO_ACTION[action_idx]
        
        return action, {
            "action_probs": mean_probs,
            "action_std": std_probs,
            "entropy": entropy,
        }
    
    def reset(self):
        """Reset agent state."""
        pass
    
    @classmethod
    def from_saved(
        cls,
        model_dir: str,
        featurize_fn,
        agent_index: int = 0,
        **kwargs,
    ) -> "BayesianBCAgent":
        """Load agent from saved model."""
        # Load config
        with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Recreate model
        model = BayesianBCModel(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dims=config["hidden_dims"],
            prior_scale=config["prior_scale"],
        )
        
        # Create guide
        if config["guide_type"] == "diagonal":
            guide = AutoDiagonalNormal(model)
        else:
            guide = AutoLowRankMultivariateNormal(model, rank=10)
        
        # Load params
        pyro.clear_param_store()
        pyro.get_param_store().load(os.path.join(model_dir, "params.pt"))
        
        return cls(model, guide, featurize_fn, agent_index, **kwargs)


def train_bayesian_bc(layout: str, verbose: bool = True, **kwargs) -> Dict[str, Any]:
    """Train Bayesian BC for a layout."""
    config = BayesianBCConfig(layout_name=layout, verbose=verbose, **kwargs)
    trainer = BayesianBCTrainer(config)
    return trainer.train()


def load_bayesian_bc(model_dir: str, device: str = None):
    """Load a trained Bayesian BC model."""
    # Load config
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Recreate model
    model = BayesianBCModel(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        hidden_dims=config["hidden_dims"],
        prior_scale=config["prior_scale"],
    ).to(device)
    
    # Create guide
    if config["guide_type"] == "diagonal":
        guide = AutoDiagonalNormal(model)
    else:
        guide = AutoLowRankMultivariateNormal(model, rank=10)
    
    # Load params
    pyro.clear_param_store()
    pyro.get_param_store().load(os.path.join(model_dir, "params.pt"), map_location=device)
    
    return model, guide, config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Bayesian BC models")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--prior_scale", type=float, default=1.0)
    parser.add_argument("--all_layouts", action="store_true")
    args = parser.parse_args()
    
    if args.all_layouts:
        layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring",
                   "forced_coordination", "counter_circuit"]
        for layout in layouts:
            print(f"\n{'='*60}")
            print(f"Training Bayesian BC for {layout}")
            train_bayesian_bc(layout, num_epochs=args.epochs, prior_scale=args.prior_scale)
    else:
        train_bayesian_bc(args.layout, num_epochs=args.epochs, prior_scale=args.prior_scale)
