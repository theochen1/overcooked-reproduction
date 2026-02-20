"""
Hierarchical Behavior Cloning with Goal Inference

This module implements a hierarchical model of human behavior that
explicitly reasons about subgoals/intentions. The model has two levels:

1. High-level: Infer latent goal/intention given state
2. Low-level: Select action given state and inferred goal

This is inspired by:
- Options framework (Sutton et al., 1999)
- Inverse planning / Goal inference (Baker et al., 2009)
- Hierarchical Bayesian models of behavior

The key insight is that human behavior is often better explained as
pursuing subgoals rather than directly mapping states to actions.

Example subgoals in Overcooked:
- "Get onion"
- "Bring onion to pot"
- "Serve soup"
- "Wait for partner"
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
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO
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
class HierarchicalBCConfig:
    """Configuration for Hierarchical BC."""
    
    # Environment
    layout_name: str = "cramped_room"
    data_path: str = CLEAN_2019_HUMAN_DATA_TRAIN
    
    # Model architecture
    num_goals: int = 8  # Number of latent subgoals
    goal_hidden_dims: Tuple[int, ...] = (64,)  # Goal inference network
    policy_hidden_dims: Tuple[int, ...] = (64,)  # Low-level policy
    
    # Goal prior (Dirichlet concentration)
    goal_prior_alpha: float = 1.0
    
    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    
    # Output
    results_dir: str = PPL_SAVE_DIR
    seed: int = 0
    verbose: bool = True


class GoalInferenceNetwork(nn.Module):
    """
    Network that infers P(goal | state).
    
    This network answers: "What subgoal is the agent likely pursuing?"
    """
    
    def __init__(self, state_dim: int, num_goals: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        
        dims = [state_dim] + list(hidden_dims) + [num_goals]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits over goals."""
        return self.network(x)


class GoalConditionedPolicy(nn.Module):
    """
    Policy network that maps (state, goal) -> action distribution.
    
    This network answers: "Given I'm pursuing goal G in state S, what action?"
    """
    
    def __init__(
        self,
        state_dim: int,
        num_goals: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...],
    ):
        super().__init__()
        
        # Embed goals
        self.goal_embedding = nn.Embedding(num_goals, hidden_dims[0] if hidden_dims else 32)
        
        # Input: state + goal embedding
        input_dim = state_dim + (hidden_dims[0] if hidden_dims else 32)
        dims = [input_dim] + list(hidden_dims) + [action_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
            goal: (batch,) goal indices
            
        Returns:
            Action logits (batch, action_dim)
        """
        goal_emb = self.goal_embedding(goal)  # (batch, emb_dim)
        x = torch.cat([state, goal_emb], dim=-1)
        return self.network(x)


class HierarchicalBCModel(PyroModule):
    """
    Hierarchical Behavior Cloning Model.
    
    Generative story:
    1. Given state s, sample goal g ~ P(g | s)
    2. Given state s and goal g, sample action a ~ P(a | s, g)
    
    This model learns:
    - What subgoals explain the data (unsupervised goal discovery)
    - How to achieve each subgoal (goal-conditioned policies)
    """
    
    def __init__(
        self,
        state_dim: int,
        num_goals: int,
        action_dim: int,
        goal_hidden_dims: Tuple[int, ...] = (64,),
        policy_hidden_dims: Tuple[int, ...] = (64,),
        goal_prior_alpha: float = 1.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_goals = num_goals
        self.action_dim = action_dim
        self.goal_prior_alpha = goal_prior_alpha
        
        # Goal inference network (encoder)
        self.goal_inference = GoalInferenceNetwork(
            state_dim, num_goals, goal_hidden_dims
        )
        
        # Goal-conditioned policy (decoder)
        self.policy = GoalConditionedPolicy(
            state_dim, num_goals, action_dim, policy_hidden_dims
        )
    
    @config_enumerate
    def model(self, states: torch.Tensor, actions: torch.Tensor = None):
        """
        Generative model with latent goals.
        
        We use enumeration over the discrete latent variable (goal)
        to marginalize it out during training.
        """
        batch_size = states.shape[0]
        
        # Goal prior: uniform or learned
        goal_prior = torch.ones(self.num_goals, device=states.device) * self.goal_prior_alpha
        
        with pyro.plate("data", batch_size):
            # Sample goal
            goal = pyro.sample("goal", dist.Categorical(logits=goal_prior.log()))
            
            # Action likelihood given state and goal
            action_logits = self.policy(states, goal)
            
            if actions is not None:
                pyro.sample("action", dist.Categorical(logits=action_logits), obs=actions)
        
        return goal
    
    def guide(self, states: torch.Tensor, actions: torch.Tensor = None):
        """
        Variational guide for goal inference.
        
        q(goal | state) approximates P(goal | state, action)
        """
        batch_size = states.shape[0]
        
        # Use neural network to infer goals
        goal_logits = self.goal_inference(states)
        
        with pyro.plate("data", batch_size):
            pyro.sample("goal", dist.Categorical(logits=goal_logits))
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning goal probs and marginalized action probs.
        """
        # Goal probabilities
        goal_logits = self.goal_inference(states)
        goal_probs = F.softmax(goal_logits, dim=-1)  # (batch, num_goals)
        
        # Marginalize over goals for action probs
        batch_size = states.shape[0]
        action_probs = torch.zeros(batch_size, self.action_dim, device=states.device)
        
        for g in range(self.num_goals):
            goal_tensor = torch.full((batch_size,), g, dtype=torch.long, device=states.device)
            action_logits_g = self.policy(states, goal_tensor)
            action_probs_g = F.softmax(action_logits_g, dim=-1)
            
            # Weight by goal probability
            action_probs += goal_probs[:, g:g+1] * action_probs_g
        
        return goal_probs, action_probs
    
    def infer_goal(self, state: torch.Tensor) -> torch.Tensor:
        """Infer most likely goal."""
        goal_logits = self.goal_inference(state)
        return torch.argmax(goal_logits, dim=-1)
    
    def get_action_for_goal(self, state: torch.Tensor, goal: int) -> torch.Tensor:
        """Get action distribution for a specific goal."""
        goal_tensor = torch.full((state.shape[0],), goal, dtype=torch.long, device=state.device)
        action_logits = self.policy(state, goal_tensor)
        return F.softmax(action_logits, dim=-1)


class HierarchicalBCTrainer:
    """Trainer for Hierarchical BC."""
    
    def __init__(self, config: HierarchicalBCConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)
        
        self._setup_environment()
        self._load_data()
        self._setup_model()
        
        if config.verbose:
            print(f"Hierarchical BC Trainer initialized")
            print(f"  Device: {self.device}")
            print(f"  Num goals: {config.num_goals}")
    
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
        
        self.model = HierarchicalBCModel(
            state_dim=self.state_dim,
            num_goals=self.config.num_goals,
            action_dim=self.action_dim,
            goal_hidden_dims=self.config.goal_hidden_dims,
            policy_hidden_dims=self.config.policy_hidden_dims,
            goal_prior_alpha=self.config.goal_prior_alpha,
        ).to(self.device)
        
        # Use TraceEnum_ELBO to marginalize discrete latents
        self.optimizer = ClippedAdam({"lr": self.config.learning_rate})
        self.svi = SVI(
            self.model.model,
            self.model.guide,
            self.optimizer,
            loss=TraceEnum_ELBO(max_plate_nesting=1),
        )
    
    def train(self) -> Dict[str, Any]:
        """Train model."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Training Hierarchical BC")
            print(f"{'='*60}")
        
        num_samples = len(self.train_states)
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        losses = []
        
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
            
            if self.config.verbose and (epoch + 1) % 10 == 0:
                acc = self._compute_accuracy()
                goal_dist = self._get_goal_distribution()
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")
                print(f"  Goal distribution: {goal_dist}")
        
        self._save()
        
        return {"losses": losses}
    
    def _compute_accuracy(self) -> float:
        """Compute accuracy."""
        with torch.no_grad():
            _, action_probs = self.model(self.train_states)
            preds = torch.argmax(action_probs, dim=-1)
            return (preds == self.train_actions).float().mean().item()
    
    def _get_goal_distribution(self) -> str:
        """Get distribution over goals."""
        with torch.no_grad():
            goal_probs, _ = self.model(self.train_states)
            avg_probs = goal_probs.mean(dim=0).cpu().numpy()
            return "[" + ", ".join([f"{p:.2f}" for p in avg_probs]) + "]"
    
    def _save(self):
        """Save model."""
        save_dir = os.path.join(self.config.results_dir, "hierarchical_bc", self.config.layout_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        
        # Save config
        config_dict = {
            "state_dim": self.state_dim,
            "num_goals": self.config.num_goals,
            "action_dim": self.action_dim,
            "goal_hidden_dims": self.config.goal_hidden_dims,
            "policy_hidden_dims": self.config.policy_hidden_dims,
            "layout_name": self.config.layout_name,
        }
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(config_dict, f)
        
        if self.config.verbose:
            print(f"Saved to {save_dir}")


class HierarchicalBCAgent(Agent):
    """Agent using Hierarchical BC model."""
    
    def __init__(
        self,
        model: HierarchicalBCModel,
        featurize_fn,
        agent_index: int = 0,
        stochastic: bool = True,
        device: str = None,
    ):
        super().__init__()
        
        self.model = model
        self.featurize_fn = featurize_fn
        self.agent_index = agent_index
        self.stochastic = stochastic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def action(self, state) -> Tuple[Any, Dict]:
        """Select action with goal inference."""
        obs = self.featurize_fn(state)
        my_obs = obs[self.agent_index]
        
        obs_tensor = torch.tensor(
            my_obs.flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            goal_probs, action_probs = self.model(obs_tensor)
            goal_probs = goal_probs.squeeze().cpu().numpy()
            action_probs = action_probs.squeeze().cpu().numpy()
        
        # Inferred goal
        inferred_goal = np.argmax(goal_probs)
        
        if self.stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)
        
        action = Action.INDEX_TO_ACTION[action_idx]
        
        return action, {
            "action_probs": action_probs,
            "goal_probs": goal_probs,
            "inferred_goal": inferred_goal,
        }
    
    def reset(self):
        pass
    
    @classmethod
    def from_saved(cls, model_dir: str, featurize_fn, **kwargs) -> "HierarchicalBCAgent":
        """Load from saved model."""
        with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        model = HierarchicalBCModel(
            state_dim=config["state_dim"],
            num_goals=config["num_goals"],
            action_dim=config["action_dim"],
            goal_hidden_dims=config["goal_hidden_dims"],
            policy_hidden_dims=config["policy_hidden_dims"],
        ).to(device)
        
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
        
        return cls(model, featurize_fn, **kwargs)


def train_hierarchical_bc(layout: str, verbose: bool = True, **kwargs) -> Dict[str, Any]:
    """Train Hierarchical BC for a layout."""
    config = HierarchicalBCConfig(layout_name=layout, verbose=verbose, **kwargs)
    trainer = HierarchicalBCTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_goals", type=int, default=8)
    parser.add_argument("--all_layouts", action="store_true")
    args = parser.parse_args()
    
    if args.all_layouts:
        layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring",
                   "forced_coordination", "counter_circuit"]
        for layout in layouts:
            print(f"\n{'='*60}")
            print(f"Training Hierarchical BC for {layout}")
            train_hierarchical_bc(layout, num_epochs=args.epochs, num_goals=args.num_goals)
    else:
        train_hierarchical_bc(args.layout, num_epochs=args.epochs, num_goals=args.num_goals)
