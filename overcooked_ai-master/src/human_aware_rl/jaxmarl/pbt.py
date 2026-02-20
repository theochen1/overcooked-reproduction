"""
Population-Based Training (PBT) for Overcooked AI using JAX.

This module implements PBT as described in the paper:
"On the Utility of Learning about Humans for Human-AI Coordination"

PBT maintains a population of agents that train in parallel, with periodic
evaluation, selection, and hyperparameter mutation.

Reference: https://arxiv.org/abs/1711.09846 (Jaderberg et al., 2017)
"""

import os
import pickle
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax
    from flax import linen as nn
    from flax.training.train_state import TrainState
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training."""
    
    # Environment
    layout_name: str = "cramped_room"
    horizon: int = 400
    num_envs: int = 8
    old_dynamics: bool = True
    
    # Population settings
    population_size: int = 8
    
    # Training
    total_env_steps: int = 8_000_000
    ppo_iteration_timesteps: int = 40000  # Steps per PPO iteration
    
    # PPO settings
    num_minibatches: int = 10
    minibatch_size: int = 2000
    num_epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    
    # Initial hyperparameters (will be mutated)
    initial_learning_rate: float = 1e-3
    initial_entropy_coeff: float = 0.5
    initial_vf_coef: float = 0.1
    
    # Network architecture
    num_hidden_layers: int = 3
    hidden_dim: int = 64
    num_filters: int = 25
    num_conv_layers: int = 3
    use_lstm: bool = False
    cell_size: int = 256
    
    # Reward shaping
    reward_shaping_factor: float = 1.0
    reward_shaping_horizon: float = float('inf')
    use_phi: bool = False
    
    # Mutation settings
    mutation_prob: float = 0.33  # 33% chance per parameter
    mutation_factor_low: float = 0.75
    mutation_factor_high: float = 1.25
    
    # Parameters that can be mutated
    mutable_params: List[str] = field(default_factory=lambda: [
        "learning_rate", "entropy_coeff", "vf_coef", "gae_lambda"
    ])
    
    # Selection
    selection_method: str = "truncation"  # "truncation" or "tournament"
    truncation_fraction: float = 0.25  # Bottom 25% replaced by top 25%
    
    # Evaluation
    eval_interval: int = 5  # Evaluate every N PPO iterations
    eval_num_games: int = 3
    
    # Logging and saving
    log_interval: int = 1
    save_interval: int = 10
    verbose: bool = True
    
    # Output
    results_dir: str = "results"
    experiment_name: str = "pbt_overcooked"


class PBTAgent:
    """A single agent in the PBT population."""
    
    def __init__(
        self,
        agent_id: int,
        config: PBTConfig,
        rng_key: jnp.ndarray,
    ):
        """
        Initialize a PBT agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: PBT configuration
            rng_key: JAX random key
        """
        self.agent_id = agent_id
        self.config = config
        self.rng_key = rng_key
        
        # Hyperparameters (will be mutated)
        self.hyperparams = {
            "learning_rate": config.initial_learning_rate,
            "entropy_coeff": config.initial_entropy_coeff,
            "vf_coef": config.initial_vf_coef,
            "gae_lambda": config.gae_lambda,
        }
        
        # Performance tracking
        self.fitness = 0.0
        self.total_timesteps = 0
        self.training_history = []
        
        # Model parameters (will be initialized by trainer)
        self.params = None
        self.opt_state = None
    
    def copy_from(self, other: "PBTAgent"):
        """Copy weights and hyperparameters from another agent."""
        self.params = copy.deepcopy(other.params)
        self.hyperparams = copy.deepcopy(other.hyperparams)
        # Don't copy fitness - it will be re-evaluated
    
    def mutate(self, rng_key: jnp.ndarray):
        """Mutate hyperparameters."""
        rng_key, *subkeys = random.split(rng_key, len(self.config.mutable_params) + 1)
        
        for i, param_name in enumerate(self.config.mutable_params):
            # Check if we should mutate this parameter
            if random.uniform(subkeys[i]) < self.config.mutation_prob:
                # Choose mutation factor
                if random.uniform(random.fold_in(subkeys[i], 1)) < 0.5:
                    factor = self.config.mutation_factor_low
                else:
                    factor = self.config.mutation_factor_high
                
                # Apply mutation
                old_value = self.hyperparams[param_name]
                new_value = old_value * factor
                
                # Clamp lambda to [0, 1]
                if param_name == "gae_lambda":
                    new_value = np.clip(new_value, 0.0, 1.0)
                
                self.hyperparams[param_name] = new_value


class PBTTrainer:
    """
    Population-Based Training for Overcooked.
    
    Maintains a population of PPO agents that train in parallel,
    with periodic evaluation, selection, and mutation.
    """
    
    def __init__(self, config: PBTConfig):
        """
        Initialize PBT trainer.
        
        Args:
            config: PBT configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for PBT training. "
                "Install with: pip install jax jaxlib flax optax"
            )
        
        self.config = config
        
        # Initialize random keys
        self.master_key = random.PRNGKey(0)
        self.master_key, *agent_keys = random.split(
            self.master_key, config.population_size + 1
        )
        
        # Create population
        self.population = [
            PBTAgent(i, config, agent_keys[i])
            for i in range(config.population_size)
        ]
        
        # Import environment and network
        from human_aware_rl.jaxmarl.overcooked_env import (
            OvercookedJaxEnvConfig,
            VectorizedOvercookedEnv,
        )
        from human_aware_rl.jaxmarl.ppo import ActorCritic
        
        # Create environment
        env_config = OvercookedJaxEnvConfig(
            layout_name=config.layout_name,
            horizon=config.horizon,
            reward_shaping_factor=config.reward_shaping_factor,
            reward_shaping_horizon=config.reward_shaping_horizon,
            use_phi=config.use_phi,
        )
        self.env = VectorizedOvercookedEnv(
            num_envs=config.num_envs,
            config=env_config
        )
        
        # Create network
        self.network = ActorCritic(
            action_dim=self.env.num_actions,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            num_filters=config.num_filters,
            num_conv_layers=config.num_conv_layers,
        )
        
        # Initialize agent parameters
        self._init_agents()
        
        # Logging
        self.train_info = {
            "timesteps": [],
            "population_fitness": [],
            "best_fitness": [],
        }
        
        # Create output directory
        os.makedirs(config.results_dir, exist_ok=True)
    
    def _init_agents(self):
        """Initialize parameters for all agents."""
        dummy_obs = jnp.zeros((1, *self.env.obs_shape))
        
        for agent in self.population:
            self.master_key, init_key = random.split(self.master_key)
            agent.params = self.network.init(init_key, dummy_obs)
    
    def _create_optimizer(self, learning_rate: float):
        """Create optimizer with given learning rate."""
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(learning_rate, eps=1e-5)
        )
    
    def _train_agent_iteration(self, agent: PBTAgent) -> float:
        """
        Run one PPO training iteration for an agent.
        
        Args:
            agent: The agent to train
            
        Returns:
            Average reward from this iteration
        """
        from human_aware_rl.jaxmarl.ppo import Transition
        
        # Create train state with agent's current hyperparameters
        tx = self._create_optimizer(agent.hyperparams["learning_rate"])
        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=agent.params,
            tx=tx,
        )
        
        # Collect rollouts
        states, obs = self.env.reset()
        transitions = []
        episode_rewards = []
        current_rewards = np.zeros(self.config.num_envs)
        
        timesteps_collected = 0
        while timesteps_collected < self.config.ppo_iteration_timesteps:
            agent.rng_key, action_key = random.split(agent.rng_key)
            
            # Get actions from both agents (self-play)
            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            
            logits_0, values_0 = train_state.apply_fn(train_state.params, obs_0)
            logits_1, _ = train_state.apply_fn(train_state.params, obs_1)
            
            actions_0 = random.categorical(action_key, logits_0)
            agent.rng_key, action_key_1 = random.split(agent.rng_key)
            actions_1 = random.categorical(action_key_1, logits_1)
            
            log_probs = jax.nn.log_softmax(logits_0)[jnp.arange(len(actions_0)), actions_0]
            
            # Step environment
            actions = {
                "agent_0": np.array(actions_0),
                "agent_1": np.array(actions_1),
            }
            states, next_obs, rewards, dones, infos = self.env.step(states, actions)
            
            # Store transition
            transition = Transition(
                done=dones["agent_0"],
                action=actions_0,
                value=values_0,
                reward=rewards["agent_0"],
                log_prob=log_probs,
                obs=obs_0,
            )
            transitions.append(transition)
            
            # Track rewards
            current_rewards += np.array(rewards["agent_0"])
            
            # Handle episode ends
            for i, done in enumerate(np.array(dones["__all__"])):
                if done:
                    episode_rewards.append(current_rewards[i])
                    current_rewards[i] = 0
                    states[i], new_obs = self.env.envs[i].reset()
                    next_obs["agent_0"] = next_obs["agent_0"].at[i].set(new_obs["agent_0"])
                    next_obs["agent_1"] = next_obs["agent_1"].at[i].set(new_obs["agent_1"])
            
            obs = next_obs
            timesteps_collected += self.config.num_envs
        
        agent.total_timesteps += timesteps_collected
        
        # Compute advantages
        _, last_value = train_state.apply_fn(train_state.params, obs["agent_0"])
        advantages, returns = self._compute_gae(transitions, last_value, agent)
        
        # PPO update
        batch = {
            "obs": jnp.concatenate([t.obs for t in transitions]),
            "actions": jnp.concatenate([t.action for t in transitions]),
            "log_probs": jnp.concatenate([t.log_prob for t in transitions]),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
        
        batch_size = len(batch["obs"])
        minibatch_size = self.config.minibatch_size
        
        for epoch in range(self.config.num_epochs):
            agent.rng_key, perm_key = random.split(agent.rng_key)
            perm = random.permutation(perm_key, batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                minibatch = {k: v[idx] for k, v in batch.items()}
                train_state = self._ppo_update(train_state, minibatch, agent)
        
        # Update agent params
        agent.params = train_state.params
        
        # Return average reward
        return np.mean(episode_rewards) if episode_rewards else 0.0
    
    def _compute_gae(self, transitions, last_value, agent: PBTAgent):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        gae_lambda = agent.hyperparams["gae_lambda"]
        
        for t in reversed(range(len(transitions))):
            if t == len(transitions) - 1:
                next_value = last_value
            else:
                next_value = transitions[t + 1].value
            
            delta = (
                transitions[t].reward + 
                self.config.gamma * next_value * (1 - transitions[t].done) - 
                transitions[t].value
            )
            gae = delta + self.config.gamma * gae_lambda * (1 - transitions[t].done) * gae
            advantages.insert(0, gae)
        
        advantages = jnp.stack(advantages)
        returns = advantages + jnp.stack([t.value for t in transitions])
        
        return advantages, returns
    
    def _ppo_update(self, train_state: TrainState, batch: Dict, agent: PBTAgent):
        """Perform single PPO update step."""
        ent_coef = agent.hyperparams["entropy_coeff"]
        vf_coef = agent.hyperparams["vf_coef"]
        clip_eps = self.config.clip_eps
        
        def loss_fn(params, obs, actions, old_log_probs, advantages, returns):
            logits, values = train_state.apply_fn(params, obs)
            
            # Actor loss
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(len(actions)), actions]
            ratio = jnp.exp(log_probs - old_log_probs)
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            actor_loss1 = -advantages * ratio
            actor_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            actor_loss = jnp.maximum(actor_loss1, actor_loss2).mean()
            
            # Critic loss
            critic_loss = ((values - returns) ** 2).mean()
            
            # Entropy
            probs = jax.nn.softmax(logits)
            entropy = -(probs * jax.nn.log_softmax(logits)).sum(axis=-1).mean()
            
            total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
            return total_loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(
            train_state.params,
            batch["obs"],
            batch["actions"],
            batch["log_probs"],
            batch["advantages"],
            batch["returns"],
        )
        
        return train_state.apply_gradients(grads=grads)
    
    def _evaluate_agent(self, agent: PBTAgent) -> float:
        """
        Evaluate an agent's fitness.
        
        Args:
            agent: Agent to evaluate
            
        Returns:
            Average reward across evaluation games
        """
        states, obs = self.env.reset()
        total_rewards = []
        current_rewards = np.zeros(self.config.num_envs)
        games_completed = 0
        
        while games_completed < self.config.eval_num_games:
            agent.rng_key, action_key = random.split(agent.rng_key)
            
            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            
            logits_0, _ = self.network.apply(agent.params, obs_0)
            logits_1, _ = self.network.apply(agent.params, obs_1)
            
            # Use greedy actions for evaluation
            actions_0 = jnp.argmax(logits_0, axis=-1)
            actions_1 = jnp.argmax(logits_1, axis=-1)
            
            actions = {
                "agent_0": np.array(actions_0),
                "agent_1": np.array(actions_1),
            }
            states, obs, rewards, dones, infos = self.env.step(states, actions)
            
            current_rewards += np.array(rewards["agent_0"])
            
            for i, done in enumerate(np.array(dones["__all__"])):
                if done:
                    total_rewards.append(current_rewards[i])
                    current_rewards[i] = 0
                    games_completed += 1
                    states[i], new_obs = self.env.envs[i].reset()
                    obs["agent_0"] = obs["agent_0"].at[i].set(new_obs["agent_0"])
                    obs["agent_1"] = obs["agent_1"].at[i].set(new_obs["agent_1"])
        
        return np.mean(total_rewards[:self.config.eval_num_games])
    
    def _exploit_and_explore(self):
        """
        Perform selection and mutation (exploit & explore).
        
        Uses truncation selection: bottom fraction copies from top fraction,
        then mutates hyperparameters.
        """
        # Sort population by fitness
        sorted_population = sorted(
            self.population, 
            key=lambda a: a.fitness, 
            reverse=True
        )
        
        n_replace = int(self.config.population_size * self.config.truncation_fraction)
        
        # Bottom performers copy from top performers
        for i in range(n_replace):
            bottom_agent = sorted_population[-(i+1)]
            top_agent = sorted_population[i]
            
            if self.config.verbose:
                print(f"  Agent {bottom_agent.agent_id} (fitness={bottom_agent.fitness:.1f}) "
                      f"<- Agent {top_agent.agent_id} (fitness={top_agent.fitness:.1f})")
            
            # Copy weights and hyperparameters
            bottom_agent.copy_from(top_agent)
            
            # Mutate hyperparameters
            self.master_key, mutate_key = random.split(self.master_key)
            bottom_agent.mutate(mutate_key)
    
    def train(self) -> Dict[str, Any]:
        """
        Run PBT training.
        
        Returns:
            Dictionary of training results
        """
        total_timesteps = 0
        iteration = 0
        num_iterations = int(
            self.config.total_env_steps / 
            (self.config.ppo_iteration_timesteps * self.config.population_size)
        )
        
        if self.config.verbose:
            print(f"Starting PBT training")
            print(f"Population size: {self.config.population_size}")
            print(f"Total env steps: {self.config.total_env_steps:,}")
            print(f"PPO iteration timesteps: {self.config.ppo_iteration_timesteps:,}")
            print(f"Estimated iterations: {num_iterations}")
        
        start_time = time.time()
        
        while total_timesteps < self.config.total_env_steps:
            iteration += 1
            
            # Train all agents for one iteration
            iteration_rewards = []
            for agent in self.population:
                reward = self._train_agent_iteration(agent)
                iteration_rewards.append(reward)
                total_timesteps += self.config.ppo_iteration_timesteps
            
            # Periodic evaluation
            if iteration % self.config.eval_interval == 0:
                if self.config.verbose:
                    print(f"\nIteration {iteration}: Evaluating population...")
                
                for agent in self.population:
                    agent.fitness = self._evaluate_agent(agent)
                
                fitnesses = [a.fitness for a in self.population]
                best_fitness = max(fitnesses)
                mean_fitness = np.mean(fitnesses)
                
                self.train_info["timesteps"].append(total_timesteps)
                self.train_info["population_fitness"].append(mean_fitness)
                self.train_info["best_fitness"].append(best_fitness)
                
                if self.config.verbose:
                    print(f"  Mean fitness: {mean_fitness:.1f}, Best: {best_fitness:.1f}")
                
                # Exploit and explore
                self._exploit_and_explore()
            
            # Logging
            if iteration % self.config.log_interval == 0 and self.config.verbose:
                elapsed = time.time() - start_time
                fps = total_timesteps / elapsed
                print(f"Iteration {iteration}, Timesteps: {total_timesteps:,}, "
                      f"FPS: {fps:.0f}, Mean reward: {np.mean(iteration_rewards):.1f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(iteration)
        
        # Final save
        self.save_checkpoint(iteration)
        
        if self.config.verbose:
            print(f"\nPBT training complete in {time.time() - start_time:.1f}s")
        
        return {
            "total_timesteps": total_timesteps,
            "train_info": self.train_info,
            "best_agent": max(self.population, key=lambda a: a.fitness),
        }
    
    def save_checkpoint(self, iteration: int):
        """Save population checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.results_dir,
            self.config.experiment_name,
            f"checkpoint_{iteration:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save each agent
        for agent in self.population:
            agent_path = os.path.join(checkpoint_dir, f"agent_{agent.agent_id}.pkl")
            with open(agent_path, "wb") as f:
                pickle.dump({
                    "params": agent.params,
                    "hyperparams": agent.hyperparams,
                    "fitness": agent.fitness,
                    "total_timesteps": agent.total_timesteps,
                }, f)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        if self.config.verbose:
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load population from checkpoint."""
        for agent in self.population:
            agent_path = os.path.join(checkpoint_dir, f"agent_{agent.agent_id}.pkl")
            with open(agent_path, "rb") as f:
                data = pickle.load(f)
                agent.params = data["params"]
                agent.hyperparams = data["hyperparams"]
                agent.fitness = data["fitness"]
                agent.total_timesteps = data["total_timesteps"]
        
        if self.config.verbose:
            print(f"Loaded checkpoint from {checkpoint_dir}")
    
    def get_best_agent(self) -> PBTAgent:
        """Return the agent with highest fitness."""
        return max(self.population, key=lambda a: a.fitness)


def train_pbt(config: PBTConfig) -> Dict[str, Any]:
    """
    Train agents using Population-Based Training.
    
    Args:
        config: PBT configuration
        
    Returns:
        Training results
    """
    trainer = PBTTrainer(config)
    return trainer.train()

