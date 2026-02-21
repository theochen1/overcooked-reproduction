"""
PPO Training for Overcooked using JAX/Flax.

This module provides a JAX-based PPO implementation for training
agents in the Overcooked environment. It supports self-play and
BC-schedule training modes.
"""

from __future__ import annotations  # Defer type hint evaluation

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, TYPE_CHECKING

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
    jax = None
    jnp = None
    nn = None
    optax = None
    TrainState = None

# For type checking only
if TYPE_CHECKING:
    import jax.numpy as jnp
    from flax.training.train_state import TrainState

from human_aware_rl.jaxmarl.overcooked_env import (
    OvercookedJaxEnv,
    OvercookedJaxEnvConfig,
    VectorizedOvercookedEnv,
)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Environment
    layout_name: str = "cramped_room"
    horizon: int = 400
    num_envs: int = 30  # Original uses sim_threads=30
    old_dynamics: bool = True  # Paper uses old dynamics
    
    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 1e-3  # Original paper uses 1e-3
    num_steps: int = 400  # Full episode per env before update (matches horizon)
    num_minibatches: int = 6  # Original uses MINIBATCHES=6
    num_epochs: int = 8  # STEPS_PER_UPDATE in original
    gamma: float = 0.99  # GAMMA in original
    gae_lambda: float = 0.98  # LAM in original (not 0.95!)
    clip_eps: float = 0.05  # CLIPPING in original (not 0.2!)
    clip_eps_end: float = 0.0  # End value for bounded schedules (default keeps strict parity behavior)
    clip_end_fraction: float = 1.0  # Fraction of training to reach clip_eps_end, then hold
    cliprange_schedule: str = "constant"  # "constant", "linear" (to zero), or "linear_to_end"
    ent_coef: float = 0.1  # ENTROPY in original
    vf_coef: float = 0.1  # VF_COEF in original (not 0.5!)
    max_grad_norm: float = 0.1  # MAX_GRAD_NORM in original (not 0.5!)
    kl_coeff: float = 0.2  # KL divergence coefficient
    
    # Learning rate annealing
    # NOTE: Original paper PPO_SP uses LR_ANNEALING=1 which means NO annealing (constant LR)
    # PPO_BC/PPO_HP uses factor-based annealing: LR decays from initial to initial/factor
    use_lr_annealing: bool = False  # Paper SP uses constant LR; BC uses factor-based
    lr_annealing_factor: float = 1.0  # Factor for LR decay: final_lr = initial_lr / factor (1.0 = no decay)
    
    # Value function clipping (original baselines uses this)
    clip_vf: bool = True  # Clip value function updates
    
    # Entropy coefficient settings
    # For paper reproduction: use fixed entropy (use_entropy_annealing=False, ent_coef=0.1)
    entropy_coeff_start: float = 0.1  # Same as ent_coef for paper reproduction
    entropy_coeff_end: float = 0.1
    entropy_coeff_horizon: float = 3e5
    use_entropy_annealing: bool = False  # Paper uses fixed entropy, not annealed
    
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
    use_phi: bool = False  # Paper doesn't use potential-based shaping
    
    # Observation encoding
    use_legacy_encoding: bool = True  # Paper uses 20-channel legacy encoding
    
    # BC schedule for training with BC agents
    # List of (timestep, bc_factor) tuples
    bc_schedule: List[Tuple[int, float]] = field(default_factory=lambda: [(0, 0.0), (float('inf'), 0.0)])
    bc_model_dir: Optional[str] = None
    
    # Logging and saving
    log_interval: int = 1  # Log every update for reward tracking
    save_interval: int = 50  # Save checkpoints frequently
    eval_interval: int = 25  # Evaluate periodically
    eval_num_games: int = 5
    verbose: bool = True
    verbose_debug: bool = False  # Enable detailed per-update diagnostics
    grad_diagnostics: bool = False  # Compute per-loss-term grad norms (expensive)
    
    # Early stopping (disabled by default for paper reproduction)
    use_early_stopping: bool = False  # Set True only for fast/debug mode
    early_stop_patience: int = 100  # Stop if no significant improvement for this many updates
    early_stop_min_reward: float = float('inf')  # Minimum reward threshold (disabled by default)
    
    # Output
    results_dir: str = "results"
    experiment_name: str = "ppo_overcooked"
    seed: int = 0
    
    # Training batch settings (paper values)
    train_batch_size: int = 12000
    num_workers: int = 30


class Transition(NamedTuple):
    """A single transition from environment interaction."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


if JAX_AVAILABLE:
    
    class ActorCritic(nn.Module):
        """Actor-Critic network for PPO.
        
        CRITICAL: Initialization must match original TensorFlow baselines:
        - Conv/Dense hidden layers: glorot_uniform (tf.layers default)
        - Policy output head: orthogonal(scale=0.01) for initial exploration
        - Value output head: glorot_uniform (tf.layers.dense default)
        
        The original TF code uses tf.layers.conv2d and tf.layers.dense which
        default to glorot_uniform, NOT orthogonal. Using orthogonal for hidden
        layers causes entropy to stay at maximum (policy never learns).
        """
        
        action_dim: int
        hidden_dim: int = 64
        num_hidden_layers: int = 3
        num_filters: int = 25
        num_conv_layers: int = 3
        
        @nn.compact
        def __call__(self, x):
            # Match original TensorFlow baselines initialization:
            # - tf.layers.conv2d/dense use glorot_uniform by default (backbone)
            # - Policy output: baselines fc() with ortho_init(scale=0.01)
            # - Value output: baselines fc() with ortho_init(scale=1.0)
            glorot_init = nn.initializers.glorot_uniform()
            ortho_init_small = nn.initializers.orthogonal(scale=0.01)  # For policy output
            ortho_init_default = nn.initializers.orthogonal(scale=1.0)  # For value output
            
            # Handle both flat and image observations
            if len(x.shape) == 4:  # Image observation (batch, H, W, C)
                # Conv layers with glorot init (matching tf.layers.conv2d default)
                for i in range(self.num_conv_layers):
                    kernel_size = (5, 5) if i == 0 else (3, 3)
                    x = nn.Conv(
                        features=self.num_filters,
                        kernel_size=kernel_size,
                        padding='SAME' if i < self.num_conv_layers - 1 else 'VALID',
                        kernel_init=glorot_init,
                        bias_init=nn.initializers.zeros
                    )(x)
                    # CRITICAL: TF uses leaky_relu with alpha=0.2, JAX defaults to 0.01
                    x = nn.leaky_relu(x, negative_slope=0.2)
                
                # Flatten
                x = x.reshape((x.shape[0], -1))
            elif len(x.shape) == 2:  # Flat observation (batch, obs_dim)
                pass  # Already flat
            else:
                raise ValueError(f"Unexpected observation shape: {x.shape}")
            
            # Hidden layers with glorot init (matching tf.layers.dense default)
            for i in range(self.num_hidden_layers):
                x = nn.Dense(
                    self.hidden_dim,
                    kernel_init=glorot_init,
                    bias_init=nn.initializers.zeros
                )(x)
                # CRITICAL: TF uses leaky_relu with alpha=0.2, JAX defaults to 0.01
                x = nn.leaky_relu(x, negative_slope=0.2)
            
            # Actor head - CRITICAL: use orthogonal with small scale (0.01)
            # This matches the original baselines pdfromlatent(init_scale=0.01)
            actor_logits = nn.Dense(
                self.action_dim,
                kernel_init=ortho_init_small,
                bias_init=nn.initializers.zeros
            )(x)
            
            # Critic head with orthogonal init (matching baselines fc() default: ortho_init(1.0))
            critic = nn.Dense(
                1,
                kernel_init=ortho_init_default,
                bias_init=nn.initializers.zeros
            )(x)
            
            return actor_logits, jnp.squeeze(critic, axis=-1)
    
    
    class ActorCriticLSTM(nn.Module):
        """Actor-Critic network with LSTM for PPO.
        
        Uses glorot_uniform for hidden layers (matching TF defaults)
        and orthogonal(scale=0.01) only for policy output.
        """
        
        action_dim: int
        hidden_dim: int = 64
        num_hidden_layers: int = 3
        cell_size: int = 256
        
        @nn.compact
        def __call__(self, x, hidden_state):
            batch_size = x.shape[0]
            glorot_init = nn.initializers.glorot_uniform()
            ortho_init_small = nn.initializers.orthogonal(scale=0.01)
            
            # Hidden layers before LSTM with glorot init (matching TF defaults)
            for i in range(self.num_hidden_layers):
                x = nn.Dense(
                    self.hidden_dim,
                    kernel_init=glorot_init,
                    bias_init=nn.initializers.zeros
                )(x)
                # CRITICAL: TF uses leaky_relu with alpha=0.2, JAX defaults to 0.01
                x = nn.leaky_relu(x, negative_slope=0.2)
            
            # LSTM
            lstm_cell = nn.LSTMCell(features=self.cell_size)
            carry, x = lstm_cell(hidden_state, x)
            
            # Actor head with small scale for exploration
            actor_logits = nn.Dense(
                self.action_dim,
                kernel_init=ortho_init_small,
                bias_init=nn.initializers.zeros
            )(x)
            
            # Critic head with glorot init
            critic = nn.Dense(
                1,
                kernel_init=glorot_init,
                bias_init=nn.initializers.zeros
            )(x)
            
            return actor_logits, jnp.squeeze(critic, axis=-1), carry
        
        def initialize_carry(self, batch_size: int):
            """Initialize LSTM carry state."""
            return (
                jnp.zeros((batch_size, self.cell_size)),
                jnp.zeros((batch_size, self.cell_size))
            )


class PPOTrainer:
    """
    PPO Trainer for Overcooked environment.
    
    Supports self-play and BC-schedule training modes.
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize the PPO trainer.
        
        Args:
            config: PPO training configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for PPO training. "
                "Install with: pip install jax jaxlib flax optax"
            )
        
        self.config = config
        
        # Create environment config
        env_config = OvercookedJaxEnvConfig(
            layout_name=config.layout_name,
            horizon=config.horizon,
            reward_shaping_factor=config.reward_shaping_factor,
            reward_shaping_horizon=config.reward_shaping_horizon,
            use_phi=config.use_phi,
            old_dynamics=config.old_dynamics,  # Critical: paper uses old_dynamics=True
            use_legacy_encoding=config.use_legacy_encoding,  # Paper uses 20-channel encoding
        )
        
        # Create vectorized environment
        self.envs = VectorizedOvercookedEnv(
            num_envs=config.num_envs,
            config=env_config
        )
        
        # Get observation and action space info
        self.obs_shape = self.envs.obs_shape
        self.num_actions = self.envs.num_actions
        
        # Initialize random key from config seed
        self.key = random.PRNGKey(config.seed)
        
        # Also set numpy seed for any numpy-based randomness
        np.random.seed(config.seed)
        
        # Create networks
        self._init_networks()
        
        # BC agent for BC-schedule training
        self.bc_agent = None
        if config.bc_model_dir:
            self._load_bc_agent()
        
        # Logging
        self.train_info = {
            "timesteps": [],
            "episode_returns": [],
            "episode_lengths": [],
            "losses": [],
        }
        
        # Create output directory
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Create JIT-compiled functions (after networks are initialized)
        self._jit_update_step = self._make_jit_update_fn()
        self._jit_inference = self._make_jit_inference_fn()

    def _init_networks(self):
        """Initialize actor-critic networks."""
        self.key, subkey = random.split(self.key)
        
        # Determine observation shape for network initialization
        dummy_obs = jnp.zeros((1, *self.obs_shape))
        
        if self.config.use_lstm:
            self.network = ActorCriticLSTM(
                action_dim=self.num_actions,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
                cell_size=self.config.cell_size,
            )
            dummy_hidden = self.network.initialize_carry(1)
            params = self.network.init(subkey, dummy_obs, dummy_hidden)
        else:
            self.network = ActorCritic(
                action_dim=self.num_actions,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
                num_filters=self.config.num_filters,
                num_conv_layers=self.config.num_conv_layers,
            )
            params = self.network.init(subkey, dummy_obs)
        
        # Create optimizer
        # Use inject_hyperparams so LR can be updated without resetting Adam state
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=self.config.learning_rate, eps=1e-5
            ),
        )
        
        # Create train state
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
        )

    def _load_bc_agent(self):
        """Load BC agent for BC-schedule training."""
        from human_aware_rl.imitation.behavior_cloning import load_bc_model
        from human_aware_rl.imitation.bc_agent import BCAgent
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState as OCState
        
        model, bc_params = load_bc_model(self.config.bc_model_dir)
        
        # Create featurization function using the same environment
        def featurize_fn(state):
            return self.envs.envs[0].base_env.featurize_state_mdp(state)
        
        self.bc_agent = BCAgent(
            model=model,
            bc_params=bc_params,
            featurize_fn=featurize_fn,
            agent_index=1,  # BC agent plays as agent 1
            stochastic=True
        )
        
        # Store reference to reconstruct raw states
        self._bc_featurize_fn = featurize_fn

    def _get_bc_factor(self, timesteps: int) -> float:
        """Get BC factor based on schedule."""
        schedule = self.config.bc_schedule
        
        # Find the two points to interpolate between
        p0 = schedule[0]
        p1 = schedule[1]
        i = 2
        
        while timesteps > p1[0] and i < len(schedule):
            p0 = p1
            p1 = schedule[i]
            i += 1
        
        t0, v0 = p0
        t1, v1 = p1
        
        if t1 == t0:
            return v0
        
        # Linear interpolation
        alpha = (timesteps - t0) / (t1 - t0)
        alpha = min(max(alpha, 0.0), 1.0)
        
        return v0 + alpha * (v1 - v0)

    @staticmethod
    def _select_action(key, logits):
        """Sample action from categorical distribution."""
        return random.categorical(key, logits)


    def _compute_gae(self, transitions, last_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(transitions))):
            if t == len(transitions) - 1:
                next_value = last_value
            else:
                next_value = transitions[t + 1].value
            
            delta = transitions[t].reward + self.config.gamma * next_value * (1 - transitions[t].done) - transitions[t].value
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - transitions[t].done) * gae
            advantages.insert(0, gae)
        
        advantages = jnp.stack(advantages)
        returns = advantages + jnp.stack([t.value for t in transitions])
        
        return advantages, returns

    def _get_entropy_coef(self, timesteps: int) -> float:
        """Get entropy coefficient with annealing."""
        # If entropy annealing is disabled, return fixed entropy coefficient
        if not self.config.use_entropy_annealing:
            # Use entropy_coeff_start as the fixed value (should equal entropy_coeff_end when no annealing)
            return self.config.entropy_coeff_start
        
        # If horizon is 0, return fixed entropy (no annealing)
        if self.config.entropy_coeff_horizon == 0:
            return self.config.entropy_coeff_start
        
        # Linear annealing from start to end over horizon
        alpha = min(1.0, timesteps / self.config.entropy_coeff_horizon)
        return self.config.entropy_coeff_start + alpha * (
            self.config.entropy_coeff_end - self.config.entropy_coeff_start
        )

    def _get_clip_eps_from_frac(self, frac: float) -> float:
        """Get PPO clip epsilon from Baselines-style progress remaining in [0, 1]."""
        schedule = self.config.cliprange_schedule.lower()
        if schedule == "constant":
            return self.config.clip_eps
        if schedule == "linear":
            # Strict linear-to-zero behavior (parity mode).
            frac = min(max(frac, 0.0), 1.0)
            return self.config.clip_eps * frac
        if schedule == "linear_to_end":
            # Bounded linear interpolation: start -> end as progress goes 1 -> 0.
            # Optional end_fraction mirrors SB3 get_linear_fn(start, end, end_fraction):
            # - reaches end value after end_fraction of training
            # - then stays at end for the remaining updates
            frac = min(max(frac, 0.0), 1.0)
            start = self.config.clip_eps
            end = self.config.clip_eps_end
            end_fraction = min(max(float(self.config.clip_end_fraction), 1e-8), 1.0)
            training_progress = 1.0 - frac  # 0 -> 1 over training
            if training_progress >= end_fraction:
                return end
            alpha = training_progress / end_fraction  # 0 -> 1 over [0, end_fraction]
            return start + (end - start) * alpha
        raise ValueError(f"Unknown cliprange_schedule: {self.config.cliprange_schedule}")

    @staticmethod
    def _progress_remaining(update_idx: int, num_updates: int) -> float:
        """Baselines-style progress remaining in [0, 1], inclusive endpoints.

        With num_updates > 1:
          - first update (idx=0) => 1.0
          - last update (idx=num_updates-1) => 0.0
        """
        if num_updates <= 1:
            return 1.0
        return 1.0 - (float(update_idx) / float(num_updates - 1))

    def _update_learning_rate(self, train_state, new_lr: float):
        """Update the learning rate in the optimizer WITHOUT resetting Adam state.
        
        The optimizer was created with optax.inject_hyperparams, so the
        learning rate lives in opt_state[1].hyperparams['learning_rate'].
        Modifying it preserves Adam's momentum and second-moment estimates,
        matching the original TF baselines behavior where LR is a placeholder
        fed at each training step.
        """
        opt_state = train_state.opt_state
        # opt_state is a tuple: (clip_by_global_norm_state, inject_hyperparams_state)
        inject_state = opt_state[1]
        new_hyperparams = {
            k: (jnp.array(new_lr, dtype=jnp.float32) if k == 'learning_rate' else v)
            for k, v in inject_state.hyperparams.items()
        }
        new_inject_state = inject_state._replace(hyperparams=new_hyperparams)
        new_opt_state = (opt_state[0], new_inject_state)
        return train_state.replace(opt_state=new_opt_state)

    def _make_jit_inference_fn(self):
        """Create a JIT-compiled forward pass for rollout collection."""
        use_lstm = self.config.use_lstm
        network = self.network

        @jax.jit
        def _jit_fwd(params, obs):
            if use_lstm:
                return network.apply(params, obs, network.initialize_carry(obs.shape[0]))
            else:
                return network.apply(params, obs)

        return _jit_fwd

    def _make_jit_update_fn(self):
        """Create a JIT-compiled PPO update step.

        Booleans that control tracing branches (use_lstm, clip_vf) are captured
        via closure so JAX traces the correct branch once and reuses it.
        """
        use_lstm = self.config.use_lstm
        clip_vf = self.config.clip_vf
        network = self.network

        max_grad_norm = self.config.max_grad_norm
        grad_diagnostics = self.config.grad_diagnostics

        @jax.jit
        def _jit_step(train_state, obs, actions, old_log_probs, advantages,
                       returns, old_values, ent_coef, vf_coef, clip_eps):
            # Explicitly detach rollout/target tensors from autograd.
            # These are fixed minibatch inputs, matching TF placeholder semantics.
            obs = jnp.asarray(obs, dtype=jnp.float32)
            actions = jnp.asarray(actions, dtype=jnp.int32)
            old_log_probs = jax.lax.stop_gradient(jnp.asarray(old_log_probs, dtype=jnp.float32))
            advantages = jax.lax.stop_gradient(jnp.asarray(advantages, dtype=jnp.float32))
            returns = jax.lax.stop_gradient(jnp.asarray(returns, dtype=jnp.float32))
            old_values = jax.lax.stop_gradient(jnp.asarray(old_values, dtype=jnp.float32))
            ent_coef = jnp.asarray(ent_coef, dtype=jnp.float32)
            vf_coef = jnp.asarray(vf_coef, dtype=jnp.float32)
            clip_eps = jnp.asarray(clip_eps, dtype=jnp.float32)

            def _compute_loss_terms(params):
                if use_lstm:
                    logits, values, _ = train_state.apply_fn(
                        params, obs, network.initialize_carry(obs.shape[0])
                    )
                else:
                    logits, values = train_state.apply_fn(params, obs)

                log_probs = jax.nn.log_softmax(logits)[jnp.arange(len(actions)), actions]
                ratio = jnp.exp(log_probs - old_log_probs)

                actor_loss1 = -advantages * ratio
                actor_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                actor_loss = jnp.maximum(actor_loss1, actor_loss2).mean()

                if clip_vf:
                    values_clipped = old_values + jnp.clip(
                        values - old_values, -clip_eps, clip_eps
                    )
                    critic_loss = 0.5 * jnp.maximum(
                        (values - returns) ** 2,
                        (values_clipped - returns) ** 2,
                    ).mean()
                else:
                    critic_loss = 0.5 * ((values - returns) ** 2).mean()

                probs = jax.nn.softmax(logits)
                entropy = -(probs * jax.nn.log_softmax(logits)).sum(axis=-1).mean()

                total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
                return total_loss, actor_loss, critic_loss, entropy, logits, values, probs, ratio, log_probs

            def loss_fn(params):
                total_loss, actor_loss, critic_loss, entropy, logits, values, probs, ratio, log_probs = _compute_loss_terms(params)

                return total_loss, {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "entropy_coef": ent_coef,
                    "actor_contribution": actor_loss,
                    "critic_contribution": vf_coef * critic_loss,
                    "entropy_contribution": -ent_coef * entropy,
                    "logits_mean": jnp.mean(logits),
                    "logits_std": jnp.std(logits),
                    "logits_min": jnp.min(logits),
                    "logits_max": jnp.max(logits),
                    "probs_max": jnp.max(probs, axis=-1).mean(),
                    "probs_min": jnp.min(probs, axis=-1).mean(),
                    "ratio_mean": jnp.mean(ratio),
                    "ratio_std": jnp.std(ratio),
                    "approx_kl": 0.5 * jnp.mean(jnp.square(log_probs - old_log_probs)),
                    "mean_abs_logratio": jnp.mean(jnp.abs(log_probs - old_log_probs)),
                    "ratio_clipped_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
                    "clip_eps": clip_eps,
                    "values_mean": jnp.mean(values),
                    "values_std": jnp.std(values),
                    "adv_mean": jnp.mean(advantages),
                    "adv_std": jnp.std(advantages),
                    "obs_mean": jnp.mean(obs),
                    "obs_std": jnp.std(obs),
                    "obs_max": jnp.max(obs),
                    "obs_min": jnp.min(obs),
                    "returns_mean": jnp.mean(returns),
                    "returns_std": jnp.std(returns),
                }

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(train_state.params)

            leaves = jax.tree_util.tree_leaves(grads)
            grad_norm_preclip = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))
            clip_scale = jnp.minimum(1.0, max_grad_norm / (grad_norm_preclip + 1e-8))
            grad_norm_postclip_est = grad_norm_preclip * clip_scale
            metrics["grad_norm"] = grad_norm_preclip
            metrics["grad_norm_preclip"] = grad_norm_preclip
            metrics["grad_clip_scale"] = clip_scale
            metrics["grad_norm_postclip_est"] = grad_norm_postclip_est
            metrics["global_clip_threshold"] = jnp.asarray(max_grad_norm, dtype=jnp.float32)

            if grad_diagnostics:
                def actor_term(params):
                    _, actor_loss, _, _, _, _, _, _, _ = _compute_loss_terms(params)
                    return actor_loss

                def critic_term(params):
                    _, _, critic_loss, _, _, _, _, _, _ = _compute_loss_terms(params)
                    return vf_coef * critic_loss

                def entropy_term(params):
                    _, _, _, entropy, _, _, _, _, _ = _compute_loss_terms(params)
                    return -ent_coef * entropy

                actor_grads = jax.grad(actor_term)(train_state.params)
                critic_grads = jax.grad(critic_term)(train_state.params)
                entropy_grads = jax.grad(entropy_term)(train_state.params)

                def tree_l2_norm(tree):
                    lvs = jax.tree_util.tree_leaves(tree)
                    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in lvs))

                actor_norm = tree_l2_norm(actor_grads)
                critic_norm = tree_l2_norm(critic_grads)
                entropy_norm = tree_l2_norm(entropy_grads)

                metrics["grad_actor_term_norm"] = actor_norm
                metrics["grad_critic_term_norm"] = critic_norm
                metrics["grad_entropy_term_norm"] = entropy_norm
                metrics["grad_actor_term_postclip_est"] = actor_norm * clip_scale
                metrics["grad_critic_term_postclip_est"] = critic_norm * clip_scale
                metrics["grad_entropy_term_postclip_est"] = entropy_norm * clip_scale

            new_train_state = train_state.apply_gradients(grads=grads)
            return new_train_state, loss, metrics, grads

        return _jit_step

    @staticmethod
    def _compute_per_layer_grad_norms(grads):
        """Compute gradient norms per layer (Python-level, not JIT-able)."""
        layer_norms = {}
        conv_total = 0.0
        dense_total = 0.0
        actor_total = 0.0
        critic_total = 0.0

        flat_grads = jax.tree_util.tree_leaves_with_path(grads)
        for path, grad in flat_grads:
            path_parts = [str(p.key) if hasattr(p, 'key') else str(p) for p in path]
            name = "/".join(path_parts)

            if 'kernel' in name or 'bias' in name:
                grad_norm = float(jnp.sqrt(jnp.sum(jnp.square(grad))))
                grad_mean = float(jnp.mean(jnp.abs(grad)))

                short_name = path_parts[-2] + "/" + path_parts[-1] if len(path_parts) >= 2 else name
                layer_norms[short_name] = {
                    'norm': grad_norm,
                    'mean_abs': grad_mean,
                    'shape': grad.shape,
                }

                name_lower = name.lower()
                if 'conv' in name_lower:
                    conv_total += grad_norm ** 2
                elif 'dense_0' in name_lower or 'dense_1' in name_lower or 'dense_2' in name_lower:
                    dense_total += grad_norm ** 2

                if hasattr(grad, 'shape') and len(grad.shape) >= 1:
                    if grad.shape[-1] == 6 or 'actor' in name_lower:
                        actor_total += grad_norm ** 2
                    elif grad.shape[-1] == 1 or 'critic' in name_lower or 'value' in name_lower:
                        critic_total += grad_norm ** 2

        return {
            'per_layer': layer_norms,
            'conv_norm': float(jnp.sqrt(conv_total)),
            'dense_norm': float(jnp.sqrt(dense_total)),
            'actor_head_norm': float(jnp.sqrt(actor_total)),
            'critic_head_norm': float(jnp.sqrt(critic_total)),
        }

    def _update(self, train_state, batch, need_debug_grads=False, clip_eps_override=None):
        """Perform PPO update. Core computation is JIT-compiled."""
        ent_coef = self._get_entropy_coef(self.total_timesteps)
        vf_coef = self.config.vf_coef
        clip_eps = clip_eps_override if clip_eps_override is not None else self.config.clip_eps
        
        new_train_state, loss, metrics, grads = self._jit_update_step(
            train_state,
            batch["obs"],
            batch["actions"],
            batch["log_probs"],
            batch["advantages"],
            batch["returns"],
            batch["old_values"],
            jnp.float32(ent_coef),
            jnp.float32(vf_coef),
            jnp.float32(clip_eps),
        )

        if need_debug_grads:
            layer_grad_info = self._compute_per_layer_grad_norms(grads)
            metrics["grad_conv_norm"] = layer_grad_info['conv_norm']
            metrics["grad_dense_norm"] = layer_grad_info['dense_norm']
            metrics["grad_actor_head_norm"] = layer_grad_info['actor_head_norm']
            metrics["grad_critic_head_norm"] = layer_grad_info['critic_head_norm']
            metrics["grad_per_layer"] = layer_grad_info['per_layer']

        return new_train_state, loss, metrics

    def train(self) -> Dict[str, Any]:
        """
        Run PPO training with reward tracking and early stopping.
        
        Returns:
            Dictionary of training results
        """
        self.total_timesteps = 0
        num_updates = self.config.total_timesteps // (self.config.num_envs * self.config.num_steps)
        
        # Reset environments
        states, obs = self.envs.reset()
        
        # Episode reward tracking
        episode_rewards = []  # Per-episode rewards
        current_episode_rewards = np.zeros(self.config.num_envs)  # Running rewards for each env
        recent_rewards = []  # For moving average
        reward_window = 100  # Number of episodes for moving average
        
        # Early stopping tracking
        best_mean_reward = float('-inf')
        no_improvement_count = 0
        
        # Periodic evaluation tracking
        eval_rewards = []  # Store rewards from periodic evaluations
        
        if self.config.verbose:
            print(f"Starting PPO training for {self.config.total_timesteps} timesteps")
            print(f"Layout: {self.config.layout_name}")
            print(f"Num envs: {self.config.num_envs}, Num steps: {self.config.num_steps}")
            print(f"Batch size: {self.config.num_envs * self.config.num_steps}")
            print(f"Entropy annealing: {self.config.entropy_coeff_start} -> {self.config.entropy_coeff_end} over {self.config.entropy_coeff_horizon:.0f} steps")
            print(f"Clip epsilon: {self.config.clip_eps}")
            print(f"Cliprange schedule: {self.config.cliprange_schedule}")
            if self.config.use_early_stopping:
                print(f"Early stopping: ENABLED (patience={self.config.early_stop_patience} updates)")
            else:
                print(f"Early stopping: DISABLED (paper reproduction mode)")
        
        
        start_time = time.time()
        
        # Store initial params for comparison (used for parameter change tracking)
        def compute_param_norm(params):
            """Compute L2 norm of all parameters."""
            leaves = jax.tree_util.tree_leaves(params)
            return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))
        
        initial_param_norm = compute_param_norm(self.train_state.params)
        
        # Initial network diagnostic at step 0 (only when verbose_debug enabled)
        if self.config.verbose_debug:
            print("\n" + "="*80)
            print("INITIAL NETWORK DIAGNOSTIC (Before Training)")
            print("="*80)
            
            # Get initial observation
            init_obs = obs["agent_0"]
            
            # Run network forward pass
            if self.config.use_lstm:
                init_logits, init_values, _ = self.train_state.apply_fn(
                    self.train_state.params,
                    init_obs,
                    self.network.initialize_carry(self.config.num_envs)
                )
            else:
                init_logits, init_values = self.train_state.apply_fn(
                    self.train_state.params, init_obs
                )
            
            # Analyze initial outputs
            init_probs = jax.nn.softmax(init_logits)
            init_entropy = -(init_probs * jax.nn.log_softmax(init_logits)).sum(axis=-1).mean()
            
            print(f"\n[OBSERVATION STATS]")
            print(f"  Shape: {init_obs.shape}")
            print(f"  Mean: {float(jnp.mean(init_obs)):.4f}, Std: {float(jnp.std(init_obs)):.4f}")
            print(f"  Min: {float(jnp.min(init_obs)):.4f}, Max: {float(jnp.max(init_obs)):.4f}")
            
            print(f"\n[INITIAL LOGITS]")
            print(f"  Shape: {init_logits.shape}")
            print(f"  Mean: {float(jnp.mean(init_logits)):.4f}, Std: {float(jnp.std(init_logits)):.4f}")
            print(f"  Min: {float(jnp.min(init_logits)):.4f}, Max: {float(jnp.max(init_logits)):.4f}")
            print(f"  Sample (first env): {[f'{x:.3f}' for x in init_logits[0].tolist()]}")
            
            print(f"\n[INITIAL PROBABILITIES]")
            print(f"  Mean max prob: {float(jnp.max(init_probs, axis=-1).mean()):.4f}")
            print(f"  Uniform would be: {1.0/6:.4f}")
            print(f"  Sample (first env): {[f'{x:.3f}' for x in init_probs[0].tolist()]}")
            
            print(f"\n[INITIAL VALUES]")
            print(f"  Mean: {float(jnp.mean(init_values)):.4f}, Std: {float(jnp.std(init_values)):.4f}")
            print(f"  Min: {float(jnp.min(init_values)):.4f}, Max: {float(jnp.max(init_values)):.4f}")
            
            print(f"\n[INITIAL ENTROPY]")
            print(f"  Entropy: {float(init_entropy):.4f} (max for 6 actions = {float(jnp.log(6)):.4f})")
            
            # Environment configuration
            print(f"\n[ENVIRONMENT CONFIG]")
            print(f"  Layout: {self.config.layout_name}")
            print(f"  Horizon: {self.config.horizon}")
            print(f"  Num Envs: {self.config.num_envs}")
            print(f"  Old Dynamics: {self.config.old_dynamics}")
            print(f"  Legacy Encoding: {self.config.use_legacy_encoding}")
            print(f"  Reward Shaping Factor: {self.config.reward_shaping_factor}")
            print(f"  Reward Shaping Horizon: {self.config.reward_shaping_horizon}")
            
            # PPO Config
            print(f"\n[PPO CONFIG]")
            print(f"  Total Timesteps: {self.config.total_timesteps:,}")
            print(f"  Learning Rate: {self.config.learning_rate}")
            print(f"  LR Annealing: {self.config.use_lr_annealing}")
            print(f"  Num Steps: {self.config.num_steps}")
            print(f"  Num Minibatches: {self.config.num_minibatches}")
            print(f"  Num Epochs: {self.config.num_epochs}")
            print(f"  Gamma: {self.config.gamma}")
            print(f"  GAE Lambda: {self.config.gae_lambda}")
            print(f"  Clip Eps: {self.config.clip_eps}")
            print(f"  Entropy Coef: {self.config.ent_coef}")
            print(f"  VF Coef: {self.config.vf_coef}")
            print(f"  Max Grad Norm: {self.config.max_grad_norm}")
            print(f"  Clip VF: {self.config.clip_vf}")
            print(f"  Grad Diagnostics: {self.config.grad_diagnostics}")
            
            # Network architecture
            print(f"\n[NETWORK ARCHITECTURE]")
            print(f"  Num Conv Layers: {self.config.num_conv_layers}")
            print(f"  Num Filters: {self.config.num_filters}")
            print(f"  Num Hidden Layers: {self.config.num_hidden_layers}")
            print(f"  Hidden Dim: {self.config.hidden_dim}")
            print(f"  Use LSTM: {self.config.use_lstm}")
            print(f"  Param Norm: {initial_param_norm:.4f}")
            
            print("="*80 + "\n")
        
        for update in range(num_updates):
            # Update reward shaping
            self.envs.anneal_reward_shaping(self.total_timesteps)
            
            # Collect rollout with reward tracking
            transitions, states, obs, ep_rewards = self._collect_rollout_with_rewards(
                self.train_state, states, obs, current_episode_rewards
            )
            
            # Update episode rewards
            if ep_rewards:
                episode_rewards.extend(ep_rewards)
                recent_rewards.extend(ep_rewards)
                # Keep only the last 'reward_window' episodes
                if len(recent_rewards) > reward_window:
                    recent_rewards = recent_rewards[-reward_window:]
            
            # Compute advantages
            last_result = self._jit_inference(self.train_state.params, obs["agent_0"])
            if self.config.use_lstm:
                _, last_value, _ = last_result
            else:
                _, last_value = last_result
            
            advantages, returns = self._compute_gae(transitions, last_value)
            
            # Get old values for value function clipping
            old_values = jnp.stack([t.value for t in transitions])
            
            # Flatten batch
            advantages_flat = advantages.reshape(-1)
            returns_flat = returns.reshape(-1)
            old_values_flat = old_values.reshape(-1)
            
            # NOTE: Do NOT normalize advantages here at batch level.
            # The original baselines normalizes per-minibatch in model.train()
            
            batch = {
                "obs": jnp.concatenate([t.obs for t in transitions]),
                "actions": jnp.concatenate([t.action for t in transitions]),
                "log_probs": jnp.concatenate([t.log_prob for t in transitions]),
                "advantages": advantages_flat,
                "returns": returns_flat,
                "old_values": old_values_flat,
            }
            
            # Update learning rate if annealing is enabled
            if self.config.use_lr_annealing:
                progress_remaining = self._progress_remaining(update, num_updates)
                factor = self.config.lr_annealing_factor
                if factor > 1.0:
                    # Factor-based annealing: LR decays from initial_lr to initial_lr / factor
                    # Paper Table 3 uses this for PPO_BC (e.g., factor=3 means LR/3 at end)
                    new_lr = self.config.learning_rate * (
                        1.0 / factor + (1.0 - 1.0 / factor) * progress_remaining
                    )
                else:
                    # Legacy linear-to-zero annealing (factor=1 or not set)
                    new_lr = self.config.learning_rate * progress_remaining
                # Update optimizer with new learning rate
                self.train_state = self._update_learning_rate(self.train_state, new_lr)
            progress_remaining = self._progress_remaining(update, num_updates)
            clip_eps_update = self._get_clip_eps_from_frac(progress_remaining)
            
            # PPO update epochs
            batch_size = len(batch["obs"])
            minibatch_size = max(1, batch_size // self.config.num_minibatches)
            
            need_debug = self.config.verbose_debug and update % 5 == 0
            for epoch in range(self.config.num_epochs):
                self.key, perm_key = random.split(self.key)
                perm = random.permutation(perm_key, batch_size)
                
                for start in range(0, batch_size, minibatch_size):
                    idx = perm[start:start + minibatch_size]
                    minibatch = {k: v[idx] for k, v in batch.items()}
                    
                    # PAPER REPRODUCTION: Normalize advantages per-minibatch (like original baselines)
                    minibatch_advs = minibatch["advantages"]
                    minibatch["advantages"] = (minibatch_advs - minibatch_advs.mean()) / (minibatch_advs.std() + 1e-8)
                    
                    self.train_state, loss, metrics = self._update(
                        self.train_state,
                        minibatch,
                        need_debug_grads=need_debug,
                        clip_eps_override=clip_eps_update,
                    )
            
            # Compute reward statistics
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            
            # Early stopping check with tolerance for variance
            # Only trigger if reward is significantly worse than best (not just slightly lower)
            if recent_rewards and len(recent_rewards) >= 30:  # Need at least 30 episodes
                # Use a tolerance of 1 std or 5% of best reward (whichever is larger)
                tolerance = max(std_reward * 0.5, best_mean_reward * 0.05, 2.0)
                
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    no_improvement_count = 0
                elif mean_reward >= best_mean_reward - tolerance:
                    # Within tolerance - don't count as "no improvement"
                    no_improvement_count = max(0, no_improvement_count - 1)  # Slight recovery
                else:
                    no_improvement_count += 1
                
                if self.config.use_early_stopping and no_improvement_count >= self.config.early_stop_patience:
                    if self.config.verbose:
                        print(f"\nEarly stopping: No significant improvement for {self.config.early_stop_patience} updates")
                        print(f"Best mean reward: {best_mean_reward:.2f}, Current: {mean_reward:.2f}")
                    break
            
            # Logging
            if update % self.config.log_interval == 0 and self.config.verbose:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed if elapsed > 0 else 0
                ent_coef = self._get_entropy_coef(self.total_timesteps)
                
                # Format reward info
                if recent_rewards:
                    reward_str = f"Reward: {mean_reward:.1f}±{std_reward:.1f}"
                else:
                    reward_str = "Reward: N/A"
                
                # Get actual policy entropy from metrics
                policy_entropy = float(metrics.get("entropy", 0.0)) if isinstance(metrics, dict) else 0.0
                approx_kl = float(metrics.get("approx_kl", 0.0)) if isinstance(metrics, dict) else 0.0
                ratio_clipped_frac = float(metrics.get("ratio_clipped_frac", 0.0)) if isinstance(metrics, dict) else 0.0
                clip_eps_val = float(metrics.get("clip_eps", clip_eps_update)) if isinstance(metrics, dict) else clip_eps_update
                
                # Calculate current learning rate for logging
                if self.config.use_lr_annealing:
                    progress_remaining = self._progress_remaining(update, num_updates)
                    factor = self.config.lr_annealing_factor
                    if factor > 1.0:
                        current_lr = self.config.learning_rate * (
                            1.0 / factor + (1.0 - 1.0 / factor) * progress_remaining
                        )
                    else:
                        current_lr = self.config.learning_rate * progress_remaining
                else:
                    current_lr = self.config.learning_rate
                
                # Get sparse reward tracking (actual soup deliveries)
                sparse_sum = getattr(self, '_sparse_reward_sum', 0.0)
                sparse_per_update = sparse_sum / max(1, update + 1)
                
                print(f"Update {update}/{num_updates} | "
                      f"Steps: {self.total_timesteps:,} | "
                      f"FPS: {fps:.0f} | "
                      f"{reward_str} | "
                      f"Sparse: {sparse_per_update:.1f}/ep | "
                      f"Loss: {loss:.4f} | "
                      f"Ent: {policy_entropy:.3f} (coef={ent_coef:.3f}) | "
                      f"KL: {approx_kl:.4f} | "
                      f"ClipFrac: {ratio_clipped_frac:.2%} | "
                      f"ClipEps: {clip_eps_val:.4f} | "
                      f"LR: {current_lr:.2e}")
                
                # Print detailed diagnostics every 5 updates (when verbose_debug enabled)
                if self.config.verbose_debug and update % 5 == 0 and isinstance(metrics, dict):
                    print("\n" + "="*80)
                    print(f"DEBUG DIAGNOSTICS - Update {update}")
                    print("="*80)
                    
                    # Loss breakdown
                    actor_loss = float(metrics.get("actor_loss", 0))
                    critic_loss = float(metrics.get("critic_loss", 0))
                    entropy_val = float(metrics.get("entropy", 0))
                    vf_coef_val = self.config.vf_coef
                    
                    # Loss contributions (actual gradient-driving values)
                    actor_contrib = float(metrics.get("actor_contribution", actor_loss))
                    critic_contrib = float(metrics.get("critic_contribution", critic_loss * vf_coef_val))
                    entropy_contrib = float(metrics.get("entropy_contribution", -ent_coef * entropy_val))
                    
                    print(f"\n[LOSS BREAKDOWN]")
                    print(f"  Actor Loss:  {actor_loss:.6f}")
                    print(f"  Critic Loss: {critic_loss:.6f} (x{vf_coef_val} = {critic_loss * vf_coef_val:.6f})")
                    print(f"  Entropy:     {entropy_val:.6f} (x{ent_coef:.3f} = {entropy_val * ent_coef:.6f})")
                    print(f"  Total Loss:  {loss:.6f}")
                    print(f"  Expected:    {actor_loss + vf_coef_val * critic_loss - ent_coef * entropy_val:.6f}")
                    
                    # Show gradient-driving contributions (absolute values)
                    print(f"\n[LOSS CONTRIBUTION MAGNITUDES (what drives gradients)]")
                    print(f"  |Actor|:    {abs(actor_contrib):.6f}")
                    print(f"  |Critic|:   {abs(critic_contrib):.6f}")
                    print(f"  |Entropy|:  {abs(entropy_contrib):.6f}")
                    total_contrib = abs(actor_contrib) + abs(critic_contrib) + abs(entropy_contrib)
                    if total_contrib > 0:
                        actor_pct = 100 * abs(actor_contrib) / total_contrib
                        critic_pct = 100 * abs(critic_contrib) / total_contrib
                        entropy_pct = 100 * abs(entropy_contrib) / total_contrib
                        print(f"  Percentages: Actor={actor_pct:.1f}%, Critic={critic_pct:.1f}%, Entropy={entropy_pct:.1f}%")
                        if entropy_pct > 80:
                            print(f"  🚨 CRITICAL: Entropy dominates loss ({entropy_pct:.1f}%)!")
                            print(f"     Gradients push toward uniform policy, preventing learning.")
                    
                    # Gradient diagnostics
                    grad_norm = float(metrics.get("grad_norm", 0))
                    print(f"\n[GRADIENTS - GLOBAL]")
                    grad_norm_preclip = float(metrics.get("grad_norm_preclip", grad_norm))
                    grad_norm_postclip_est = float(metrics.get("grad_norm_postclip_est", grad_norm))
                    grad_clip_scale = float(metrics.get("grad_clip_scale", 1.0))
                    global_clip_threshold = float(metrics.get("global_clip_threshold", self.config.max_grad_norm))
                    print(f"  Global Grad Norm (pre-clip):  {grad_norm_preclip:.6f}")
                    print(f"  Clip Threshold:               {global_clip_threshold:.6f}")
                    print(f"  Clip Scale Factor:            {grad_clip_scale:.6f}")
                    print(f"  Global Grad Norm (post est):  {grad_norm_postclip_est:.6f}")
                    if grad_norm < 1e-6:
                        print(f"  ⚠️  WARNING: Gradient norm is extremely small! Policy may not be learning.")
                    elif grad_norm > 10:
                        print(f"  ⚠️  WARNING: Gradient norm is very large! May cause instability.")

                    if self.config.grad_diagnostics:
                        actor_term_norm = float(metrics.get("grad_actor_term_norm", 0.0))
                        critic_term_norm = float(metrics.get("grad_critic_term_norm", 0.0))
                        entropy_term_norm = float(metrics.get("grad_entropy_term_norm", 0.0))
                        actor_term_post = float(metrics.get("grad_actor_term_postclip_est", 0.0))
                        critic_term_post = float(metrics.get("grad_critic_term_postclip_est", 0.0))
                        entropy_term_post = float(metrics.get("grad_entropy_term_postclip_est", 0.0))

                        print(f"\n[GRADIENTS - PER LOSS TERM]")
                        print(f"  Actor Term Norm:   {actor_term_norm:.6f} (post est: {actor_term_post:.6f})")
                        print(f"  Critic Term Norm:  {critic_term_norm:.6f} (post est: {critic_term_post:.6f})")
                        print(f"  Entropy Term Norm: {entropy_term_norm:.6f} (post est: {entropy_term_post:.6f})")
                    
                    # Per-layer gradient diagnostics (CRITICAL for debugging)
                    conv_grad_norm = float(metrics.get("grad_conv_norm", 0))
                    dense_grad_norm = float(metrics.get("grad_dense_norm", 0))
                    actor_grad_norm = float(metrics.get("grad_actor_head_norm", 0))
                    critic_grad_norm = float(metrics.get("grad_critic_head_norm", 0))
                    
                    print(f"\n[GRADIENTS - PER LAYER CATEGORY]")
                    print(f"  Conv Layers:    {conv_grad_norm:.6f}")
                    print(f"  Dense Layers:   {dense_grad_norm:.6f}")
                    print(f"  Actor Head:     {actor_grad_norm:.6f}")
                    print(f"  Critic Head:    {critic_grad_norm:.6f}")
                    
                    # Check for gradient flow issues
                    if actor_grad_norm < 1e-6 and critic_grad_norm > 1e-4:
                        print(f"  🚨 CRITICAL: Actor head gradients near ZERO but critic has gradients!")
                        print(f"     This explains why entropy stays high - policy isn't being updated!")
                    elif actor_grad_norm < critic_grad_norm * 0.01:
                        print(f"  ⚠️  WARNING: Actor head gradients are {critic_grad_norm/max(actor_grad_norm,1e-10):.0f}x smaller than critic!")
                        print(f"     Policy learning may be dominated by value function.")
                    
                    # Print individual layer gradients if available
                    per_layer = metrics.get("grad_per_layer", {})
                    if per_layer:
                        print(f"\n[GRADIENTS - INDIVIDUAL LAYERS]")
                        for layer_name, info in sorted(per_layer.items()):
                            norm = info.get('norm', 0)
                            shape = info.get('shape', ())
                            # Identify if this is actor or critic head
                            label = ""
                            if len(shape) >= 1:
                                if shape[-1] == 6:
                                    label = " [ACTOR HEAD]"
                                elif shape[-1] == 1:
                                    label = " [CRITIC HEAD]"
                            print(f"    {layer_name}: norm={norm:.6f}, shape={shape}{label}")
                    
                    # Parameter change tracking
                    current_param_norm = compute_param_norm(self.train_state.params)
                    param_delta = abs(current_param_norm - initial_param_norm)
                    print(f"\n[PARAMETER CHANGE]")
                    print(f"  Initial Param Norm: {initial_param_norm:.4f}")
                    print(f"  Current Param Norm: {current_param_norm:.4f}")
                    print(f"  Delta: {param_delta:.6f} ({100*param_delta/initial_param_norm:.4f}% of initial)")
                    if param_delta < 1e-4:
                        print(f"  ⚠️  WARNING: Parameters barely changed! Optimizer may not be updating.")
                    
                    # Logits diagnostics (key for understanding policy)
                    logits_mean = float(metrics.get("logits_mean", 0))
                    logits_std = float(metrics.get("logits_std", 0))
                    logits_min = float(metrics.get("logits_min", 0))
                    logits_max = float(metrics.get("logits_max", 0))
                    print(f"\n[LOGITS (Network Output)]")
                    print(f"  Mean: {logits_mean:.4f}, Std: {logits_std:.4f}")
                    print(f"  Min:  {logits_min:.4f}, Max: {logits_max:.4f}")
                    if logits_std < 0.1:
                        print(f"  ⚠️  WARNING: Logits have very low variance! Policy is near-uniform.")
                    
                    # Action probability diagnostics
                    probs_max = float(metrics.get("probs_max", 0))
                    probs_min = float(metrics.get("probs_min", 0))
                    uniform_prob = 1.0 / 6  # 6 actions
                    print(f"\n[ACTION PROBABILITIES]")
                    print(f"  Avg Max Prob: {probs_max:.4f} (uniform would be {uniform_prob:.4f})")
                    print(f"  Avg Min Prob: {probs_min:.4f}")
                    if probs_max < uniform_prob + 0.05:
                        print(f"  ⚠️  WARNING: Policy is essentially uniform! Not learning to prefer actions.")
                    
                    # Ratio diagnostics (PPO clipping)
                    ratio_mean = float(metrics.get("ratio_mean", 0))
                    ratio_std = float(metrics.get("ratio_std", 0))
                    ratio_clipped = float(metrics.get("ratio_clipped_frac", 0))
                    print(f"\n[PPO RATIO (exp(log_prob - old_log_prob))]")
                    print(f"  Mean: {ratio_mean:.4f}, Std: {ratio_std:.4f}")
                    print(f"  Clipped Fraction: {ratio_clipped:.2%}")
                    if ratio_clipped > 0.5:
                        print(f"  ⚠️  WARNING: High clipping fraction! Policy updates may be too aggressive.")
                    
                    # Value function diagnostics
                    values_mean = float(metrics.get("values_mean", 0))
                    values_std = float(metrics.get("values_std", 0))
                    returns_mean = float(metrics.get("returns_mean", 0))
                    returns_std = float(metrics.get("returns_std", 0))
                    print(f"\n[VALUE FUNCTION]")
                    print(f"  Values Mean: {values_mean:.4f}, Std: {values_std:.4f}")
                    print(f"  Returns Mean: {returns_mean:.4f}, Std: {returns_std:.4f}")
                    
                    # Advantage diagnostics
                    adv_mean = float(metrics.get("adv_mean", 0))
                    adv_std = float(metrics.get("adv_std", 0))
                    print(f"\n[ADVANTAGES (after normalization)]")
                    print(f"  Mean: {adv_mean:.4f} (should be ~0), Std: {adv_std:.4f} (should be ~1)")
                    
                    # Observation diagnostics
                    obs_mean = float(metrics.get("obs_mean", 0))
                    obs_std = float(metrics.get("obs_std", 0))
                    obs_max = float(metrics.get("obs_max", 0))
                    obs_min = float(metrics.get("obs_min", 0))
                    print(f"\n[OBSERVATIONS]")
                    print(f"  Mean: {obs_mean:.4f}, Std: {obs_std:.4f}")
                    print(f"  Min: {obs_min:.4f}, Max: {obs_max:.4f}")
                    
                    # Action distribution (cumulative over training)
                    action_names = ["North", "South", "East", "West", "Stay", "Interact"]
                    if hasattr(self, '_rollout_action_counts'):
                        total_actions = self._rollout_action_counts.sum()
                        if total_actions > 0:
                            action_freqs = self._rollout_action_counts / total_actions
                            print(f"\n[ACTION DISTRIBUTION (cumulative)]")
                            for i, (name, freq) in enumerate(zip(action_names, action_freqs)):
                                bar = "█" * int(freq * 30)
                                print(f"  {name:8s}: {freq:.3f} ({self._rollout_action_counts[i]:6d}) {bar}")
                            
                            # Check for action imbalance
                            max_freq = max(action_freqs)
                            min_freq = min(action_freqs)
                            if max_freq > 0.3:
                                print(f"  ⚠️  Action '{action_names[np.argmax(action_freqs)]}' is over-represented")
                    
                    # Summary of potential issues
                    print(f"\n[DIAGNOSIS SUMMARY]")
                    issues = []
                    if entropy_val > 1.7:
                        issues.append("HIGH ENTROPY: Policy is near-random (log(6)=1.79)")
                    if grad_norm < 1e-4:
                        issues.append("VANISHING GRADIENTS: Grads too small to learn")
                    if logits_std < 0.1:
                        issues.append("FLAT LOGITS: Network not differentiating actions")
                    if probs_max < uniform_prob + 0.03:
                        issues.append("UNIFORM POLICY: No action preference emerging")
                    if actor_loss < 0 and abs(actor_loss) < 0.01:
                        issues.append("ACTOR LOSS NEAR ZERO: No policy gradient signal")
                    
                    if issues:
                        for issue in issues:
                            print(f"  ❌ {issue}")
                    else:
                        print(f"  ✓ No obvious issues detected")
                    
                    print("="*80 + "\n")
            
            # Save checkpoint
            if update % self.config.save_interval == 0:
                self.save_checkpoint(update)
            
            # Periodic evaluation with greedy action selection
            if update % self.config.eval_interval == 0:
                eval_reward = self.evaluate(self.config.eval_num_games)
                eval_rewards.append(eval_reward)
                if self.config.verbose:
                    print(f"  [Eval] Update {update}: Mean reward over {self.config.eval_num_games} games = {eval_reward:.2f}")
        
        # Final evaluation
        final_eval_reward = self.evaluate(self.config.eval_num_games)
        eval_rewards.append(final_eval_reward)
        
        # Final save
        self.save_checkpoint(num_updates)
        
        # Compute evaluation statistics
        mean_eval_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        
        if self.config.verbose:
            total_time = time.time() - start_time
            final_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Final mean reward (training): {final_reward:.2f}")
            print(f"Best mean reward (training): {best_mean_reward:.2f}")
            print(f"Mean eval reward (periodic): {mean_eval_reward:.2f}")
            print(f"Final eval reward: {final_eval_reward:.2f}")
            print(f"Total training episodes: {len(episode_rewards)}")
            print(f"Total evaluations: {len(eval_rewards)}")
        
        return {
            "total_timesteps": self.total_timesteps,
            "train_info": self.train_info,
            "episode_rewards": episode_rewards,
            "final_mean_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
            "best_mean_reward": best_mean_reward,
            "eval_rewards": eval_rewards,
            "mean_eval_reward": mean_eval_reward,
            "final_eval_reward": final_eval_reward,
        }

    def _collect_rollout_with_rewards(self, train_state, states, obs, current_episode_rewards):
        """Collect a rollout from all environments with reward tracking."""
        transitions = []
        completed_episode_rewards = []
        
        # DEBUG: Track action distribution in this rollout
        if not hasattr(self, '_rollout_action_counts'):
            self._rollout_action_counts = np.zeros(self.num_actions, dtype=np.int64)
        rollout_action_counts = np.zeros(self.num_actions, dtype=np.int64)
        
        for step in range(self.config.num_steps):
            self.key, action_key = random.split(self.key)
            
            # Get actions for agent 0 from policy
            obs_0 = obs["agent_0"]
            
            result_0 = self._jit_inference(train_state.params, obs_0)
            if self.config.use_lstm:
                logits, values, _ = result_0
            else:
                logits, values = result_0
            
            # Sample actions
            actions_0 = self._select_action(action_key, logits)
            log_probs = jax.nn.log_softmax(logits)[jnp.arange(len(actions_0)), actions_0]
            
            # DEBUG: Track action distribution
            actions_0_np = np.array(actions_0)
            for a in actions_0_np:
                rollout_action_counts[a] += 1
                self._rollout_action_counts[a] += 1
            
            # For agent 1, either use self-play or BC
            bc_factor = self._get_bc_factor(self.total_timesteps)
            
            if bc_factor > 0 and self.bc_agent is not None and np.random.random() < bc_factor:
                # Use BC agent for agent 1
                from overcooked_ai_py.mdp.actions import Action
                bc_actions = []
                for i in range(self.config.num_envs):
                    raw_state = self.envs.envs[i].base_env.state
                    action, _ = self.bc_agent.action(raw_state)
                    action_idx = Action.ACTION_TO_INDEX[action]
                    bc_actions.append(action_idx)
                actions_1 = jnp.array(bc_actions)
            else:
                # Self-play: use same policy for agent 1
                self.key, action_key_1 = random.split(self.key)
                obs_1 = obs["agent_1"]
                
                result_1 = self._jit_inference(train_state.params, obs_1)
                if self.config.use_lstm:
                    logits_1, _, _ = result_1
                else:
                    logits_1, _ = result_1
                
                actions_1 = self._select_action(action_key_1, logits_1)
            
            # Step environments
            actions = {
                "agent_0": np.array(actions_0),
                "agent_1": np.array(actions_1),
            }
            
            states, next_obs, rewards, dones, infos = self.envs.step(states, actions)
            
            # Track rewards (shaped rewards = sparse + reward_shaping * dense)
            # Note: During training, this includes reward shaping which anneals to 0
            shaped_rewards = np.array(rewards["agent_0"])
            current_episode_rewards += shaped_rewards
            
            # Also track sparse rewards (actual soup deliveries) for debugging
            # infos is a list of dicts from vectorized env
            if isinstance(infos, list):
                sparse_rewards = np.array([info.get("sparse_reward", 0) for info in infos])
            else:
                sparse_rewards = np.array([0] * self.config.num_envs)
            
            if not hasattr(self, '_sparse_reward_sum'):
                self._sparse_reward_sum = 0.0
            self._sparse_reward_sum += sparse_rewards.sum()
            
            # Store transition (for agent 0)
            transition = Transition(
                done=dones["agent_0"],
                action=actions_0,
                value=values,
                reward=rewards["agent_0"],
                log_prob=log_probs,
                obs=obs_0,
            )
            transitions.append(transition)
            
            obs = next_obs
            self.total_timesteps += self.config.num_envs
            
            # Handle episode ends
            for i, done in enumerate(np.array(dones["__all__"])):
                if done:
                    # Record completed episode reward
                    completed_episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0
                    
                    # Reset environment
                    states[i], new_obs = self.envs.envs[i].reset()
                    obs["agent_0"] = obs["agent_0"].at[i].set(new_obs["agent_0"])
                    obs["agent_1"] = obs["agent_1"].at[i].set(new_obs["agent_1"])
        
        return transitions, states, obs, completed_episode_rewards

    def save_checkpoint(self, step: int):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.results_dir,
            self.config.experiment_name,
            f"checkpoint_{step:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model parameters
        with open(os.path.join(checkpoint_dir, "params.pkl"), "wb") as f:
            pickle.dump(self.train_state.params, f)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        if self.config.verbose:
            print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load a training checkpoint."""
        with open(os.path.join(checkpoint_dir, "params.pkl"), "rb") as f:
            params = pickle.load(f)
        
        self.train_state = self.train_state.replace(params=params)
        
        if self.config.verbose:
            print(f"Loaded checkpoint from {checkpoint_dir}")

    def get_policy(self):
        """Return the trained policy for evaluation."""
        return self.train_state.params

    def evaluate(self, num_games: int = None) -> float:
        """
        Run evaluation episodes with greedy (deterministic) action selection.
        
        Uses a separate environment instance to avoid polluting training state.
        
        Args:
            num_games: Number of evaluation games to run (default: config.eval_num_games)
            
        Returns:
            Mean reward across all evaluation episodes
        """
        if num_games is None:
            num_games = self.config.eval_num_games
        
        eval_rewards = []
        
        # Create a separate environment for evaluation to avoid polluting training envs
        # CRITICAL: Use SAME encoding as training, but no reward shaping
        from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig
        eval_config = OvercookedJaxEnvConfig(
            layout_name=self.config.layout_name,
            horizon=self.config.horizon,
            old_dynamics=self.config.old_dynamics,  # Match training
            reward_shaping_factor=0.0,  # No reward shaping during evaluation - only sparse
            use_phi=False,  # No potential shaping during eval
            use_legacy_encoding=self.config.use_legacy_encoding,  # CRITICAL: Must match training!
        )
        eval_env = OvercookedJaxEnv(config=eval_config)
        
        # Track action distribution during eval (for debugging)
        eval_action_counts = {i: 0 for i in range(6)}
        first_game_actions = []
        
        for game_idx in range(num_games):
            states, obs = eval_env.reset()
            episode_reward = 0.0
            done = False
            step_count = 0
            
            while not done:
                # Get observations
                obs_0 = jnp.array(obs["agent_0"])[None]  # Add batch dimension
                obs_1 = jnp.array(obs["agent_1"])[None]
                
                result_eval_0 = self._jit_inference(self.train_state.params, obs_0)
                result_eval_1 = self._jit_inference(self.train_state.params, obs_1)
                if self.config.use_lstm:
                    logits_0, _, _ = result_eval_0
                    logits_1, _, _ = result_eval_1
                else:
                    logits_0, _ = result_eval_0
                    logits_1, _ = result_eval_1
                
                # Use STOCHASTIC sampling during evaluation (matching training behavior)
                # With high entropy, argmax causes deterministic loops that prevent coordination
                # The original TF implementation uses stochastic sampling during training episodes
                # which is what ep_sparse_rew_mean tracks
                self.key, key_0, key_1 = random.split(self.key, 3)
                action_0 = int(random.categorical(key_0, logits_0[0]))
                action_1 = int(random.categorical(key_1, logits_1[0]))
                
                # Track actions for debugging
                eval_action_counts[action_0] += 1
                eval_action_counts[action_1] += 1
                step_count += 1
                
                if game_idx == 0 and step_count <= 20:
                    first_game_actions.append((action_0, action_1))
                
                # Step environment - use scalar actions for single env (not arrays!)
                actions = {
                    "agent_0": action_0,
                    "agent_1": action_1,
                }
                
                states, obs, rewards, dones, infos = eval_env.step(states, actions)
                
                # Accumulate reward (handle both array and scalar returns)
                reward = rewards["agent_0"]
                if hasattr(reward, '__getitem__'):
                    episode_reward += float(reward[0])
                else:
                    episode_reward += float(reward)
                
                # Check done (handle both array and scalar returns)
                done_flag = dones["__all__"]
                if hasattr(done_flag, '__getitem__'):
                    done = bool(done_flag[0])
                else:
                    done = bool(done_flag)
            
            eval_rewards.append(episode_reward)
        
        # Print eval details when verbose_debug is enabled
        if self.config.verbose_debug:
            action_names = ["N", "S", "E", "W", "X", "I"]
            actions_str = " ".join([f"({action_names[a0]},{action_names[a1]})" for a0, a1 in first_game_actions[:20]])
            print(f"    [Eval] Game 1: {len(first_game_actions)} steps, First 20 actions: {actions_str}")
            
            total_actions = sum(eval_action_counts.values())
            if total_actions > 0:
                action_names = ["North", "South", "East", "West", "Stay", "Interact"]
                dist_str = ", ".join([f"{action_names[i]}:{eval_action_counts[i]/total_actions*100:.1f}%" 
                                      for i in range(6)])
                print(f"    [Eval] Action distribution: {dist_str}")
        
        return np.mean(eval_rewards)


def train_ppo(config: PPOConfig) -> Dict[str, Any]:
    """
    Train a PPO agent in the Overcooked environment.
    
    Args:
        config: PPO training configuration
        
    Returns:
        Dictionary of training results
    """
    trainer = PPOTrainer(config)
    return trainer.train()

