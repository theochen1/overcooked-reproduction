"""
Evaluate Test_1 trained models against Human Proxy (HP).

Reproduces Figure 4 metrics from the paper:
- SP + HP: Self-play PPO paired with Human Proxy
- PPO_BC + HP: PPO trained with BC partner, paired with Human Proxy
- BC + HP: Behavior Cloning paired with Human Proxy
- SP + SP: Self-play PPO paired with itself (baseline)
"""

import os
import pickle
import argparse
from typing import Dict, List, Tuple
import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

from human_aware_rl.data_dir import DATA_DIR

# Test_1 directory (inside src/human_aware_rl)
TEST1_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Test_1")

# Layouts
LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

# Layout name mapping (paper name -> environment name)
LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}


class BCAgentWrapper(Agent):
    """Wrapper for BC model."""
    
    def __init__(self, model, featurize_fn, stochastic=True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        
    def action(self, state):
        import torch
        import torch.nn.functional as F
        
        obs = self.featurize_fn(state)[self.agent_index]
        obs_flat = obs.flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if self.stochastic:
                action_idx = torch.multinomial(probs, 1).item()
            else:
                action_idx = probs.argmax(dim=-1).item()
        
        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": probs.numpy()}


class PPOAgentWrapper(Agent):
    """Wrapper for JAX PPO model."""
    
    def __init__(self, params, config, lossless_encode_fn, featurize_fn=None, stochastic=True):
        super().__init__()
        self.params = params
        self.config = config
        self.lossless_encode_fn = lossless_encode_fn
        self.featurize_fn = featurize_fn  # Fallback encoding
        self.stochastic = stochastic
        
        # Import JAX
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        
        self.jax = jax
        self.jnp = jnp
        self.nn = nn
        
        # Get config values
        if hasattr(config, 'hidden_dim'):
            self.hidden_dim = config.hidden_dim
            self.num_hidden_layers = config.num_hidden_layers
            self.action_dim = 6
        else:
            self.hidden_dim = config.get("hidden_dim", 64)
            self.num_hidden_layers = config.get("num_hidden_layers", 3)
            self.action_dim = 6
        
        # Infer input dimension from saved params
        self._infer_input_dim()
        
        # Build ActorCritic model
        self._build_network()
    
    def _infer_input_dim(self):
        """Infer input dimension from saved model weights."""
        params = self.params
        if "params" in params:
            params = params["params"]
        
        if "Dense_0" in params:
            kernel = params["Dense_0"]["kernel"]
            self.input_dim = kernel.shape[0]
        else:
            self.input_dim = None
            for key in params:
                if "Dense" in key and "kernel" in params[key]:
                    self.input_dim = params[key]["kernel"].shape[0]
                    break
        
        if self.input_dim is None:
            raise ValueError("Could not infer input dimension from saved params")
        
    def _build_network(self):
        """Build the ActorCritic network for inference (matching training architecture)."""
        from flax import linen as nn
        import jax.numpy as jnp
        
        # Get conv layer config from saved config
        if hasattr(self.config, 'num_conv_layers'):
            num_conv_layers = self.config.num_conv_layers
            num_filters = getattr(self.config, 'num_filters', 25)
        else:
            num_conv_layers = self.config.get("num_conv_layers", 3)
            num_filters = self.config.get("num_filters", 25)
        
        class ActorCritic(nn.Module):
            action_dim: int
            hidden_dim: int = 64
            num_hidden_layers: int = 3
            num_conv_layers: int = 3
            num_filters: int = 25
            
            @nn.compact
            def __call__(self, x):
                # Handle batch dimension
                if len(x.shape) == 1:
                    x = x[None, :]  # Add batch dim
                
                # If input is 4D (image), apply conv layers
                if len(x.shape) == 4:
                    for i in range(self.num_conv_layers):
                        kernel_size = (5, 5) if i == 0 else (3, 3)
                        x = nn.Conv(
                            features=self.num_filters,
                            kernel_size=kernel_size,
                            padding='SAME' if i < self.num_conv_layers - 1 else 'VALID'
                        )(x)
                        x = nn.leaky_relu(x)
                    x = x.reshape((x.shape[0], -1))  # Flatten
                
                # Hidden layers
                for i in range(self.num_hidden_layers):
                    x = nn.Dense(self.hidden_dim)(x)
                    x = nn.leaky_relu(x)
                
                # Actor head
                actor_logits = nn.Dense(self.action_dim)(x)
                
                # Critic head
                critic = nn.Dense(1)(x)
                
                return actor_logits, jnp.squeeze(critic, axis=-1)
        
        self.model = ActorCritic(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_conv_layers=num_conv_layers,
            num_filters=num_filters,
        )
        
    def action(self, state):
        import jax
        import jax.numpy as jnp
        from jax import random
        
        # Get observation using lossless encoding (preserves image structure for conv layers)
        obs = self.lossless_encode_fn(state)[self.agent_index]
        obs = np.array(obs, dtype=np.float32)
        
        # Keep the original shape if it's multi-dimensional (for conv layers)
        # Otherwise flatten
        if len(obs.shape) > 1:
            obs_jnp = jnp.array(obs)
            # Add batch dimension if needed
            if len(obs_jnp.shape) == 3:  # (H, W, C)
                obs_jnp = obs_jnp[None, ...]  # (1, H, W, C)
        else:
            obs_jnp = jnp.array(obs.flatten())
        
        # Handle nested params structure
        if "params" in self.params:
            variables = self.params
        else:
            variables = {"params": self.params}
        
        # Forward pass
        logits, _ = self.model.apply(variables, obs_jnp)
        logits = logits.squeeze()
        probs = jax.nn.softmax(logits)
        
        if self.stochastic:
            key = random.PRNGKey(np.random.randint(0, 2**31))
            action_idx = random.categorical(key, logits).item()
        else:
            action_idx = jnp.argmax(logits).item()
        
        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": np.array(probs)}


def load_bc_model_from_dir(model_dir: str):
    """Load BC model from directory using the existing load function."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model as _load_bc_model
    return _load_bc_model(model_dir, verbose=False)


def load_ppo_model(model_dir: str, verbose: bool = False):
    """Load PPO model from directory."""
    # Find checkpoint
    checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint_")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    latest_checkpoint = sorted(checkpoint_dirs)[-1]
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    
    # Load params
    params_path = os.path.join(checkpoint_path, "params.pkl")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    # Load config
    config_path = os.path.join(checkpoint_path, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    else:
        config = {}
    
    # Debug: print model info
    if verbose:
        p = params
        if "params" in p:
            p = p["params"]
        print(f"    Param keys: {list(p.keys())[:5]}...")  # First 5 keys
        if "Dense_0" in p:
            print(f"    Model input dim: {p['Dense_0']['kernel'].shape[0]}")
        # Print config info
        if hasattr(config, 'num_conv_layers'):
            print(f"    Conv layers: {config.num_conv_layers}")
        if hasattr(config, 'use_lossless_encoding'):
            print(f"    Lossless encoding: {config.use_lossless_encoding}")
    
    return params, config


def evaluate_agent_pair(
    agent1: Agent,
    agent2: Agent,
    layout: str,
    num_games: int = 10,
    horizon: int = 400,
) -> Dict[str, float]:
    """Evaluate an agent pair."""
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": horizon},
    )
    
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)
    agent_pair = AgentPair(agent1, agent2)
    
    results = ae.evaluate_agent_pair(agent_pair, num_games=num_games, display=False)
    
    return {
        "mean_reward": np.mean(results["ep_returns"]),
        "std_reward": np.std(results["ep_returns"]),
        "rewards": results["ep_returns"],
    }


def evaluate_all(num_games: int = 10, verbose: bool = True):
    """Evaluate all agent combinations."""
    import torch
    
    results = {}
    
    for layout in LAYOUTS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {layout}")
            print(f"{'='*60}")
        
        env_layout = LAYOUT_TO_ENV.get(layout, layout)
        
        # Setup featurization
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )
        
        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)
        
        def lossless_encode_fn(state):
            return ae.env.lossless_state_encoding_mdp(state)
        
        layout_results = {}
        
        # Load Human Proxy (BC/test)
        hp_dir = os.path.join(TEST1_DIR, "BC", "test", layout)
        try:
            hp_model, hp_metadata = load_bc_model_from_dir(hp_dir)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            if verbose:
                print(f"  Loaded HP from {hp_dir}")
        except Exception as e:
            print(f"  ERROR loading HP: {e}")
            continue
        
        # Load BC (train)
        bc_dir = os.path.join(TEST1_DIR, "BC", "train", layout)
        try:
            bc_model, bc_metadata = load_bc_model_from_dir(bc_dir)
            bc_agent = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            if verbose:
                print(f"  Loaded BC from {bc_dir}")
        except Exception as e:
            print(f"  ERROR loading BC: {e}")
            bc_agent = None
        
        # Load PPO_SP
        ppo_sp_dir = os.path.join(TEST1_DIR, "PPO_SP", f"{layout}_seed0")
        try:
            ppo_sp_params, ppo_sp_config = load_ppo_model(ppo_sp_dir)
            ppo_sp_agent = PPOAgentWrapper(ppo_sp_params, ppo_sp_config, lossless_encode_fn, featurize_fn, stochastic=True)
            ppo_sp_agent2 = PPOAgentWrapper(ppo_sp_params, ppo_sp_config, lossless_encode_fn, featurize_fn, stochastic=True)
            if verbose:
                print(f"  Loaded PPO_SP from {ppo_sp_dir}")
        except Exception as e:
            print(f"  ERROR loading PPO_SP: {e}")
            import traceback
            traceback.print_exc()
            ppo_sp_agent = None
            ppo_sp_agent2 = None
        
        # Load PPO_BC
        ppo_bc_dir = os.path.join(TEST1_DIR, "PPO_BC", f"{layout}_seed0")
        try:
            ppo_bc_params, ppo_bc_config = load_ppo_model(ppo_bc_dir)
            ppo_bc_agent = PPOAgentWrapper(ppo_bc_params, ppo_bc_config, lossless_encode_fn, featurize_fn, stochastic=True)
            if verbose:
                print(f"  Loaded PPO_BC from {ppo_bc_dir}")
        except Exception as e:
            print(f"  ERROR loading PPO_BC: {e}")
            import traceback
            traceback.print_exc()
            ppo_bc_agent = None
        
        # Evaluate: SP + SP (baseline)
        if ppo_sp_agent and ppo_sp_agent2:
            if verbose:
                print(f"\n  Evaluating SP + SP...")
            try:
                sp_sp_results = evaluate_agent_pair(ppo_sp_agent, ppo_sp_agent2, layout, num_games)
                layout_results["SP+SP"] = sp_sp_results
                if verbose:
                    print(f"    SP+SP: {sp_sp_results['mean_reward']:.1f} ± {sp_sp_results['std_reward']:.1f}")
            except Exception as e:
                print(f"    ERROR: {e}")
        
        # Evaluate: SP + HP
        if ppo_sp_agent:
            if verbose:
                print(f"\n  Evaluating SP + HP...")
            try:
                # Reset agents
                ppo_sp_agent_fresh = PPOAgentWrapper(ppo_sp_params, ppo_sp_config, lossless_encode_fn, featurize_fn, stochastic=True)
                hp_agent_fresh = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
                
                sp_hp_results = evaluate_agent_pair(ppo_sp_agent_fresh, hp_agent_fresh, layout, num_games)
                layout_results["SP+HP"] = sp_hp_results
                if verbose:
                    print(f"    SP+HP: {sp_hp_results['mean_reward']:.1f} ± {sp_hp_results['std_reward']:.1f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Evaluate: PPO_BC + HP
        if ppo_bc_agent:
            if verbose:
                print(f"\n  Evaluating PPO_BC + HP...")
            try:
                ppo_bc_agent_fresh = PPOAgentWrapper(ppo_bc_params, ppo_bc_config, lossless_encode_fn, featurize_fn, stochastic=True)
                hp_agent_fresh = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
                
                ppo_bc_hp_results = evaluate_agent_pair(ppo_bc_agent_fresh, hp_agent_fresh, layout, num_games)
                layout_results["PPO_BC+HP"] = ppo_bc_hp_results
                if verbose:
                    print(f"    PPO_BC+HP: {ppo_bc_hp_results['mean_reward']:.1f} ± {ppo_bc_hp_results['std_reward']:.1f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Evaluate: BC + HP
        if bc_agent:
            if verbose:
                print(f"\n  Evaluating BC + HP...")
            try:
                bc_agent_fresh = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
                hp_agent_fresh = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
                
                bc_hp_results = evaluate_agent_pair(bc_agent_fresh, hp_agent_fresh, layout, num_games)
                layout_results["BC+HP"] = bc_hp_results
                if verbose:
                    print(f"    BC+HP: {bc_hp_results['mean_reward']:.1f} ± {bc_hp_results['std_reward']:.1f}")
            except Exception as e:
                print(f"    ERROR: {e}")
        
        results[layout] = layout_results
    
    return results


def print_summary(results: Dict):
    """Print summary table like Figure 4."""
    print("\n" + "="*80)
    print("SUMMARY: Test_1 Results (Mean Reward ± Std)")
    print("="*80)
    print(f"{'Layout':<25} {'SP+SP':<15} {'SP+HP':<15} {'PPO_BC+HP':<15} {'BC+HP':<15}")
    print("-"*80)
    
    for layout in LAYOUTS:
        if layout not in results:
            continue
        
        lr = results[layout]
        
        sp_sp = f"{lr['SP+SP']['mean_reward']:.1f}±{lr['SP+SP']['std_reward']:.1f}" if "SP+SP" in lr else "N/A"
        sp_hp = f"{lr['SP+HP']['mean_reward']:.1f}±{lr['SP+HP']['std_reward']:.1f}" if "SP+HP" in lr else "N/A"
        ppo_bc_hp = f"{lr['PPO_BC+HP']['mean_reward']:.1f}±{lr['PPO_BC+HP']['std_reward']:.1f}" if "PPO_BC+HP" in lr else "N/A"
        bc_hp = f"{lr['BC+HP']['mean_reward']:.1f}±{lr['BC+HP']['std_reward']:.1f}" if "BC+HP" in lr else "N/A"
        
        print(f"{layout:<25} {sp_sp:<15} {sp_hp:<15} {ppo_bc_hp:<15} {bc_hp:<15}")
    
    print("="*80)
    print("\nKey observations from paper:")
    print("- SP+SP should be much higher than SP+HP (self-play agents fail with humans)")
    print("- PPO_BC+HP should be higher than SP+HP (training with BC helps)")
    print("- BC+HP is somewhere in between")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Test_1 models")
    parser.add_argument("--num_games", type=int, default=10, 
                        help="Number of games per evaluation")
    parser.add_argument("--layout", type=str, default=None,
                        help="Specific layout to evaluate (default: all)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Test_1 Model Evaluation")
    print("="*60)
    print(f"Test_1 directory: {TEST1_DIR}")
    print(f"Number of games: {args.num_games}")
    
    results = evaluate_all(num_games=args.num_games)
    print_summary(results)


if __name__ == "__main__":
    main()

