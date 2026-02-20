"""
Evaluate Run_2 models and generate Paper Figure 4.

This script evaluates all trained models from Run_2:
- BC (train) paired with Human Proxy
- PPO_SP paired with itself (baseline)
- PPO_SP paired with Human Proxy
- PPO_BC paired with Human Proxy

Results are aggregated over 5 seeds with standard error.
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS

# Layouts
LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]

LAYOUT_TO_ENV = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymmetric_advantages",
    "coordination_ring": "coordination_ring",
    "forced_coordination": "forced_coordination",
    "counter_circuit": "counter_circuit_o_1order",
}

SEEDS = [0, 10, 20, 30, 40]

# Run_2 directory (note the space in folder name)
RUN2_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Run_2 ", "models")


class BCAgentWrapper(Agent):
    """Wrapper for BC model."""
    
    def __init__(self, model, featurize_fn, stochastic=True):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.stochastic = stochastic
        
    def action(self, state):
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
    """Wrapper for JAX PPO model - uses jaxmarl env for consistent observation encoding."""
    
    def __init__(self, params, config, layout_name, stochastic=True):
        super().__init__()
        self.params = params
        self.config = config
        self.layout_name = layout_name
        self.stochastic = stochastic
        self.jax_env = None  # Lazy init
        
        import jax
        import jax.numpy as jnp
        
        self.jax = jax
        self.jnp = jnp
        
        # Get the actual params dict
        self.p = params["params"] if "params" in params else params
        
        # Infer model structure from params
        self._analyze_params()
        
    def _analyze_params(self):
        """Analyze params to understand model structure."""
        self.conv_layers = sorted([k for k in self.p.keys() if k.startswith("Conv_")])
        self.dense_layers = sorted([k for k in self.p.keys() if k.startswith("Dense_")])
        self.has_conv = len(self.conv_layers) > 0
    
    def _init_jax_env(self):
        """Initialize JAX environment for observation encoding."""
        from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig
        
        # Map layout names
        env_name = LAYOUT_TO_ENV.get(self.layout_name, self.layout_name)
        config = OvercookedJaxEnvConfig(
            layout_name=env_name,
            horizon=400,
            use_lossless_encoding=True,  # Same as training
            old_dynamics=True,
        )
        self.jax_env = OvercookedJaxEnv(config)
    
    def _forward_conv(self, x, layer_name):
        """Apply a conv layer manually."""
        import jax.numpy as jnp
        from jax import lax
        
        kernel = self.p[layer_name]["kernel"]
        bias = self.p[layer_name]["bias"]
        
        # Determine padding based on layer index
        layer_idx = int(layer_name.split("_")[1])
        if layer_idx < len(self.conv_layers) - 1:
            padding = "SAME"
        else:
            padding = "VALID"
        
        # Apply convolution
        x = lax.conv_general_dilated(
            x, kernel, 
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        x = x + bias
        return x
    
    def _forward_dense(self, x, layer_name):
        """Apply a dense layer."""
        import jax.numpy as jnp
        kernel = self.p[layer_name]["kernel"]
        bias = self.p[layer_name]["bias"]
        return jnp.dot(x, kernel) + bias
    
    def _leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation."""
        import jax.numpy as jnp
        return jnp.where(x > 0, x, alpha * x)
        
    def action(self, state):
        import jax
        import jax.numpy as jnp
        from jax import random
        
        # Initialize JAX env on first call
        if self.jax_env is None:
            self._init_jax_env()
        
        # Get observation using the jaxmarl env's encoding (same as training)
        obs = self.jax_env.base_env.lossless_state_encoding_mdp(state)[self.agent_index]
        obs = np.array(obs, dtype=np.float32)
        
        # Process through conv layers if present
        if self.has_conv and len(obs.shape) >= 2:
            if len(obs.shape) == 2:
                obs = obs[..., None]  # Add channel: (H, W) -> (H, W, 1)
            x = jnp.array(obs)[None, ...]  # Add batch: (H, W, C) -> (1, H, W, C)
            
            # Apply conv layers with saved params
            for layer_name in self.conv_layers:
                x = self._forward_conv(x, layer_name)
                x = self._leaky_relu(x)
            
            # Flatten
            x = x.reshape(-1)
        else:
            x = jnp.array(obs.flatten())
        
        # Dense layers (skip last 2 which are actor/critic heads)
        hidden_layers = self.dense_layers[:-2]
        for layer_name in hidden_layers:
            x = self._forward_dense(x, layer_name)
            x = self._leaky_relu(x)
        
        # Actor head (second to last dense layer)
        actor_layer = self.dense_layers[-2]
        logits = self._forward_dense(x, actor_layer)
        
        probs = jax.nn.softmax(logits)
        
        if self.stochastic:
            key = random.PRNGKey(np.random.randint(0, 2**31))
            action_idx = random.categorical(key, logits).item()
        else:
            action_idx = jnp.argmax(logits).item()
        
        return Action.INDEX_TO_ACTION[action_idx], {"action_probs": np.array(probs)}


def load_bc_model(model_dir: str):
    """Load BC model."""
    from human_aware_rl.imitation.behavior_cloning import load_bc_model as _load
    return _load(model_dir, verbose=False)


def load_ppo_model(model_dir: str, verbose: bool = False):
    """Load PPO model from checkpoint."""
    # Find latest checkpoint
    checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint_")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint in {model_dir}")
    
    latest = sorted(checkpoint_dirs)[-1]
    checkpoint_path = os.path.join(model_dir, latest)
    
    with open(os.path.join(checkpoint_path, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    
    config_path = os.path.join(checkpoint_path, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    else:
        config = {}
    
    if verbose:
        # Debug: print params structure
        p = params["params"] if "params" in params else params
        print(f"    Params keys: {list(p.keys())}")
        if "Dense_0" in p:
            print(f"    Dense_0 kernel shape: {p['Dense_0']['kernel'].shape}")
        print(f"    Config: {config}")
    
    return params, config


def evaluate_pair(agent1, agent2, layout, num_games=5, swapped=False):
    """Evaluate an agent pair."""
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    
    if swapped:
        agent2.set_agent_index(0)
        agent1.set_agent_index(1)
        pair = AgentPair(agent2, agent1)
    else:
        agent1.set_agent_index(0)
        agent2.set_agent_index(1)
        pair = AgentPair(agent1, agent2)
    
    results = ae.evaluate_agent_pair(pair, num_games=num_games, display=False)
    return results["ep_returns"]


def evaluate_all_seeds(
    layout: str,
    agent_type: str,  # 'sp_sp', 'sp_hp', 'ppo_bc_hp', 'bc_hp'
    hp_model,
    bc_model,
    featurize_fn,
    lossless_fn,
    num_games_per_seed: int = 5,
    swapped: bool = False,
    verbose: bool = True,
) -> Dict:
    """Evaluate across all seeds and aggregate."""
    
    all_rewards = []
    
    for seed in SEEDS:
        if agent_type == 'bc_hp':
            # BC doesn't have seeds, just evaluate once
            if seed != SEEDS[0]:
                continue
            agent = BCAgentWrapper(bc_model, featurize_fn, stochastic=True)
            hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
            rewards = evaluate_pair(agent, hp_agent, layout, num_games_per_seed, swapped)
            all_rewards.extend(rewards)
            
        elif agent_type == 'sp_sp':
            # Load PPO_SP for this seed
            sp_dir = os.path.join(RUN2_DIR, "ppo_sp", f"ppo_sp_{layout}_seed{seed}")
            try:
                params, config = load_ppo_model(sp_dir, verbose=(seed == SEEDS[0] and verbose))
                agent1 = PPOAgentWrapper(params, config, layout, stochastic=True)
                agent2 = PPOAgentWrapper(params, config, layout, stochastic=True)
                rewards = evaluate_pair(agent1, agent2, layout, num_games_per_seed, swapped)
                all_rewards.extend(rewards)
            except Exception as e:
                if verbose:
                    print(f"    Seed {seed}: Error - {e}")
                    
        elif agent_type == 'sp_hp':
            # PPO_SP paired with HP
            sp_dir = os.path.join(RUN2_DIR, "ppo_sp", f"ppo_sp_{layout}_seed{seed}")
            try:
                params, config = load_ppo_model(sp_dir)
                sp_agent = PPOAgentWrapper(params, config, layout, stochastic=True)
                hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
                rewards = evaluate_pair(sp_agent, hp_agent, layout, num_games_per_seed, swapped)
                all_rewards.extend(rewards)
            except Exception as e:
                if verbose:
                    print(f"    Seed {seed}: Error - {e}")
                    
        elif agent_type == 'ppo_bc_hp':
            # PPO_BC paired with HP
            bc_dir = os.path.join(RUN2_DIR, "ppo_bc", f"ppo_bc_{layout}_seed{seed}")
            try:
                params, config = load_ppo_model(bc_dir)
                ppo_bc_agent = PPOAgentWrapper(params, config, layout, stochastic=True)
                hp_agent = BCAgentWrapper(hp_model, featurize_fn, stochastic=True)
                rewards = evaluate_pair(ppo_bc_agent, hp_agent, layout, num_games_per_seed, swapped)
                all_rewards.extend(rewards)
            except Exception as e:
                if verbose:
                    print(f"    Seed {seed}: Error - {e}")
    
    if len(all_rewards) == 0:
        return {"mean": 0, "std": 0, "se": 0, "n": 0}
    
    return {
        "mean": float(np.mean(all_rewards)),
        "std": float(np.std(all_rewards)),
        "se": float(np.std(all_rewards) / np.sqrt(len(all_rewards))),
        "n": len(all_rewards),
    }


def run_full_evaluation(num_games_per_seed: int = 5, verbose: bool = True):
    """Run full evaluation for paper Figure 4."""
    
    results = {}
    
    for layout in LAYOUTS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layout: {layout}")
            print(f"{'='*60}")
        
        env_layout = LAYOUT_TO_ENV.get(layout, layout)
        layout_results = {}
        
        # Setup environment
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": env_layout, "old_dynamics": True},
            env_params=DEFAULT_ENV_PARAMS,
        )
        
        def featurize_fn(state):
            return ae.env.featurize_state_mdp(state)
        
        def lossless_fn(state):
            return ae.env.lossless_state_encoding_mdp(state)
        
        # Load HP (BC test)
        hp_dir = os.path.join(RUN2_DIR, "bc_runs", "test", layout)
        try:
            hp_model, _ = load_bc_model(hp_dir)
            if verbose:
                print(f"  ✓ Loaded HP")
        except Exception as e:
            print(f"  ✗ Failed to load HP: {e}")
            continue
        
        # Load BC (train)
        bc_dir = os.path.join(RUN2_DIR, "bc_runs", "train", layout)
        try:
            bc_model, _ = load_bc_model(bc_dir)
            if verbose:
                print(f"  ✓ Loaded BC")
        except Exception as e:
            print(f"  ✗ Failed to load BC: {e}")
            bc_model = None
        
        # Evaluate SP+SP (baseline - white bars)
        if verbose:
            print(f"\n  Evaluating SP+SP...")
        layout_results['sp_sp'] = evaluate_all_seeds(
            layout, 'sp_sp', hp_model, bc_model, featurize_fn, lossless_fn,
            num_games_per_seed, swapped=False, verbose=verbose
        )
        if verbose:
            print(f"    SP+SP: {layout_results['sp_sp']['mean']:.1f} ± {layout_results['sp_sp']['se']:.1f}")
        
        # Evaluate SP+HP (teal bars)
        if verbose:
            print(f"\n  Evaluating SP+HP...")
        layout_results['sp_hp'] = evaluate_all_seeds(
            layout, 'sp_hp', hp_model, bc_model, featurize_fn, lossless_fn,
            num_games_per_seed, swapped=False, verbose=verbose
        )
        layout_results['sp_hp_swapped'] = evaluate_all_seeds(
            layout, 'sp_hp', hp_model, bc_model, featurize_fn, lossless_fn,
            num_games_per_seed, swapped=True, verbose=verbose
        )
        if verbose:
            print(f"    SP+HP: {layout_results['sp_hp']['mean']:.1f} ± {layout_results['sp_hp']['se']:.1f}")
            print(f"    SP+HP (swapped): {layout_results['sp_hp_swapped']['mean']:.1f} ± {layout_results['sp_hp_swapped']['se']:.1f}")
        
        # Evaluate PPO_BC+HP (orange bars)
        if verbose:
            print(f"\n  Evaluating PPO_BC+HP...")
        layout_results['ppo_bc_hp'] = evaluate_all_seeds(
            layout, 'ppo_bc_hp', hp_model, bc_model, featurize_fn, lossless_fn,
            num_games_per_seed, swapped=False, verbose=verbose
        )
        layout_results['ppo_bc_hp_swapped'] = evaluate_all_seeds(
            layout, 'ppo_bc_hp', hp_model, bc_model, featurize_fn, lossless_fn,
            num_games_per_seed, swapped=True, verbose=verbose
        )
        if verbose:
            print(f"    PPO_BC+HP: {layout_results['ppo_bc_hp']['mean']:.1f} ± {layout_results['ppo_bc_hp']['se']:.1f}")
            print(f"    PPO_BC+HP (swapped): {layout_results['ppo_bc_hp_swapped']['mean']:.1f} ± {layout_results['ppo_bc_hp_swapped']['se']:.1f}")
        
        # Evaluate BC+HP (gray bars)
        if bc_model is not None:
            if verbose:
                print(f"\n  Evaluating BC+HP...")
            layout_results['bc_hp'] = evaluate_all_seeds(
                layout, 'bc_hp', hp_model, bc_model, featurize_fn, lossless_fn,
                num_games_per_seed * 5, swapped=False, verbose=verbose  # More games since no seeds
            )
            layout_results['bc_hp_swapped'] = evaluate_all_seeds(
                layout, 'bc_hp', hp_model, bc_model, featurize_fn, lossless_fn,
                num_games_per_seed * 5, swapped=True, verbose=verbose
            )
            if verbose:
                print(f"    BC+HP: {layout_results['bc_hp']['mean']:.1f} ± {layout_results['bc_hp']['se']:.1f}")
                print(f"    BC+HP (swapped): {layout_results['bc_hp_swapped']['mean']:.1f} ± {layout_results['bc_hp_swapped']['se']:.1f}")
        
        results[layout] = layout_results
    
    return results


def print_paper_table(results: Dict):
    """Print results in paper table format."""
    print("\n" + "="*100)
    print("PAPER FIGURE 4 RESULTS")
    print("="*100)
    print(f"{'Layout':<22} {'SP+SP':<12} {'SP+HP':<12} {'PPO_BC+HP':<12} {'BC+HP':<12}")
    print("-"*100)
    
    for layout in LAYOUTS:
        if layout not in results:
            continue
        r = results[layout]
        
        sp_sp = f"{r['sp_sp']['mean']:.1f}±{r['sp_sp']['se']:.1f}" if 'sp_sp' in r else "N/A"
        sp_hp = f"{r['sp_hp']['mean']:.1f}±{r['sp_hp']['se']:.1f}" if 'sp_hp' in r else "N/A"
        ppo_bc = f"{r['ppo_bc_hp']['mean']:.1f}±{r['ppo_bc_hp']['se']:.1f}" if 'ppo_bc_hp' in r else "N/A"
        bc_hp = f"{r['bc_hp']['mean']:.1f}±{r['bc_hp']['se']:.1f}" if 'bc_hp' in r else "N/A"
        
        print(f"{layout:<22} {sp_sp:<12} {sp_hp:<12} {ppo_bc:<12} {bc_hp:<12}")
    
    print("="*100)
    
    # Key findings
    print("\nKEY FINDINGS:")
    print("-"*50)
    for layout in LAYOUTS:
        if layout not in results:
            continue
        r = results[layout]
        
        sp_sp = r.get('sp_sp', {}).get('mean', 0)
        sp_hp = r.get('sp_hp', {}).get('mean', 0)
        ppo_bc = r.get('ppo_bc_hp', {}).get('mean', 0)
        bc_hp = r.get('bc_hp', {}).get('mean', 0)
        
        drop = ((sp_sp - sp_hp) / sp_sp * 100) if sp_sp > 0 else 0
        improvement = ((ppo_bc - sp_hp) / sp_hp * 100) if sp_hp > 0 else 0
        
        print(f"{layout}:")
        print(f"  SP drop with HP: {drop:.0f}% (SP+SP: {sp_sp:.0f} → SP+HP: {sp_hp:.0f})")
        print(f"  PPO_BC improvement over SP: {improvement:.0f}% (SP+HP: {sp_hp:.0f} → PPO_BC+HP: {ppo_bc:.0f})")


def plot_figure_4(results: Dict, save_path: str = None):
    """Generate Figure 4 style plot."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Colors from paper
    COLORS = {
        'self_play': '#4A90A4',      # Teal
        'ppo_bc': '#E8944A',         # Orange
        'bc': '#808080',             # Gray
    }
    
    LAYOUT_NAMES = {
        'cramped_room': 'Cramped Rm.',
        'asymmetric_advantages': 'Asymm. Adv.',
        'coordination_ring': 'Coord. Ring',
        'forced_coordination': 'Forced Coord.',
        'counter_circuit': 'Counter Circ.',
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(LAYOUTS))
    width = 0.15
    
    # Extract data
    sp_sp = [results[l].get('sp_sp', {}).get('mean', 0) for l in LAYOUTS]
    sp_sp_se = [results[l].get('sp_sp', {}).get('se', 0) for l in LAYOUTS]
    
    sp_hp = [results[l].get('sp_hp', {}).get('mean', 0) for l in LAYOUTS]
    sp_hp_se = [results[l].get('sp_hp', {}).get('se', 0) for l in LAYOUTS]
    sp_hp_sw = [results[l].get('sp_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    sp_hp_sw_se = [results[l].get('sp_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    ppo_bc = [results[l].get('ppo_bc_hp', {}).get('mean', 0) for l in LAYOUTS]
    ppo_bc_se = [results[l].get('ppo_bc_hp', {}).get('se', 0) for l in LAYOUTS]
    ppo_bc_sw = [results[l].get('ppo_bc_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    ppo_bc_sw_se = [results[l].get('ppo_bc_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    bc_hp = [results[l].get('bc_hp', {}).get('mean', 0) for l in LAYOUTS]
    bc_hp_se = [results[l].get('bc_hp', {}).get('se', 0) for l in LAYOUTS]
    bc_hp_sw = [results[l].get('bc_hp_swapped', {}).get('mean', 0) for l in LAYOUTS]
    bc_hp_sw_se = [results[l].get('bc_hp_swapped', {}).get('se', 0) for l in LAYOUTS]
    
    # Plot bars
    ax.bar(x - 2.5*width, sp_sp, width, yerr=sp_sp_se,
           color='white', edgecolor='black', linewidth=1.5, capsize=2, label='SP+SP')
    
    ax.bar(x - 1.5*width, sp_hp, width, yerr=sp_hp_se,
           color=COLORS['self_play'], edgecolor='black', linewidth=0.5, capsize=2, label='SP+H$_{Proxy}$')
    ax.bar(x - 0.5*width, sp_hp_sw, width, yerr=sp_hp_sw_se,
           color=COLORS['self_play'], edgecolor='black', linewidth=0.5, hatch='///', capsize=2)
    
    ax.bar(x + 0.5*width, ppo_bc, width, yerr=ppo_bc_se,
           color=COLORS['ppo_bc'], edgecolor='black', linewidth=0.5, capsize=2, label='PPO$_{BC}$+H$_{Proxy}$')
    ax.bar(x + 1.5*width, ppo_bc_sw, width, yerr=ppo_bc_sw_se,
           color=COLORS['ppo_bc'], edgecolor='black', linewidth=0.5, hatch='///', capsize=2)
    
    ax.bar(x + 2.5*width, bc_hp, width, yerr=bc_hp_se,
           color=COLORS['bc'], edgecolor='black', linewidth=0.5, capsize=2, label='BC+H$_{Proxy}$')
    ax.bar(x + 3.5*width, bc_hp_sw, width, yerr=bc_hp_sw_se,
           color=COLORS['bc'], edgecolor='black', linewidth=0.5, hatch='///', capsize=2)
    
    # Formatting
    ax.set_ylabel('Average reward per episode', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYOUT_NAMES[l] for l in LAYOUTS], fontsize=11)
    ax.set_ylim(0, max(sp_sp) * 1.2 if max(sp_sp) > 0 else 300)
    ax.set_title('Performance with human proxy model\n(Comparison with agents trained in self-play)', 
                 fontsize=13, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='SP+SP'),
        mpatches.Patch(facecolor=COLORS['self_play'], edgecolor='black', label='SP+H$_{Proxy}$'),
        mpatches.Patch(facecolor=COLORS['ppo_bc'], edgecolor='black', label='PPO$_{BC}$+H$_{Proxy}$'),
        mpatches.Patch(facecolor=COLORS['bc'], edgecolor='black', label='BC+H$_{Proxy}$'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Switched indices'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Run_2 models (Paper Figure 4)")
    parser.add_argument("--num_games", type=int, default=5,
                        help="Number of games per seed (total = num_games * 5 seeds)")
    parser.add_argument("--save_results", type=str, default="run2_paper_results.json",
                        help="Save results to JSON")
    parser.add_argument("--save_plot", type=str, default="run2_figure4.png",
                        help="Save plot to file")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Run_2 Paper Evaluation (Figure 4)")
    print("="*60)
    print(f"Models: {RUN2_DIR}")
    print(f"Games per seed: {args.num_games}")
    print(f"Total games per agent pair: {args.num_games * len(SEEDS)}")
    
    results = run_full_evaluation(num_games_per_seed=args.num_games)
    print_paper_table(results)
    
    # Save results
    with open(args.save_results, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save_results}")
    
    # Plot
    if not args.no_plot:
        try:
            plot_figure_4(results, args.save_plot)
        except ImportError:
            print("\nMatplotlib not available - skipping plot")


if __name__ == "__main__":
    main()

