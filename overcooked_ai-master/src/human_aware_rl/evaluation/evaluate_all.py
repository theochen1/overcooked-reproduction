"""
Batch Evaluation Script for Overcooked AI Agents.

This module provides utilities for evaluating trained agents:
- BC self-play
- PPO self-play
- BC + PPO cross-play
- Agent + Human Proxy evaluation

Usage:
    python -m human_aware_rl.evaluation.evaluate_all --results_dir results/ --output_file eval_results.json
"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.ppo.configs.paper_configs import PAPER_LAYOUTS, LAYOUT_TO_ENV
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR


def load_bc_agent(
    bc_model_dir: str,
    layout_name: str,
    agent_index: int = 0,
    stochastic: bool = True,
) -> Agent:
    """
    Load a BC agent from a model directory.
    
    Args:
        bc_model_dir: Path to BC model directory
        layout_name: Layout name for featurization
        agent_index: Agent index (0 or 1)
        stochastic: Whether to use stochastic action selection
        
    Returns:
        BCAgent instance
    """
    from human_aware_rl.imitation.behavior_cloning import load_bc_model
    from human_aware_rl.imitation.bc_agent import BCAgent
    
    model, bc_params = load_bc_model(bc_model_dir)
    
    # Create agent evaluator for featurization
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name, "old_dynamics": True},
        env_params={"horizon": 400}
    )
    
    featurize_fn = lambda state: ae.env.featurize_state_mdp(state)
    
    return BCAgent(
        model=model,
        bc_params=bc_params,
        featurize_fn=featurize_fn,
        agent_index=agent_index,
        stochastic=stochastic,
    )


def load_jax_agent(
    checkpoint_dir: str,
    layout_name: str,
    agent_index: int = 0,
    stochastic: bool = True,
) -> Agent:
    """
    Load a JAX PPO agent from a checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        layout_name: Layout name for featurization
        agent_index: Agent index (0 or 1)
        stochastic: Whether to use stochastic action selection
        
    Returns:
        JaxPolicyAgent instance
    """
    from human_aware_rl.bridge.jax_agent import JaxPolicyAgent
    
    # Create agent evaluator for featurization
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name, "old_dynamics": True},
        env_params={"horizon": 400}
    )
    
    featurize_fn = lambda state: ae.env.lossless_state_encoding_mdp(state)
    
    return JaxPolicyAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        featurize_fn=featurize_fn,
        agent_index=agent_index,
        stochastic=stochastic,
        use_lossless_encoding=True,
    )


def evaluate_agent_pair(
    agent_0: Agent,
    agent_1: Agent,
    layout_name: str,
    num_games: int = 10,
    horizon: int = 400,
    display: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a pair of agents.
    
    Args:
        agent_0: First agent
        agent_1: Second agent  
        layout_name: Layout name
        num_games: Number of games to play
        horizon: Episode length
        display: Whether to display games
        
    Returns:
        Evaluation results
    """
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name, "old_dynamics": True},
        env_params={"horizon": horizon}
    )
    
    agent_pair = AgentPair(agent_0, agent_1)
    
    results = ae.evaluate_agent_pair(
        agent_pair,
        num_games=num_games,
        display=display,
    )
    
    ep_returns = results["ep_returns"]
    
    return {
        "mean_reward": float(np.mean(ep_returns)),
        "std_reward": float(np.std(ep_returns)),
        "stderr_reward": float(np.std(ep_returns) / np.sqrt(len(ep_returns))),
        "min_reward": float(np.min(ep_returns)),
        "max_reward": float(np.max(ep_returns)),
        "num_games": num_games,
        "ep_returns": [float(r) for r in ep_returns],
    }


def evaluate_bc_self_play(
    bc_model_dir: str,
    layout: str,
    num_games: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate BC agent playing with itself.
    
    Args:
        bc_model_dir: Path to BC model
        layout: Paper layout name
        num_games: Number of games
        
    Returns:
        Evaluation results
    """
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    agent_0 = load_bc_agent(bc_model_dir, env_layout, agent_index=0)
    agent_1 = load_bc_agent(bc_model_dir, env_layout, agent_index=1)
    
    return evaluate_agent_pair(agent_0, agent_1, env_layout, num_games)


def evaluate_ppo_self_play(
    ppo_checkpoint_dir: str,
    layout: str,
    num_games: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate PPO agent playing with itself.
    
    Args:
        ppo_checkpoint_dir: Path to PPO checkpoint
        layout: Paper layout name
        num_games: Number of games
        
    Returns:
        Evaluation results
    """
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    agent_0 = load_jax_agent(ppo_checkpoint_dir, env_layout, agent_index=0)
    agent_1 = load_jax_agent(ppo_checkpoint_dir, env_layout, agent_index=1)
    
    return evaluate_agent_pair(agent_0, agent_1, env_layout, num_games)


def evaluate_bc_with_ppo(
    bc_model_dir: str,
    ppo_checkpoint_dir: str,
    layout: str,
    num_games: int = 10,
    bc_index: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate BC agent paired with PPO agent.
    
    Args:
        bc_model_dir: Path to BC model
        ppo_checkpoint_dir: Path to PPO checkpoint
        layout: Paper layout name
        num_games: Number of games
        bc_index: Index for BC agent (0 or 1)
        
    Returns:
        Evaluation results
    """
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    ppo_index = 1 - bc_index
    
    bc_agent = load_bc_agent(bc_model_dir, env_layout, agent_index=bc_index)
    ppo_agent = load_jax_agent(ppo_checkpoint_dir, env_layout, agent_index=ppo_index)
    
    if bc_index == 0:
        return evaluate_agent_pair(bc_agent, ppo_agent, env_layout, num_games)
    else:
        return evaluate_agent_pair(ppo_agent, bc_agent, env_layout, num_games)


def evaluate_agent_with_human_proxy(
    agent_checkpoint: str,
    agent_type: str,
    hp_model_dir: str,
    layout: str,
    num_games: int = 10,
    agent_index: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate an agent paired with a Human Proxy (HP) model.
    
    Args:
        agent_checkpoint: Path to agent checkpoint
        agent_type: Type of agent ("bc" or "ppo")
        hp_model_dir: Path to Human Proxy BC model
        layout: Paper layout name
        num_games: Number of games
        agent_index: Index for the agent (0 or 1)
        
    Returns:
        Evaluation results
    """
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    hp_index = 1 - agent_index
    
    # Load the agent
    if agent_type.lower() == "bc":
        agent = load_bc_agent(agent_checkpoint, env_layout, agent_index=agent_index)
    elif agent_type.lower() == "ppo":
        agent = load_jax_agent(agent_checkpoint, env_layout, agent_index=agent_index)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Load human proxy
    hp_agent = load_bc_agent(hp_model_dir, env_layout, agent_index=hp_index)
    
    if agent_index == 0:
        return evaluate_agent_pair(agent, hp_agent, env_layout, num_games)
    else:
        return evaluate_agent_pair(hp_agent, agent, env_layout, num_games)


def run_all_evaluations(
    results_dir: str,
    layouts: Optional[List[str]] = None,
    num_games: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all evaluations for trained agents.
    
    Args:
        results_dir: Directory containing trained models
        layouts: List of layouts to evaluate
        num_games: Number of games per evaluation
        verbose: Whether to print progress
        
    Returns:
        Dictionary of all evaluation results
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    
    all_results = {}
    
    for layout in layouts:
        if verbose:
            print(f"\nEvaluating {layout}...")
        
        layout_results = {}
        env_layout = LAYOUT_TO_ENV.get(layout, layout)
        
        # BC self-play
        bc_train_dir = os.path.join(BC_SAVE_DIR, "train", layout)
        if os.path.exists(bc_train_dir):
            try:
                layout_results["bc_bc"] = evaluate_bc_self_play(
                    bc_train_dir, layout, num_games
                )
                if verbose:
                    print(f"  BC+BC: {layout_results['bc_bc']['mean_reward']:.1f}")
            except Exception as e:
                if verbose:
                    print(f"  BC+BC: Error - {e}")
        
        # PPO self-play (find checkpoints)
        ppo_sp_dir = os.path.join(results_dir, "ppo_sp")
        if os.path.exists(ppo_sp_dir):
            # Find latest checkpoint
            for exp_name in os.listdir(ppo_sp_dir):
                if layout in exp_name:
                    exp_dir = os.path.join(ppo_sp_dir, exp_name)
                    checkpoints = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint")]
                    if checkpoints:
                        latest_checkpoint = sorted(checkpoints)[-1]
                        checkpoint_path = os.path.join(exp_dir, latest_checkpoint)
                        try:
                            layout_results["ppo_sp_ppo_sp"] = evaluate_ppo_self_play(
                                checkpoint_path, layout, num_games
                            )
                            if verbose:
                                print(f"  PPO_SP+PPO_SP: {layout_results['ppo_sp_ppo_sp']['mean_reward']:.1f}")
                        except Exception as e:
                            if verbose:
                                print(f"  PPO_SP+PPO_SP: Error - {e}")
                        break
        
        # Human proxy evaluations
        hp_dir = os.path.join(BC_SAVE_DIR, "test", layout)
        if os.path.exists(hp_dir):
            # BC + HP
            if os.path.exists(bc_train_dir):
                try:
                    layout_results["bc_hp"] = evaluate_agent_with_human_proxy(
                        bc_train_dir, "bc", hp_dir, layout, num_games, agent_index=0
                    )
                    if verbose:
                        print(f"  BC+HP: {layout_results['bc_hp']['mean_reward']:.1f}")
                except Exception as e:
                    if verbose:
                        print(f"  BC+HP: Error - {e}")
        
        all_results[layout] = layout_results
    
    return all_results


def save_results(results: Dict[str, Any], output_file: str):
    """Save evaluation results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Overcooked AI agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing trained models"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--layouts",
        type=str,
        default=None,
        help="Comma-separated list of layouts (default: all)"
    )
    
    parser.add_argument(
        "--num_games",
        type=int,
        default=10,
        help="Number of games per evaluation"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    layouts = args.layouts.split(",") if args.layouts else None
    
    results = run_all_evaluations(
        results_dir=args.results_dir,
        layouts=layouts,
        num_games=args.num_games,
        verbose=not args.quiet,
    )
    
    save_results(results, args.output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for layout, layout_results in results.items():
        print(f"\n{layout}:")
        for eval_type, result in layout_results.items():
            if "error" in result:
                print(f"  {eval_type}: Error")
            else:
                print(f"  {eval_type}: {result['mean_reward']:.1f} ± {result['stderr_reward']:.1f}")


if __name__ == "__main__":
    main()

