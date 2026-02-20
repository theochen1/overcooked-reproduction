"""
Evaluation utilities for Overcooked agents.

This module provides functions for evaluating trained agents
(from JAX/PyTorch) in the Overcooked environment.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator


def evaluate_agent_pair(
    agent_0: Agent,
    agent_1: Agent,
    layout_name: str,
    num_games: int = 10,
    horizon: int = 400,
    display: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a pair of agents in the Overcooked environment.
    
    Args:
        agent_0: First agent
        agent_1: Second agent
        layout_name: Name of the layout
        num_games: Number of games to play
        horizon: Episode length
        display: Whether to display the game
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation results
    """
    # Create evaluator
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={"horizon": horizon}
    )
    
    # Create agent pair
    agent_pair = AgentPair(agent_0, agent_1)
    
    # Run evaluation
    results = ae.evaluate_agent_pair(
        agent_pair,
        num_games=num_games,
        display=display,
        info=verbose
    )
    
    # Compute summary statistics
    ep_returns = results["ep_returns"]
    
    summary = {
        "mean_return": np.mean(ep_returns),
        "std_return": np.std(ep_returns),
        "min_return": np.min(ep_returns),
        "max_return": np.max(ep_returns),
        "num_games": num_games,
        "ep_returns": ep_returns,
        "ep_lengths": results["ep_lengths"],
    }
    
    return summary


def evaluate_self_play(
    agent: Agent,
    layout_name: str,
    num_games: int = 10,
    horizon: int = 400,
    display: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate an agent playing with itself (self-play).
    
    Args:
        agent: The agent to evaluate
        layout_name: Name of the layout
        num_games: Number of games to play
        horizon: Episode length
        display: Whether to display the game
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation results
    """
    # Create evaluator
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={"horizon": horizon}
    )
    
    # Get featurize function for creating second agent
    # We need to create a copy to avoid state sharing issues
    import copy
    agent_0 = agent
    agent_1 = copy.deepcopy(agent)
    
    # Create agent pair with duplicate agents allowed
    agent_pair = AgentPair(agent_0, agent_1, allow_duplicate_agents=True)
    
    # Run evaluation
    results = ae.evaluate_agent_pair(
        agent_pair,
        num_games=num_games,
        display=display,
        info=verbose
    )
    
    # Compute summary statistics
    ep_returns = results["ep_returns"]
    
    summary = {
        "mean_return": np.mean(ep_returns),
        "std_return": np.std(ep_returns),
        "min_return": np.min(ep_returns),
        "max_return": np.max(ep_returns),
        "num_games": num_games,
        "ep_returns": ep_returns,
        "ep_lengths": results["ep_lengths"],
    }
    
    return summary


def load_and_evaluate(
    checkpoint_path: str,
    layout_name: str,
    agent_type: str = "jax",
    num_games: int = 10,
    horizon: int = 400,
    stochastic: bool = True,
    display: bool = False,
    verbose: bool = False,
    partner_checkpoint_path: Optional[str] = None,
    partner_agent_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load agents from checkpoints and evaluate them.
    
    Args:
        checkpoint_path: Path to the checkpoint for agent 0
        layout_name: Name of the layout
        agent_type: Type of agent ("jax" or "bc")
        num_games: Number of games to play
        horizon: Episode length
        stochastic: Whether to use stochastic action selection
        display: Whether to display the game
        verbose: Whether to print progress
        partner_checkpoint_path: Optional path to checkpoint for agent 1
        partner_agent_type: Type of partner agent (defaults to same as agent_type)
        
    Returns:
        Dictionary containing evaluation results
    """
    from overcooked_ai_py.agents.benchmarking import AgentEvaluator
    
    # Create evaluator for getting featurize function
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={"horizon": horizon}
    )
    
    # Load agent 0
    agent_0 = _load_agent(
        checkpoint_path=checkpoint_path,
        agent_type=agent_type,
        layout_name=layout_name,
        agent_index=0,
        stochastic=stochastic,
        ae=ae
    )
    
    # Load agent 1
    if partner_checkpoint_path:
        agent_1 = _load_agent(
            checkpoint_path=partner_checkpoint_path,
            agent_type=partner_agent_type or agent_type,
            layout_name=layout_name,
            agent_index=1,
            stochastic=stochastic,
            ae=ae
        )
    else:
        # Self-play with same checkpoint
        import copy
        agent_1 = copy.deepcopy(agent_0)
        agent_1.set_agent_index(1)
    
    # Evaluate
    return evaluate_agent_pair(
        agent_0=agent_0,
        agent_1=agent_1,
        layout_name=layout_name,
        num_games=num_games,
        horizon=horizon,
        display=display,
        verbose=verbose
    )


def _load_agent(
    checkpoint_path: str,
    agent_type: str,
    layout_name: str,
    agent_index: int,
    stochastic: bool,
    ae: AgentEvaluator
) -> Agent:
    """
    Load an agent from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        agent_type: Type of agent ("jax" or "bc")
        layout_name: Name of the layout
        agent_index: Agent index (0 or 1)
        stochastic: Whether to use stochastic action selection
        ae: AgentEvaluator for getting featurize function
        
    Returns:
        Loaded agent
    """
    if agent_type.lower() == "jax":
        from human_aware_rl.bridge.jax_agent import JaxPolicyAgent
        
        featurize_fn = lambda state: ae.env.lossless_state_encoding_mdp(state)
        
        return JaxPolicyAgent.from_checkpoint(
            checkpoint_dir=checkpoint_path,
            featurize_fn=featurize_fn,
            agent_index=agent_index,
            stochastic=stochastic,
            use_lossless_encoding=True
        )
    
    elif agent_type.lower() == "bc":
        from human_aware_rl.imitation.bc_agent import BCAgent
        from human_aware_rl.imitation.behavior_cloning import load_bc_model
        
        model, bc_params = load_bc_model(checkpoint_path)
        featurize_fn = lambda state: ae.env.featurize_state_mdp(state)
        
        return BCAgent(
            model=model,
            bc_params=bc_params,
            featurize_fn=featurize_fn,
            agent_index=agent_index,
            stochastic=stochastic
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_bc_vs_jax(
    bc_checkpoint: str,
    jax_checkpoint: str,
    layout_name: str,
    num_games: int = 10,
    horizon: int = 400,
    display: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a BC agent against a JAX-trained agent.
    
    Args:
        bc_checkpoint: Path to BC model checkpoint
        jax_checkpoint: Path to JAX model checkpoint
        layout_name: Name of the layout
        num_games: Number of games to play
        horizon: Episode length
        display: Whether to display the game
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation results
    """
    return load_and_evaluate(
        checkpoint_path=bc_checkpoint,
        layout_name=layout_name,
        agent_type="bc",
        num_games=num_games,
        horizon=horizon,
        display=display,
        verbose=verbose,
        partner_checkpoint_path=jax_checkpoint,
        partner_agent_type="jax"
    )


def run_evaluation_suite(
    checkpoint_path: str,
    agent_type: str,
    layouts: Optional[List[str]] = None,
    num_games: int = 10,
    horizon: int = 400,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run evaluation across multiple layouts.
    
    Args:
        checkpoint_path: Path to the agent checkpoint
        agent_type: Type of agent ("jax" or "bc")
        layouts: List of layouts to evaluate on (defaults to common layouts)
        num_games: Number of games per layout
        horizon: Episode length
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping layout names to evaluation results
    """
    if layouts is None:
        layouts = [
            "cramped_room",
            "asymmetric_advantages",
            "coordination_ring",
            "forced_coordination",
            "counter_circuit",
        ]
    
    results = {}
    
    for layout in layouts:
        if verbose:
            print(f"Evaluating on {layout}...")
        
        try:
            result = load_and_evaluate(
                checkpoint_path=checkpoint_path,
                layout_name=layout,
                agent_type=agent_type,
                num_games=num_games,
                horizon=horizon,
                verbose=verbose
            )
            results[layout] = result
            
            if verbose:
                print(f"  Mean return: {result['mean_return']:.2f} ± {result['std_return']:.2f}")
                
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            results[layout] = {"error": str(e)}
    
    return results

