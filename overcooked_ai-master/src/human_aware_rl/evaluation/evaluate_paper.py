"""
Paper-style Evaluation Script for Overcooked AI.

This script reproduces the evaluation methodology from Figure 4 of:
"On the Utility of Learning about Humans for Human-AI Coordination"

Figure 4(a) - Self-Play Comparison:
- PPO_HP + HP (red dotted line - gold standard)
- SP + SP (white bars - self-play baseline)
- SP + HP (teal bars - self-play with human proxy)
- PPO_BC + HP (orange bars - BC-trained PPO with human proxy)
- BC + HP (gray bars - BC baseline)
- Switched indices shown with hatching

Figure 4(b) - PBT Comparison:
- PPO_HP + HP (red dotted line - gold standard)
- PBT + PBT (white bars - PBT self-play baseline)
- PBT + HP (teal bars - PBT with human proxy)
- PPO_BC + HP (orange bars - same as 4a)
- BC + HP (gray bars - same as 4a)

Usage:
    python -m human_aware_rl.evaluation.evaluate_paper \\
        --ppo_data_dir /path/to/DATA_DIR/ppo_runs \\
        --run_name_sp "ppo_sp__layout-{layout}" \\
        --run_name_bc "ppo_bc__partner-bc_train__layout-{layout}" \\
        --run_name_hp "ppo_hp__layout-{layout}" \\
        --agent_dir_sp ppo_agent \\
        --agent_dir_bc ppo_bc_agent \\
        --agent_dir_hp ppo_hp_agent \\
        --output_file paper_results.json

By default this uses strict canonical run-registry loading from DATA_DIR/ppo_runs.
Legacy results/* scanning is available only when --paper_strict is disabled.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.ppo.configs.paper_configs import PAPER_LAYOUTS, LAYOUT_TO_ENV
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR
from human_aware_rl.ppo.run_paths import default_ppo_data_dir, format_run_template
from human_aware_rl.ppo.run_registry_defaults import (
    DEFAULT_AGENT_DIRS,
    DEFAULT_RUN_NAME_TEMPLATES,
)
from human_aware_rl.evaluation.evaluate_all import (
    load_bc_agent,
    load_jax_agent,
    evaluate_agent_pair,
)


# =============================================================================
# Figure 4(a) Evaluation Configs - Self-Play Comparison
# =============================================================================
FIGURE_4A_CONFIGS = {
    # Gold standard (red dotted line)
    "ppo_hp_hp": {
        "description": "PPO trained with HP + HP (Gold Standard)",
        "display_name": "PPO_HProxy+HProxy",
        "agent_0_type": "ppo",
        "agent_1_type": "bc",
        "agent_0_source": "ppo_hp",
        "agent_1_source": "hp",
        "agent_1_source_split": "test",
        "color": "red",
        "style": "dotted_line",
    },
    # Self-play baseline (white bars)
    "sp_sp": {
        "description": "PPO Self-Play + PPO Self-Play",
        "display_name": "SP+SP",
        "agent_0_type": "ppo",
        "agent_1_type": "ppo",
        "agent_0_source": "ppo_sp",
        "agent_1_source": "ppo_sp",
        "color": "white",
        "style": "bar_dotted_border",
    },
    # Self-play with human proxy (teal bars)
    "sp_hp": {
        "description": "PPO Self-Play + Human Proxy",
        "display_name": "SP+HProxy",
        "agent_0_type": "ppo",
        "agent_1_type": "bc",
        "agent_0_source": "ppo_sp",
        "agent_1_source": "hp",
        "agent_1_source_split": "test",
        "color": "#2d6777",  # Teal
        "style": "bar",
    },
    # PPO_BC with human proxy (orange bars)
    "ppo_bc_hp": {
        "description": "PPO_BC + Human Proxy",
        "display_name": "PPO_BC+HProxy",
        "agent_0_type": "ppo",
        "agent_1_type": "bc",
        "agent_0_source": "ppo_bc",
        "agent_1_source": "hp",
        "agent_1_source_split": "test",
        "color": "#F79646",  # Orange
        "style": "bar",
    },
    # BC baseline (gray bars)
    "bc_hp": {
        "description": "BC + Human Proxy",
        "display_name": "BC+HProxy",
        "agent_0_type": "bc",
        "agent_1_type": "bc",
        "agent_0_source": "bc",
        "agent_1_source": "hp",
        "agent_0_source_split": "train",
        "agent_1_source_split": "test",
        "color": "#7f7f7f",  # Gray
        "style": "bar",
    },
}

# =============================================================================
# Figure 4(b) Evaluation Configs - PBT Comparison
# =============================================================================
FIGURE_4B_CONFIGS = {
    # Gold standard (red dotted line) - same as 4a
    "ppo_hp_hp": FIGURE_4A_CONFIGS["ppo_hp_hp"],
    # PBT self-play baseline (white bars)
    "pbt_pbt": {
        "description": "PBT + PBT",
        "display_name": "PBT+PBT",
        "agent_0_type": "ppo",
        "agent_1_type": "ppo",
        "agent_0_source": "pbt",
        "agent_1_source": "pbt",
        "color": "white",
        "style": "bar_dotted_border",
    },
    # PBT with human proxy (teal bars)
    "pbt_hp": {
        "description": "PBT + Human Proxy",
        "display_name": "PBT+HProxy",
        "agent_0_type": "ppo",
        "agent_1_type": "bc",
        "agent_0_source": "pbt",
        "agent_1_source": "hp",
        "color": "#2d6777",  # Teal
        "style": "bar",
    },
    # PPO_BC with human proxy (orange bars) - same as 4a
    "ppo_bc_hp": FIGURE_4A_CONFIGS["ppo_bc_hp"],
    # BC baseline (gray bars) - same as 4a
    "bc_hp": FIGURE_4A_CONFIGS["bc_hp"],
}

# =============================================================================
# GAIL Comparison Evaluation Configs
# =============================================================================
# These configs compare PPO_GAIL (controlled and optimized) against PPO_BC
# to isolate the partner model as the experimental variable.
GAIL_COMPARISON_CONFIGS = {
    # PPO_BC baseline (same as Figure 4a orange bar)
    "ppo_bc_hp": FIGURE_4A_CONFIGS["ppo_bc_hp"],
    # PPO_GAIL_controlled + Human Proxy (fair comparison: Table 3 HPs, GAIL partner)
    "ppo_gail_hp": {
        "description": "PPO_GAIL (controlled) + Human Proxy",
        "display_name": "PPO_GAIL+HProxy",
        "agent_0_type": "ppo",
        "agent_1_type": "bc",
        "agent_0_source": "ppo_gail",
        "agent_1_source": "hp",
        "agent_1_source_split": "test",
        "color": "#9467BD",  # Purple
        "style": "bar",
    },
    # PPO_GAIL_optimized + Human Proxy (ablation: Bayesian HPs)
    "ppo_gail_opt_hp": {
        "description": "PPO_GAIL (optimized) + Human Proxy",
        "display_name": "PPO_GAIL_opt+HProxy",
        "agent_0_type": "ppo",
        "agent_1_type": "bc",
        "agent_0_source": "ppo_gail_opt",
        "agent_1_source": "hp",
        "agent_1_source_split": "test",
        "color": "#D62728",  # Dark red
        "style": "bar",
    },
    # PPO_SP_optimized + Human Proxy (ablation: isolate HP effect from GAIL)
    "ppo_sp_opt_hp": {
        "description": "PPO_SP (optimized) + Human Proxy",
        "display_name": "PPO_SP_opt+HProxy",
        "agent_0_type": "ppo",
        "agent_1_type": "bc",
        "agent_0_source": "ppo_sp_opt",
        "agent_1_source": "hp",
        "agent_1_source_split": "test",
        "color": "#17BECF",  # Cyan
        "style": "bar",
    },
    # Gold standard for reference
    "ppo_hp_hp": FIGURE_4A_CONFIGS["ppo_hp_hp"],
    # BC baseline for reference
    "bc_hp": FIGURE_4A_CONFIGS["bc_hp"],
}

# Combined configs for full evaluation
ALL_EVALUATION_CONFIGS = {
    **FIGURE_4A_CONFIGS,
    "pbt_pbt": FIGURE_4B_CONFIGS["pbt_pbt"],
    "pbt_hp": FIGURE_4B_CONFIGS["pbt_hp"],
    **{k: v for k, v in GAIL_COMPARISON_CONFIGS.items()
       if k not in FIGURE_4A_CONFIGS and k not in FIGURE_4B_CONFIGS},
}


def find_checkpoint(base_dir: str, layout: str, seed: Optional[int] = None) -> Optional[str]:
    """
    Find the latest checkpoint for a layout.
    
    Args:
        base_dir: Base directory for experiments
        layout: Layout name
        seed: Optional seed to look for
        
    Returns:
        Path to checkpoint or None
    """
    if not os.path.exists(base_dir):
        return None
    
    for exp_name in os.listdir(base_dir):
        if layout in exp_name:
            if seed is not None and f"seed{seed}" not in exp_name:
                continue
            
            exp_dir = os.path.join(base_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue
                
            checkpoints = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint")]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                return os.path.join(exp_dir, latest)
    
    return None


def find_checkpoint_from_run(
    ppo_data_dir: str,
    run_name_template: str,
    layout: str,
    seed: int,
    agent_name: str,
) -> Optional[str]:
    """Resolve checkpoint from canonical ppo_runs run/seed/agent structure."""
    run_name = format_run_template(run_name_template, layout)
    agent_dir = os.path.join(
        ppo_data_dir,
        run_name,
        f"seed{seed}",
        agent_name,
    )
    if not os.path.isdir(agent_dir):
        return None
    checkpoints = [d for d in os.listdir(agent_dir) if d.startswith("checkpoint")]
    if not checkpoints:
        return None
    return os.path.join(agent_dir, sorted(checkpoints)[-1])


def expected_agent_dir_from_run(
    ppo_data_dir: str,
    run_name_template: str,
    layout: str,
    seed: int,
    agent_name: str,
) -> str:
    """Return canonical run-registry agent directory for a layout/seed."""
    run_name = format_run_template(run_name_template, layout)
    return os.path.join(ppo_data_dir, run_name, f"seed{seed}", agent_name)


def resolve_bc_root(source_split: Optional[str], bc_dir: str, hp_dir: str) -> str:
    """Resolve BC-model root from split metadata."""
    if source_split == "train":
        return bc_dir
    if source_split == "test":
        return hp_dir
    return bc_dir


def evaluate_paper_config(
    config_name: str,
    layout: str,
    ppo_sp_dir: Optional[str],
    ppo_bc_dir: Optional[str],
    ppo_hp_dir: Optional[str] = "results/ppo_hp",
    pbt_dir: Optional[str] = "results/pbt",
    ppo_gail_dir: str = "results/ppo_gail",
    ppo_gail_opt_dir: str = "results/ppo_gail_opt",
    ppo_sp_opt_dir: str = "results/ppo_sp_opt",
    bc_dir: Optional[str] = None,
    hp_dir: Optional[str] = None,
    num_games: int = 10,
    seed: Optional[int] = None,
    agent_order: int = 0,  # 0 = normal order, 1 = swapped
    ppo_data_dir: Optional[str] = None,
    run_name_templates: Optional[Dict[str, str]] = None,
    agent_dirs: Optional[Dict[str, str]] = None,
    prefer_run_registry: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a single paper configuration.
    
    Args:
        config_name: Name of evaluation config
        layout: Paper layout name
        ppo_sp_dir: Directory with PPO self-play checkpoints (non-strict fallback only)
        ppo_bc_dir: Directory with PPO_BC checkpoints (non-strict fallback only)
        ppo_hp_dir: Directory with PPO_HP checkpoints (non-strict fallback only)
        pbt_dir: Directory with PBT checkpoints (non-strict fallback only)
        ppo_gail_dir: Directory with PPO_GAIL (controlled) checkpoints
        ppo_gail_opt_dir: Directory with PPO_GAIL (optimized) checkpoints
        ppo_sp_opt_dir: Directory with PPO_SP (optimized) checkpoints
        bc_dir: Directory with BC models (default: BC_SAVE_DIR/train)
        hp_dir: Directory with Human Proxy models (default: BC_SAVE_DIR/test)
        num_games: Number of games to play
        seed: Random seed to evaluate (None = any)
        agent_order: 0 for normal order, 1 for swapped
        
    Returns:
        Evaluation results
    """
    config = ALL_EVALUATION_CONFIGS[config_name]
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    
    # Set default directories
    if bc_dir is None:
        bc_dir = os.path.join(BC_SAVE_DIR, "train")
    if hp_dir is None:
        hp_dir = os.path.join(BC_SAVE_DIR, "test")
    if ppo_data_dir is None:
        ppo_data_dir = default_ppo_data_dir()
    if run_name_templates is None:
        run_name_templates = DEFAULT_RUN_NAME_TEMPLATES
    if agent_dirs is None:
        agent_dirs = DEFAULT_AGENT_DIRS
    
    # Determine agent indices based on order
    if agent_order == 0:
        idx_0, idx_1 = 0, 1
    else:
        idx_0, idx_1 = 1, 0
    
    # Load agents
    def load_agent(
        agent_type: str,
        source: str,
        agent_index: int,
        source_split: Optional[str],
        agent_slot: int,
        strict: bool,
    ):
        run_template = run_name_templates.get(source)
        run_agent_name = agent_dirs.get(source)
        if source == "pbt":
            run_template = run_template or "pbt_{layout}"
            run_agent_name = run_agent_name or "pbt_agent"
        registry_sources = {"ppo_sp", "ppo_bc", "ppo_hp", "pbt"}
        checkpoint = None
        if strict and seed is None:
            raise ValueError(
                "Strict paper evaluation requires an explicit seed so checkpoint "
                "resolution is deterministic."
            )
        if strict and source in registry_sources:
            if not prefer_run_registry:
                raise ValueError(
                    "Strict paper evaluation requires run-registry loading; "
                    f"got prefer_run_registry=False for source={source}."
                )
            if not run_template or not run_agent_name:
                raise ValueError(
                    "Strict paper evaluation requires run template and agent directory "
                    f"for source={source}."
                )
            checkpoint = find_checkpoint_from_run(
                ppo_data_dir=ppo_data_dir,
                run_name_template=run_template,
                layout=layout,
                seed=seed,
                agent_name=run_agent_name,
            )
            if checkpoint is None:
                expected_agent_dir = expected_agent_dir_from_run(
                    ppo_data_dir=ppo_data_dir,
                    run_name_template=run_template,
                    layout=layout,
                    seed=seed,
                    agent_name=run_agent_name,
                )
                raise FileNotFoundError(
                    "Strict paper evaluation could not resolve checkpoint from run registry "
                    f"for source={source}. Expected checkpoint under: {expected_agent_dir}"
                )
        elif prefer_run_registry and run_template and run_agent_name and seed is not None:
            checkpoint = find_checkpoint_from_run(
                ppo_data_dir=ppo_data_dir,
                run_name_template=run_template,
                layout=layout,
                seed=seed,
                agent_name=run_agent_name,
            )

        def _fallback_checkpoint(base_dir: Optional[str]) -> Optional[str]:
            if strict:
                return None
            if not base_dir:
                return None
            return find_checkpoint(base_dir, layout, seed)

        if source == "ppo_sp":
            if checkpoint is None:
                checkpoint = _fallback_checkpoint(ppo_sp_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"No PPO_SP checkpoint found for {layout}")
            return load_jax_agent(checkpoint, env_layout, agent_index)
        
        elif source == "ppo_bc":
            if checkpoint is None:
                checkpoint = _fallback_checkpoint(ppo_bc_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"No PPO_BC checkpoint found for {layout}")
            return load_jax_agent(checkpoint, env_layout, agent_index)
        
        elif source == "ppo_hp":
            if checkpoint is None:
                checkpoint = _fallback_checkpoint(ppo_hp_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"No PPO_HP checkpoint found for {layout}")
            return load_jax_agent(checkpoint, env_layout, agent_index)
        
        elif source == "pbt":
            if checkpoint is None:
                checkpoint = _fallback_checkpoint(pbt_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"No PBT checkpoint found for {layout}")
            return load_jax_agent(checkpoint, env_layout, agent_index)
        
        elif source == "ppo_gail":
            checkpoint = find_checkpoint(ppo_gail_dir, layout, seed)
            if checkpoint is None:
                raise FileNotFoundError(f"No PPO_GAIL (controlled) checkpoint found for {layout}")
            return load_jax_agent(checkpoint, env_layout, agent_index)
        
        elif source == "ppo_gail_opt":
            checkpoint = find_checkpoint(ppo_gail_opt_dir, layout, seed)
            if checkpoint is None:
                raise FileNotFoundError(f"No PPO_GAIL (optimized) checkpoint found for {layout}")
            return load_jax_agent(checkpoint, env_layout, agent_index)
        
        elif source == "ppo_sp_opt":
            checkpoint = find_checkpoint(ppo_sp_opt_dir, layout, seed)
            if checkpoint is None:
                raise FileNotFoundError(f"No PPO_SP (optimized) checkpoint found for {layout}")
            return load_jax_agent(checkpoint, env_layout, agent_index)
        
        elif source == "bc":
            bc_root = resolve_bc_root(source_split, bc_dir, hp_dir)
            bc_path = os.path.join(bc_root, layout)
            if not os.path.exists(bc_path):
                raise FileNotFoundError(f"No BC model found at {bc_path}")
            return load_bc_agent(bc_path, env_layout, agent_index)
        
        elif source == "hp":
            split = source_split or "test"
            if split == "train":
                raise ValueError(
                    f"Invalid split for HP source on agent_{agent_slot}: "
                    "source='hp' requires test split in this codebase."
                )
            hp_root = resolve_bc_root(split, bc_dir, hp_dir)
            hp_path = os.path.join(hp_root, layout)
            if not os.path.exists(hp_path):
                raise FileNotFoundError(f"No HP model found at {hp_path}")
            return load_bc_agent(hp_path, env_layout, agent_index)
        
        else:
            raise ValueError(f"Unknown source: {source}")
    
    agent_0 = load_agent(
        config["agent_0_type"],
        config["agent_0_source"],
        idx_0,
        config.get("agent_0_source_split"),
        0,
        strict,
    )
    agent_1 = load_agent(
        config["agent_1_type"],
        config["agent_1_source"],
        idx_1,
        config.get("agent_1_source_split"),
        1,
        strict,
    )
    
    return evaluate_agent_pair(agent_0, agent_1, env_layout, num_games)


def evaluate_figure_4a(
    ppo_sp_dir: Optional[str],
    ppo_bc_dir: Optional[str],
    ppo_hp_dir: Optional[str] = "results/ppo_hp",
    bc_dir: Optional[str] = None,
    hp_dir: Optional[str] = None,
    layouts: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_games: int = 10,
    verbose: bool = True,
    ppo_data_dir: Optional[str] = None,
    run_name_templates: Optional[Dict[str, str]] = None,
    agent_dirs: Optional[Dict[str, str]] = None,
    prefer_run_registry: bool = True,
    strict: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run Figure 4(a) evaluations (Self-Play comparison).
    
    Args:
        ppo_sp_dir: Directory with PPO self-play checkpoints
        ppo_bc_dir: Directory with PPO_BC checkpoints
        ppo_hp_dir: Directory with PPO_HP checkpoints
        bc_dir: Directory with BC models
        hp_dir: Directory with Human Proxy models
        layouts: Layouts to evaluate (default: all)
        seeds: Seeds to average over (default: [0])
        num_games: Number of games per evaluation
        verbose: Whether to print progress
        
    Returns:
        Nested dict: {layout: {config_name: {order: results}}}
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    if seeds is None:
        seeds = [0]
    
    configs_to_run = ["ppo_hp_hp", "sp_sp", "sp_hp", "ppo_bc_hp", "bc_hp"]
    
    return _run_evaluations(
        configs=configs_to_run,
        ppo_sp_dir=ppo_sp_dir,
        ppo_bc_dir=ppo_bc_dir,
        ppo_hp_dir=ppo_hp_dir,
        pbt_dir="",  # Not used for 4a
        bc_dir=bc_dir,
        hp_dir=hp_dir,
        layouts=layouts,
        seeds=seeds,
        num_games=num_games,
        verbose=verbose,
        figure_name="Figure 4(a) - Self-Play Comparison",
        ppo_data_dir=ppo_data_dir,
        run_name_templates=run_name_templates,
        agent_dirs=agent_dirs,
        prefer_run_registry=prefer_run_registry,
        strict=strict,
    )


def evaluate_figure_4b(
    ppo_bc_dir: Optional[str],
    ppo_hp_dir: Optional[str] = "results/ppo_hp",
    pbt_dir: Optional[str] = "results/pbt",
    bc_dir: Optional[str] = None,
    hp_dir: Optional[str] = None,
    layouts: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_games: int = 10,
    verbose: bool = True,
    ppo_data_dir: Optional[str] = None,
    run_name_templates: Optional[Dict[str, str]] = None,
    agent_dirs: Optional[Dict[str, str]] = None,
    prefer_run_registry: bool = True,
    strict: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run Figure 4(b) evaluations (PBT comparison).
    
    Args:
        ppo_bc_dir: Directory with PPO_BC checkpoints
        ppo_hp_dir: Directory with PPO_HP checkpoints
        pbt_dir: Directory with PBT checkpoints
        bc_dir: Directory with BC models
        hp_dir: Directory with Human Proxy models
        layouts: Layouts to evaluate (default: all)
        seeds: Seeds to average over (default: [0])
        num_games: Number of games per evaluation
        verbose: Whether to print progress
        
    Returns:
        Nested dict: {layout: {config_name: {order: results}}}
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    if seeds is None:
        seeds = [0]
    
    configs_to_run = ["ppo_hp_hp", "pbt_pbt", "pbt_hp", "ppo_bc_hp", "bc_hp"]
    
    return _run_evaluations(
        configs=configs_to_run,
        ppo_sp_dir="",  # Not used for 4b
        ppo_bc_dir=ppo_bc_dir,
        ppo_hp_dir=ppo_hp_dir,
        pbt_dir=pbt_dir,
        bc_dir=bc_dir,
        hp_dir=hp_dir,
        layouts=layouts,
        seeds=seeds,
        num_games=num_games,
        verbose=verbose,
        figure_name="Figure 4(b) - PBT Comparison",
        ppo_data_dir=ppo_data_dir,
        run_name_templates=run_name_templates,
        agent_dirs=agent_dirs,
        prefer_run_registry=prefer_run_registry,
        strict=strict,
    )


def evaluate_gail_comparison(
    ppo_bc_dir: str,
    ppo_gail_dir: str = "results/ppo_gail",
    ppo_gail_opt_dir: str = "results/ppo_gail_opt",
    ppo_sp_opt_dir: str = "results/ppo_sp_opt",
    ppo_hp_dir: str = "results/ppo_hp",
    bc_dir: Optional[str] = None,
    hp_dir: Optional[str] = None,
    layouts: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_games: int = 50,
    verbose: bool = True,
    ppo_data_dir: Optional[str] = None,
    run_name_templates: Optional[Dict[str, str]] = None,
    agent_dirs: Optional[Dict[str, str]] = None,
    prefer_run_registry: bool = True,
    strict: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run GAIL comparison evaluations.

    Compares PPO_GAIL_controlled vs PPO_BC (fair comparison using Table 3 HPs),
    with optional ablations (PPO_GAIL_optimized, PPO_SP_optimized).

    All conditions are evaluated identically: agent paired with Human Proxy model,
    5 seeds, 50+ games, both agent orders.

    Args:
        ppo_bc_dir: Directory with PPO_BC checkpoints (baseline)
        ppo_gail_dir: Directory with PPO_GAIL (controlled) checkpoints
        ppo_gail_opt_dir: Directory with PPO_GAIL (optimized) checkpoints
        ppo_sp_opt_dir: Directory with PPO_SP (optimized) checkpoints
        ppo_hp_dir: Directory with PPO_HP checkpoints (gold standard)
        bc_dir: Directory with BC models
        hp_dir: Directory with Human Proxy models
        layouts: Layouts to evaluate (default: all)
        seeds: Seeds to average over (default: [0,10,20,30,40])
        num_games: Number of games per evaluation (default: 50)
        verbose: Whether to print progress

    Returns:
        Nested dict: {layout: {config_name: {order: results}}}
    """
    if layouts is None:
        layouts = PAPER_LAYOUTS
    if seeds is None:
        seeds = [0, 10, 20, 30, 40]

    configs_to_run = [
        "ppo_hp_hp",        # Gold standard reference
        "ppo_bc_hp",        # PPO_BC baseline (Table 3 HPs, BC partner)
        "ppo_gail_hp",      # PPO_GAIL controlled (Table 3 HPs, GAIL partner)
        "ppo_gail_opt_hp",  # PPO_GAIL optimized (Bayesian HPs, GAIL partner)
        "ppo_sp_opt_hp",    # PPO_SP optimized (Bayesian HPs, no partner)
        "bc_hp",            # BC baseline reference
    ]

    return _run_evaluations(
        configs=configs_to_run,
        ppo_sp_dir="",  # Not directly used
        ppo_bc_dir=ppo_bc_dir,
        ppo_hp_dir=ppo_hp_dir,
        pbt_dir="",  # Not used
        bc_dir=bc_dir,
        hp_dir=hp_dir,
        layouts=layouts,
        seeds=seeds,
        num_games=num_games,
        verbose=verbose,
        figure_name="GAIL Comparison - Partner Model Ablation",
        ppo_gail_dir=ppo_gail_dir,
        ppo_gail_opt_dir=ppo_gail_opt_dir,
        ppo_sp_opt_dir=ppo_sp_opt_dir,
        ppo_data_dir=ppo_data_dir,
        run_name_templates=run_name_templates,
        agent_dirs=agent_dirs,
        prefer_run_registry=prefer_run_registry,
        strict=strict,
    )


def _run_evaluations(
    configs: List[str],
    ppo_sp_dir: Optional[str],
    ppo_bc_dir: Optional[str],
    ppo_hp_dir: Optional[str],
    pbt_dir: Optional[str],
    bc_dir: Optional[str],
    hp_dir: Optional[str],
    layouts: List[str],
    seeds: List[int],
    num_games: int,
    verbose: bool,
    figure_name: str,
    ppo_gail_dir: str = "results/ppo_gail",
    ppo_gail_opt_dir: str = "results/ppo_gail_opt",
    ppo_sp_opt_dir: str = "results/ppo_sp_opt",
    ppo_data_dir: Optional[str] = None,
    run_name_templates: Optional[Dict[str, str]] = None,
    agent_dirs: Optional[Dict[str, str]] = None,
    prefer_run_registry: bool = True,
    strict: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Internal helper to run evaluations."""
    all_results = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(figure_name)
        print(f"{'='*60}")
    
    for layout in layouts:
        if verbose:
            print(f"\nEvaluating {layout}")
            print("-" * 40)
        
        layout_results = {}
        
        for config_name in configs:
            config_results = {"order_0": [], "order_1": []}
            
            for seed in seeds:
                for order in [0, 1]:
                    try:
                        result = evaluate_paper_config(
                            config_name=config_name,
                            layout=layout,
                            ppo_sp_dir=ppo_sp_dir,
                            ppo_bc_dir=ppo_bc_dir,
                            ppo_hp_dir=ppo_hp_dir,
                            pbt_dir=pbt_dir,
                            ppo_gail_dir=ppo_gail_dir,
                            ppo_gail_opt_dir=ppo_gail_opt_dir,
                            ppo_sp_opt_dir=ppo_sp_opt_dir,
                            bc_dir=bc_dir,
                            hp_dir=hp_dir,
                            num_games=num_games,
                            seed=seed,
                            agent_order=order,
                            ppo_data_dir=ppo_data_dir,
                            run_name_templates=run_name_templates,
                            agent_dirs=agent_dirs,
                            prefer_run_registry=prefer_run_registry,
                            strict=strict,
                        )
                        config_results[f"order_{order}"].append(result)
                        
                    except Exception as e:
                        if verbose:
                            print(f"  {config_name} (order={order}, seed={seed}): Error - {e}")
                        config_results[f"order_{order}"].append({"error": str(e)})
            
            # Aggregate results
            for order_key in ["order_0", "order_1"]:
                valid_results = [r for r in config_results[order_key] if "error" not in r]
                if valid_results:
                    all_rewards = []
                    for r in valid_results:
                        all_rewards.extend(r["ep_returns"])
                    
                    config_results[order_key] = {
                        "mean_reward": float(np.mean(all_rewards)),
                        "std_reward": float(np.std(all_rewards)),
                        "stderr_reward": float(np.std(all_rewards) / np.sqrt(len(all_rewards))),
                        "num_games": len(all_rewards),
                        "num_seeds": len(valid_results),
                    }
                elif config_results[order_key]:
                    config_results[order_key] = config_results[order_key][0]
            
            layout_results[config_name] = config_results
            
            if verbose and "error" not in config_results.get("order_0", {}):
                display_name = ALL_EVALUATION_CONFIGS[config_name]["display_name"]
                mean_0 = config_results["order_0"].get("mean_reward", 0)
                stderr_0 = config_results["order_0"].get("stderr_reward", 0)
                mean_1 = config_results["order_1"].get("mean_reward", 0)
                stderr_1 = config_results["order_1"].get("stderr_reward", 0)
                print(f"  {display_name}: {mean_0:.1f}±{stderr_0:.1f} | switched: {mean_1:.1f}±{stderr_1:.1f}")
        
        all_results[layout] = layout_results
    
    return all_results


def evaluate_all_paper_experiments(
    ppo_sp_dir: Optional[str] = "results/ppo_sp",
    ppo_bc_dir: Optional[str] = "results/ppo_bc",
    ppo_hp_dir: Optional[str] = "results/ppo_hp",
    pbt_dir: Optional[str] = "results/pbt",
    ppo_gail_dir: str = "results/ppo_gail",
    ppo_gail_opt_dir: str = "results/ppo_gail_opt",
    ppo_sp_opt_dir: str = "results/ppo_sp_opt",
    bc_dir: Optional[str] = None,
    hp_dir: Optional[str] = None,
    layouts: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_games: int = 10,
    verbose: bool = True,
    ppo_data_dir: Optional[str] = None,
    run_name_templates: Optional[Dict[str, str]] = None,
    agent_dirs: Optional[Dict[str, str]] = None,
    prefer_run_registry: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Run all paper evaluations for Figure 4(a), 4(b), and GAIL comparison.
    
    Returns:
        Dictionary with 'figure_4a', 'figure_4b', and 'gail_comparison' results
    """
    results = {}
    
    # Figure 4(a) - Self-Play comparison
    results["figure_4a"] = evaluate_figure_4a(
        ppo_sp_dir=ppo_sp_dir,
        ppo_bc_dir=ppo_bc_dir,
        ppo_hp_dir=ppo_hp_dir,
        bc_dir=bc_dir,
        hp_dir=hp_dir,
        layouts=layouts,
        seeds=seeds,
        num_games=num_games,
        verbose=verbose,
        ppo_data_dir=ppo_data_dir,
        run_name_templates=run_name_templates,
        agent_dirs=agent_dirs,
        prefer_run_registry=prefer_run_registry,
        strict=strict,
    )
    
    # Figure 4(b) - PBT comparison
    results["figure_4b"] = evaluate_figure_4b(
        ppo_bc_dir=ppo_bc_dir,
        ppo_hp_dir=ppo_hp_dir,
        pbt_dir=pbt_dir,
        bc_dir=bc_dir,
        hp_dir=hp_dir,
        layouts=layouts,
        seeds=seeds,
        num_games=num_games,
        verbose=verbose,
        ppo_data_dir=ppo_data_dir,
        run_name_templates=run_name_templates,
        agent_dirs=agent_dirs,
        prefer_run_registry=prefer_run_registry,
        strict=strict,
    )
    
    # GAIL comparison (fair partner-model ablation)
    results["gail_comparison"] = evaluate_gail_comparison(
        ppo_bc_dir=ppo_bc_dir,
        ppo_gail_dir=ppo_gail_dir,
        ppo_gail_opt_dir=ppo_gail_opt_dir,
        ppo_sp_opt_dir=ppo_sp_opt_dir,
        ppo_hp_dir=ppo_hp_dir,
        bc_dir=bc_dir,
        hp_dir=hp_dir,
        layouts=layouts,
        seeds=seeds,
        num_games=50,  # 50+ games for conference-grade evaluation
        verbose=verbose,
        ppo_data_dir=ppo_data_dir,
        run_name_templates=run_name_templates,
        agent_dirs=agent_dirs,
        prefer_run_registry=prefer_run_registry,
        strict=strict,
    )
    
    # Add config metadata for plotting
    results["configs"] = {
        "figure_4a": {k: {kk: vv for kk, vv in v.items() if kk in ["display_name", "color", "style"]}
                      for k, v in FIGURE_4A_CONFIGS.items()},
        "figure_4b": {k: {kk: vv for kk, vv in v.items() if kk in ["display_name", "color", "style"]}
                      for k, v in FIGURE_4B_CONFIGS.items()},
        "gail_comparison": {k: {kk: vv for kk, vv in v.items() if kk in ["display_name", "color", "style"]}
                            for k, v in GAIL_COMPARISON_CONFIGS.items()},
    }
    
    return results


def print_paper_table(results: Dict[str, Any]):
    """Print results in paper table format."""
    print("\n" + "="*100)
    print("PAPER RESULTS - Figure 4(a) Self-Play Comparison")
    print("="*100)
    
    if "figure_4a" in results:
        _print_figure_table(results["figure_4a"], FIGURE_4A_CONFIGS)
    
    print("\n" + "="*100)
    print("PAPER RESULTS - Figure 4(b) PBT Comparison")
    print("="*100)
    
    if "figure_4b" in results:
        _print_figure_table(results["figure_4b"], FIGURE_4B_CONFIGS)
    
    if "gail_comparison" in results:
        print("\n" + "="*100)
        print("GAIL COMPARISON - Partner Model Ablation")
        print("="*100)
        print("PPO_BC: Paper Table 3 HPs, BC partner (baseline)")
        print("PPO_GAIL: Paper Table 3 HPs, GAIL partner (controlled)")
        print("PPO_GAIL_opt: Bayesian HPs, GAIL partner (optimized ablation)")
        print("PPO_SP_opt: Bayesian HPs, no partner (HP ablation)")
        print()
        _print_figure_table(results["gail_comparison"], GAIL_COMPARISON_CONFIGS)


def _print_figure_table(figure_results: Dict, configs: Dict):
    """Print table for a single figure."""
    configs_list = list(configs.keys())
    
    # Header
    header = f"{'Layout':<20}" + "".join(f"{configs[c]['display_name']:<18}" for c in configs_list)
    print(header)
    print("-"*100)
    
    for layout in figure_results:
        row = f"{layout:<20}"
        for config in configs_list:
            if config in figure_results[layout]:
                cfg_results = figure_results[layout][config]
                if "error" not in cfg_results.get("order_0", {}):
                    mean = cfg_results["order_0"]["mean_reward"]
                    stderr = cfg_results["order_0"]["stderr_reward"]
                    row += f"{mean:.1f}±{stderr:.1f}".ljust(18)
                else:
                    row += f"{'Error':<18}"
            else:
                row += f"{'-':<18}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agents using paper methodology (Figure 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--ppo_sp_dir",
        type=str,
        default="results/ppo_sp",
        help="Directory with PPO self-play checkpoints"
    )
    
    parser.add_argument(
        "--ppo_bc_dir",
        type=str,
        default="results/ppo_bc",
        help="Directory with PPO_BC checkpoints"
    )
    
    parser.add_argument(
        "--ppo_hp_dir",
        type=str,
        default="results/ppo_hp",
        help="Directory with PPO_HP (gold standard) checkpoints"
    )
    
    parser.add_argument(
        "--pbt_dir",
        type=str,
        default="results/pbt",
        help="Directory with PBT checkpoints"
    )
    
    parser.add_argument(
        "--ppo_gail_dir",
        type=str,
        default="results/ppo_gail",
        help="Directory with PPO_GAIL (controlled) checkpoints"
    )
    
    parser.add_argument(
        "--ppo_gail_opt_dir",
        type=str,
        default="results/ppo_gail_opt",
        help="Directory with PPO_GAIL (optimized) checkpoints"
    )
    
    parser.add_argument(
        "--ppo_sp_opt_dir",
        type=str,
        default="results/ppo_sp_opt",
        help="Directory with PPO_SP (optimized) checkpoints"
    )
    
    parser.add_argument(
        "--bc_dir",
        type=str,
        default=None,
        help="Directory with BC models (default: BC_SAVE_DIR/train)"
    )
    
    parser.add_argument(
        "--hp_dir",
        type=str,
        default=None,
        help="Directory with Human Proxy models (default: BC_SAVE_DIR/test)"
    )
    parser.add_argument(
        "--ppo_data_dir",
        type=str,
        default=default_ppo_data_dir(),
        help="Canonical PPO run directory (DATA_DIR/ppo_runs)"
    )
    parser.add_argument(
        "--run_name_sp",
        type=str,
        default=DEFAULT_RUN_NAME_TEMPLATES["ppo_sp"],
        help="Run-name template for PPO_SP (supports {layout})"
    )
    parser.add_argument(
        "--run_name_bc",
        type=str,
        default=DEFAULT_RUN_NAME_TEMPLATES["ppo_bc"],
        help="Run-name template for PPO_BC (supports {layout})"
    )
    parser.add_argument(
        "--run_name_hp",
        type=str,
        default=DEFAULT_RUN_NAME_TEMPLATES["ppo_hp"],
        help="Run-name template for PPO_HP (supports {layout})"
    )
    parser.add_argument(
        "--agent_dir_sp",
        type=str,
        default=DEFAULT_AGENT_DIRS["ppo_sp"],
        help="Agent subdirectory name for PPO_SP runs"
    )
    parser.add_argument(
        "--agent_dir_bc",
        type=str,
        default=DEFAULT_AGENT_DIRS["ppo_bc"],
        help="Agent subdirectory name for PPO_BC runs"
    )
    parser.add_argument(
        "--agent_dir_hp",
        type=str,
        default=DEFAULT_AGENT_DIRS["ppo_hp"],
        help="Agent subdirectory name for PPO_HP runs"
    )
    parser.add_argument(
        "--run_name_pbt",
        type=str,
        default=DEFAULT_RUN_NAME_TEMPLATES["pbt"],
        help="Run-name template for PBT (supports {layout})"
    )
    parser.add_argument(
        "--agent_dir_pbt",
        type=str,
        default=DEFAULT_AGENT_DIRS["pbt"],
        help="Agent subdirectory name for PBT runs"
    )
    parser.add_argument(
        "--disable_run_registry",
        action="store_true",
        help="Disable deterministic run_name/seed loading and use legacy directory scans first"
    )
    parser.add_argument(
        "--strict",
        dest="paper_strict",
        action="store_true",
        default=True,
        help="Require strict run-registry resolution for paper checkpoints"
    )
    parser.add_argument(
        "--no_strict",
        dest="paper_strict",
        action="store_false",
        help="Allow legacy fallback checkpoint scans"
    )
    parser.add_argument(
        "--paper_strict",
        dest="paper_strict",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_paper_strict",
        dest="paper_strict",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="paper_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=["4a", "4b", "gail", "all"],
        help="Which evaluation to run (4a, 4b, gail, or all)"
    )
    
    parser.add_argument(
        "--layouts",
        type=str,
        default=None,
        help="Comma-separated list of layouts (default: all)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,10,20,30,40",
        help="Comma-separated list of seeds to evaluate"
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
    seeds = [int(s) for s in args.seeds.split(",")]
    verbose = not args.quiet
    run_name_templates = {
        "ppo_sp": args.run_name_sp,
        "ppo_bc": args.run_name_bc,
        "ppo_hp": args.run_name_hp,
        "pbt": args.run_name_pbt,
    }
    agent_dirs = {
        "ppo_sp": args.agent_dir_sp,
        "ppo_bc": args.agent_dir_bc,
        "ppo_hp": args.agent_dir_hp,
        "pbt": args.agent_dir_pbt,
    }
    prefer_run_registry = not args.disable_run_registry
    if args.paper_strict and not prefer_run_registry:
        parser.error("--paper_strict requires run-registry loading; remove --disable_run_registry.")
    if args.paper_strict:
        if args.ppo_sp_dir != "results/ppo_sp":
            parser.error("ppo_sp_dir override is disallowed in strict mode.")
        if args.ppo_bc_dir != "results/ppo_bc":
            parser.error("ppo_bc_dir override is disallowed in strict mode.")
        if args.ppo_hp_dir != "results/ppo_hp":
            parser.error("ppo_hp_dir override is disallowed in strict mode.")
        if args.pbt_dir != "results/pbt":
            parser.error("pbt_dir override is disallowed in strict mode.")
    
    if args.figure == "all":
        results = evaluate_all_paper_experiments(
            ppo_sp_dir=None if args.paper_strict else args.ppo_sp_dir,
            ppo_bc_dir=None if args.paper_strict else args.ppo_bc_dir,
            ppo_hp_dir=None if args.paper_strict else args.ppo_hp_dir,
            pbt_dir=None if args.paper_strict else args.pbt_dir,
            ppo_gail_dir=args.ppo_gail_dir,
            ppo_gail_opt_dir=args.ppo_gail_opt_dir,
            ppo_sp_opt_dir=args.ppo_sp_opt_dir,
            bc_dir=args.bc_dir,
            hp_dir=args.hp_dir,
            layouts=layouts,
            seeds=seeds,
            num_games=args.num_games,
            verbose=verbose,
            ppo_data_dir=args.ppo_data_dir,
            run_name_templates=run_name_templates,
            agent_dirs=agent_dirs,
            prefer_run_registry=prefer_run_registry,
            strict=args.paper_strict,
        )
    elif args.figure == "4a":
        results = {
            "figure_4a": evaluate_figure_4a(
                ppo_sp_dir=None if args.paper_strict else args.ppo_sp_dir,
                ppo_bc_dir=None if args.paper_strict else args.ppo_bc_dir,
                ppo_hp_dir=None if args.paper_strict else args.ppo_hp_dir,
                bc_dir=args.bc_dir,
                hp_dir=args.hp_dir,
                layouts=layouts,
                seeds=seeds,
                num_games=args.num_games,
                verbose=verbose,
                ppo_data_dir=args.ppo_data_dir,
                run_name_templates=run_name_templates,
                agent_dirs=agent_dirs,
                prefer_run_registry=prefer_run_registry,
                strict=args.paper_strict,
            )
        }
    elif args.figure == "4b":
        results = {
            "figure_4b": evaluate_figure_4b(
                ppo_bc_dir=None if args.paper_strict else args.ppo_bc_dir,
                ppo_hp_dir=None if args.paper_strict else args.ppo_hp_dir,
                pbt_dir=None if args.paper_strict else args.pbt_dir,
                bc_dir=args.bc_dir,
                hp_dir=args.hp_dir,
                layouts=layouts,
                seeds=seeds,
                num_games=args.num_games,
                verbose=verbose,
                ppo_data_dir=args.ppo_data_dir,
                run_name_templates=run_name_templates,
                agent_dirs=agent_dirs,
                prefer_run_registry=prefer_run_registry,
                strict=args.paper_strict,
            )
        }
    elif args.figure == "gail":
        results = {
            "gail_comparison": evaluate_gail_comparison(
                ppo_bc_dir=args.ppo_bc_dir,
                ppo_gail_dir=args.ppo_gail_dir,
                ppo_gail_opt_dir=args.ppo_gail_opt_dir,
                ppo_sp_opt_dir=args.ppo_sp_opt_dir,
                ppo_hp_dir=args.ppo_hp_dir,
                bc_dir=args.bc_dir,
                hp_dir=args.hp_dir,
                layouts=layouts,
                seeds=seeds,
                num_games=args.num_games if args.num_games != 10 else 50,  # Default 50 for GAIL
                verbose=verbose,
                ppo_data_dir=args.ppo_data_dir,
                run_name_templates=run_name_templates,
                agent_dirs=agent_dirs,
                prefer_run_registry=prefer_run_registry,
                strict=args.paper_strict,
            )
        }
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")
    
    # Print table
    print_paper_table(results)


if __name__ == "__main__":
    main()
