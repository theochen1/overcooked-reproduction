"""
Helpers for deprecated-style PPO run directory layout.

Canonical path shape:
  DATA_DIR/ppo_runs/<run_name>/seed<seed>/<agent_name>/checkpoint_*
"""

import os
from datetime import datetime
from typing import Optional

from human_aware_rl.data_dir import DATA_DIR


def default_ppo_data_dir() -> str:
    """Return canonical PPO data directory."""
    return os.path.join(DATA_DIR, "ppo_runs")


def build_run_name(ex_name: str, timestamp_dir: bool) -> str:
    """Build a deprecated-style run name."""
    if not timestamp_dir:
        return ex_name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{ex_name}"


def format_run_template(run_name_template: Optional[str], layout: str) -> Optional[str]:
    """Format run-name templates that may include {layout}."""
    if run_name_template is None:
        return None
    if "{layout}" in run_name_template:
        return run_name_template.format(layout=layout)
    return run_name_template


def build_training_output_paths(
    *,
    ppo_data_dir: str,
    run_name: str,
    seed: int,
    agent_name: str,
) -> dict:
    """
    Build training output paths for PPOTrainer.

    Returns dict with:
      - run_dir
      - seed_dir
      - trainer_results_dir
      - trainer_experiment_name
      - agent_dir
    """
    run_dir = os.path.join(ppo_data_dir, run_name)
    seed_dir = os.path.join(run_dir, f"seed{seed}")
    agent_dir = os.path.join(seed_dir, agent_name)
    return {
        "run_dir": run_dir,
        "seed_dir": seed_dir,
        "trainer_results_dir": seed_dir,
        "trainer_experiment_name": agent_name,
        "agent_dir": agent_dir,
    }
