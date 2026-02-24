"""Analysis helpers for paper reproduction (Figures 5-7)."""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.imitation.bc_agent import BCAgent
from human_aware_rl.imitation.behavior_cloning import load_bc_model
from human_aware_rl.ppo.configs.paper_configs import LAYOUT_TO_ENV
from human_aware_rl.utils import accuracy, cross_entropy
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action


@dataclass
class ModelSpec:
    """Specifies a model to score against held-out trajectories."""

    model_type: str  # "bc" | "ppo"
    name: str
    path: str


def _collect_probs_for_traj(agent, observations, actions, agent_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collect policy probabilities and integer labels for a single trajectory."""
    agent.reset()
    agent.set_agent_index(agent_idx)
    probs: List[np.ndarray] = []
    labels: List[int] = []
    for obs, action in zip(observations, actions):
        _, info = agent.action(obs)
        probs.append(np.asarray(info["action_probs"], dtype=np.float64))
        labels.append(Action.ACTION_TO_INDEX[action])
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def get_trajs_losses_for_model(trajs: Dict, agent, eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Port of deprecated losses_experiments.get_trajs_losses_for_model."""
    losses: List[float] = []
    accuracies: List[float] = []
    for obs_seq, act_seq, idx_seq in zip(
        trajs["ep_observations"],
        trajs["ep_actions"],
        trajs["ep_agent_idxs"],
    ):
        # idx_seq is scalar or len-T list depending on source; keep first index.
        if isinstance(idx_seq, (list, tuple, np.ndarray)):
            agent_idx = int(idx_seq[0])
        else:
            agent_idx = int(idx_seq)
        probs, labels = _collect_probs_for_traj(agent, obs_seq, act_seq, agent_idx)
        losses.append(float(cross_entropy(probs, labels, eps=eps)))
        accuracies.append(float(accuracy(probs, labels)))
    return np.asarray(losses), np.asarray(accuracies)


def _make_bc_agent(model_dir: str, env_layout: str) -> BCAgent:
    model, bc_params = load_bc_model(model_dir)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    featurize_fn = lambda state: ae.env.featurize_state_mdp(state)
    return BCAgent(
        model=model,
        bc_params=bc_params,
        featurize_fn=featurize_fn,
        agent_index=0,
        stochastic=False,
    )


def _make_jax_agent(checkpoint_dir: str, env_layout: str):
    from human_aware_rl.bridge.jax_agent import JaxPolicyAgent

    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    featurize_fn = lambda state: ae.env.lossless_state_encoding_mdp(state)
    return JaxPolicyAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        featurize_fn=featurize_fn,
        agent_index=0,
        stochastic=False,
        use_lossless_encoding=True,
    )


def evaluate_layout_loss_for_bc_models(
    model_specs: Iterable[ModelSpec],
    layout: str,
    trajs: Dict,
    eps: float = 1e-4,
) -> Dict[str, Dict[str, float]]:
    """Evaluate BC models on held-out trajectories."""
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    out: Dict[str, Dict[str, float]] = {}
    for spec in model_specs:
        if spec.model_type != "bc":
            continue
        agent = _make_bc_agent(spec.path, env_layout)
        losses, accs = get_trajs_losses_for_model(trajs, agent, eps=eps)
        out[spec.name] = {
            "loss_mean": float(np.mean(losses)),
            "loss_stderr": float(np.std(losses) / np.sqrt(max(1, len(losses)))),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_stderr": float(np.std(accs) / np.sqrt(max(1, len(accs)))),
        }
    return out


def evaluate_layout_loss_for_ppo_models(
    model_specs: Iterable[ModelSpec],
    layout: str,
    trajs: Dict,
    eps: float = 1e-4,
) -> Dict[str, Dict[str, float]]:
    """Evaluate JAX PPO checkpoints on held-out trajectories."""
    env_layout = LAYOUT_TO_ENV.get(layout, layout)
    out: Dict[str, Dict[str, float]] = {}
    for spec in model_specs:
        if spec.model_type != "ppo":
            continue
        agent = _make_jax_agent(spec.path, env_layout)
        losses, accs = get_trajs_losses_for_model(trajs, agent, eps=eps)
        out[spec.name] = {
            "loss_mean": float(np.mean(losses)),
            "loss_stderr": float(np.std(losses) / np.sqrt(max(1, len(losses)))),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_stderr": float(np.std(accs) / np.sqrt(max(1, len(accs)))),
        }
    return out


def compute_off_distribution_metrics(
    checkpoint_dirs: Dict[str, Dict[str, str]],
    human_data_path: Optional[str] = None,
    layouts: Optional[List[str]] = None,
    eps: float = 1e-4,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute off-distribution loss/accuracy metrics for Figures 6/7.

    Args:
        checkpoint_dirs: mapping `{layout: {model_name: path}}`
          model_name prefixes determine type:
          - `bc_*` => BC model dir
          - `ppo_*` => PPO checkpoint dir
        human_data_path: optional override for human trajectory pickle
        layouts: optional subset of layouts
        eps: cross-entropy floor
    """
    target_layouts = layouts or sorted(checkpoint_dirs.keys())
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for layout in target_layouts:
        if layout not in checkpoint_dirs:
            continue
        trajs = get_human_human_trajectories(
            layouts=[layout],
            dataset_type="test",
            data_path=human_data_path,
            featurize_states=False,
            check_trajectories=False,
            silent=True,
        )
        specs: List[ModelSpec] = []
        for model_name, path in checkpoint_dirs[layout].items():
            model_type = "ppo" if model_name.startswith("ppo_") else "bc"
            specs.append(ModelSpec(model_type=model_type, name=model_name, path=path))

        layout_results: Dict[str, Dict[str, float]] = {}
        layout_results.update(evaluate_layout_loss_for_bc_models(specs, layout, trajs, eps=eps))
        layout_results.update(evaluate_layout_loss_for_ppo_models(specs, layout, trajs, eps=eps))
        results[layout] = layout_results
    return results


def save_off_distribution_metrics(metrics: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(metrics, f)

