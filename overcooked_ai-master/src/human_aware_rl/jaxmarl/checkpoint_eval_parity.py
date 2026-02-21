"""
Checkpoint-backed evaluator parity:
compare one saved JAX PPO checkpoint through two production evaluators.

Evaluator A: JAX network + OvercookedJaxEnv rollout (deterministic by default)
Evaluator B: AgentEvaluator + PPOAgentWrapper from evaluation/model_utils.py
"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import jax.numpy as jnp

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from human_aware_rl.evaluation import model_utils as mu
from human_aware_rl.jaxmarl.ppo import ActorCritic
from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig


def _load_checkpoint(checkpoint_dir: str) -> Tuple[dict, Any]:
    params_path = os.path.join(checkpoint_dir, "params.pkl")
    config_path = os.path.join(checkpoint_dir, "config.pkl")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing {params_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path}")
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return params, config


def _jax_eval_returns(
    params: dict,
    config: Any,
    num_games: int,
    base_seed: int,
    deterministic: bool,
) -> List[float]:
    p = params["params"] if "params" in params else params
    env_cfg = OvercookedJaxEnvConfig(
        layout_name=config.layout_name,
        horizon=getattr(config, "horizon", 400),
        old_dynamics=getattr(config, "old_dynamics", True),
        reward_shaping_factor=0.0,
        use_phi=False,
        use_legacy_encoding=getattr(config, "use_legacy_encoding", True),
    )
    env = OvercookedJaxEnv(env_cfg)
    net = ActorCritic(
        action_dim=6,
        hidden_dim=getattr(config, "hidden_dim", 64),
        num_hidden_layers=getattr(config, "num_hidden_layers", 3),
        num_filters=getattr(config, "num_filters", 25),
        num_conv_layers=getattr(config, "num_conv_layers", 3),
    )

    returns: List[float] = []
    for ep in range(num_games):
        np.random.seed(base_seed + ep)
        state, obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs0 = jnp.array(obs["agent_0"])[None]
            obs1 = jnp.array(obs["agent_1"])[None]
            logits0, _ = net.apply({"params": p} if "params" not in params else params, obs0)
            logits1, _ = net.apply({"params": p} if "params" not in params else params, obs1)
            if deterministic:
                a0 = int(jnp.argmax(logits0[0]))
                a1 = int(jnp.argmax(logits1[0]))
            else:
                probs0 = np.array(jnp.exp(logits0[0] - jnp.max(logits0[0])))
                probs0 = probs0 / probs0.sum()
                probs1 = np.array(jnp.exp(logits1[0] - jnp.max(logits1[0])))
                probs1 = probs1 / probs1.sum()
                a0 = int(np.random.choice(len(probs0), p=probs0))
                a1 = int(np.random.choice(len(probs1), p=probs1))
            state, obs, rewards, dones, _infos = env.step(state, {"agent_0": a0, "agent_1": a1})
            ep_ret += float(rewards["agent_0"])
            done = bool(dones["__all__"])
        returns.append(ep_ret)
    return returns


def _agent_evaluator_returns(
    params: dict,
    config: Any,
    layout_name_for_wrapper: str,
    num_games: int,
    base_seed: int,
    deterministic: bool,
) -> List[float]:
    env_layout = config.layout_name
    if env_layout == "random0_legacy":
        layout_name_for_wrapper = "forced_coordination"
    elif env_layout == "random3_legacy":
        layout_name_for_wrapper = "counter_circuit"
    elif env_layout == "coordination_ring_legacy":
        layout_name_for_wrapper = "coordination_ring"
    elif env_layout == "asymmetric_advantages_legacy":
        layout_name_for_wrapper = "asymmetric_advantages"
    elif env_layout == "cramped_room_legacy":
        layout_name_for_wrapper = "cramped_room"

    np.random.seed(base_seed)
    agent_a = mu.PPOAgentWrapper(params=params, config=config, layout_name=layout_name_for_wrapper, stochastic=not deterministic)
    agent_b = mu.PPOAgentWrapper(params=params, config=config, layout_name=layout_name_for_wrapper, stochastic=not deterministic)
    agent_a.set_agent_index(0)
    agent_b.set_agent_index(1)

    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": env_layout, "old_dynamics": getattr(config, "old_dynamics", True)},
        env_params={"horizon": getattr(config, "horizon", 400)},
    )
    result = ae.evaluate_agent_pair(AgentPair(agent_a, agent_b), num_games=num_games, display=False)
    ep_returns = result["ep_returns"]
    return [float(x) for x in ep_returns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action sampling")
    parser.add_argument("--out-json", type=str, required=True)
    args = parser.parse_args()

    params, config = _load_checkpoint(args.checkpoint_dir)
    deterministic = not args.stochastic

    ret_jax = _jax_eval_returns(
        params=params,
        config=config,
        num_games=args.num_games,
        base_seed=args.seed,
        deterministic=deterministic,
    )
    ret_agent_eval = _agent_evaluator_returns(
        params=params,
        config=config,
        layout_name_for_wrapper=getattr(config, "layout_name", "coordination_ring"),
        num_games=args.num_games,
        base_seed=args.seed,
        deterministic=deterministic,
    )

    diffs = [abs(a - b) for a, b in zip(ret_jax, ret_agent_eval)]
    report: Dict[str, Any] = {
        "checkpoint_dir": args.checkpoint_dir,
        "num_games": args.num_games,
        "seed": args.seed,
        "deterministic": deterministic,
        "layout_name": getattr(config, "layout_name", None),
        "jax_eval_returns": ret_jax,
        "agent_evaluator_returns": ret_agent_eval,
        "episode_abs_diffs": diffs,
        "mean_jax": float(np.mean(ret_jax)),
        "mean_agent_eval": float(np.mean(ret_agent_eval)),
        "mean_abs_diff": float(np.mean(diffs)),
        "max_abs_diff": float(np.max(diffs)) if diffs else 0.0,
        "passed_exact": all(np.isclose(a, b, atol=1e-6) for a, b in zip(ret_jax, ret_agent_eval)),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Wrote checkpoint parity report:", args.out_json)
    print("mean_jax:", report["mean_jax"])
    print("mean_agent_eval:", report["mean_agent_eval"])
    print("mean_abs_diff:", report["mean_abs_diff"])
    print("max_abs_diff:", report["max_abs_diff"])
    print("passed_exact:", report["passed_exact"])


if __name__ == "__main__":
    main()
