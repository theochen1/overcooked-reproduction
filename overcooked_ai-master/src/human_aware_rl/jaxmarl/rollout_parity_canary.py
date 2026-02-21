"""
Rollout parity canaries for TF-style vs JAX Overcooked rollouts.

This script focuses on rollout-level drift (env + observation semantics), which
can still exist even when PPO minibatch math is already matched.

Canaries:
1) action_replay: replay the same joint action sequence in both paths
2) step_hash: run a fixed deterministic policy and compare step traces
3) encoding_swap: validate observation channel/swap semantics over time
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from human_aware_rl.jaxmarl.overcooked_env import OvercookedJaxEnv, OvercookedJaxEnvConfig


LAYOUT_ALIASES = {
    "forced_coordination": "random0_legacy",
    "counter_circuit": "random3_legacy",
}


@dataclass
class CanaryResult:
    name: str
    passed: bool
    first_divergence_step: Optional[int]
    first_divergence_field: Optional[str]
    message: str
    details: Dict[str, Any]


def _canonicalize_layout(layout_name: str) -> str:
    return LAYOUT_ALIASES.get(layout_name, layout_name)


def _hash_array(arr: np.ndarray) -> str:
    # Hash semantic observation content, not backend-specific dtype/storage.
    arr_np = np.asarray(arr, dtype=np.float32)
    payload = arr_np.tobytes() + str(arr_np.shape).encode("utf-8") + str(arr_np.dtype).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _state_hash(state: Any) -> str:
    state_dict = _to_jsonable(state.to_dict())
    serialized = json.dumps(state_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _encode_obs(base_env: OvercookedEnv, state: Any, use_legacy_encoding: bool) -> Tuple[np.ndarray, np.ndarray]:
    if use_legacy_encoding:
        obs_pair = base_env.lossless_state_encoding_mdp_legacy(state)
    else:
        obs_pair = base_env.lossless_state_encoding_mdp(state)
    return np.asarray(obs_pair[0]), np.asarray(obs_pair[1])


def _map_obs_by_agent_idx(obs_p0: np.ndarray, obs_p1: np.ndarray, agent_idx: int) -> Dict[str, np.ndarray]:
    if agent_idx == 0:
        return {"agent_0": obs_p0, "agent_1": obs_p1}
    return {"agent_0": obs_p1, "agent_1": obs_p0}


class ReferenceEnv:
    """
    TF-style reference path built directly on OvercookedEnv.

    It mirrors the JAX wrapper's action/observation swapping behavior so that
    any differences should be attributable to wrapper logic drift.
    """

    def __init__(
        self,
        layout_name: str,
        old_dynamics: bool,
        horizon: int,
        reward_shaping_factor: float,
        rew_shaping_params: Dict[str, float],
        use_phi: bool,
        use_legacy_encoding: bool,
        fixed_agent_idx: Optional[int],
    ):
        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name,
            rew_shaping_params=rew_shaping_params,
            old_dynamics=old_dynamics,
        )
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon, info_level=0)
        self.reward_shaping_factor = reward_shaping_factor
        self.use_phi = use_phi
        self.use_legacy_encoding = use_legacy_encoding
        self.fixed_agent_idx = fixed_agent_idx
        self.agent_idx = 0

    def reset(self, seed: int) -> Dict[str, np.ndarray]:
        np.random.seed(seed)
        self.base_env.reset()
        self.agent_idx = self.fixed_agent_idx if self.fixed_agent_idx is not None else int(np.random.choice([0, 1]))
        obs_p0, obs_p1 = _encode_obs(self.base_env, self.base_env.state, self.use_legacy_encoding)
        return _map_obs_by_agent_idx(obs_p0, obs_p1, self.agent_idx)

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, float, float, bool, Dict[str, Any]]:
        a0 = int(actions["agent_0"])
        a1 = int(actions["agent_1"])

        if self.agent_idx == 0:
            joint_action = (Action.INDEX_TO_ACTION[a0], Action.INDEX_TO_ACTION[a1])
        else:
            joint_action = (Action.INDEX_TO_ACTION[a1], Action.INDEX_TO_ACTION[a0])

        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=True)
            dense_total = float(info.get("phi_s_prime", 0.0) - info.get("phi_s", 0.0))
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)
            shaped_by_agent = info.get("shaped_r_by_agent", (0.0, 0.0))
            dense_total = float(np.sum(shaped_by_agent))

        shaped_total = float(sparse_reward + self.reward_shaping_factor * dense_total)
        obs_p0, obs_p1 = _encode_obs(self.base_env, next_state, self.use_legacy_encoding)
        obs = _map_obs_by_agent_idx(obs_p0, obs_p1, self.agent_idx)
        return obs, float(sparse_reward), dense_total, shaped_total, bool(done), info


class JaxPathEnv:
    def __init__(
        self,
        layout_name: str,
        old_dynamics: bool,
        horizon: int,
        reward_shaping_factor: float,
        rew_shaping_params: Dict[str, float],
        use_phi: bool,
        use_legacy_encoding: bool,
        fixed_agent_idx: Optional[int],
    ):
        self.config = OvercookedJaxEnvConfig(
            layout_name=layout_name,
            old_dynamics=old_dynamics,
            horizon=horizon,
            rew_shaping_params=rew_shaping_params,
            reward_shaping_factor=reward_shaping_factor,
            use_phi=use_phi,
            use_legacy_encoding=use_legacy_encoding,
        )
        self.env = OvercookedJaxEnv(self.config)
        self.state = None
        self.fixed_agent_idx = fixed_agent_idx

    def reset(self, seed: int) -> Dict[str, np.ndarray]:
        np.random.seed(seed)
        self.state, obs = self.env.reset()
        if self.fixed_agent_idx is not None:
            self.env.agent_idx = int(self.fixed_agent_idx)
            obs_p0, obs_p1 = _encode_obs(self.env.base_env, self.env.base_env.state, self.config.use_legacy_encoding)
            obs = _map_obs_by_agent_idx(obs_p0, obs_p1, self.env.agent_idx)
        return {k: np.asarray(v) for k, v in obs.items()}

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, float, float, bool, Dict[str, Any]]:
        self.state, obs, rewards, dones, infos = self.env.step(self.state, actions)
        sparse_reward = float(infos.get("sparse_reward", 0.0))
        if self.config.use_phi:
            info_agent = infos.get("agent_0", {})
            dense_total = float(info_agent.get("phi_s_prime", 0.0) - info_agent.get("phi_s", 0.0))
        else:
            info_agent = infos.get("agent_0", {})
            shaped_by_agent = info_agent.get("shaped_r_by_agent", (0.0, 0.0))
            dense_total = float(np.sum(shaped_by_agent))
        shaped_total = float(rewards["agent_0"])
        done = bool(dones["__all__"])
        return {k: np.asarray(v) for k, v in obs.items()}, sparse_reward, dense_total, shaped_total, done, infos


def _generate_action_sequence(seed: int, steps: int, num_actions: int = 6) -> List[Dict[str, int]]:
    rng = np.random.RandomState(seed)
    seq: List[Dict[str, int]] = []
    for _ in range(steps):
        seq.append(
            {
                "agent_0": int(rng.randint(0, num_actions)),
                "agent_1": int(rng.randint(0, num_actions)),
            }
        )
    return seq


def _build_linear_policy(seed: int, obs_shape: Tuple[int, ...], num_actions: int = 6):
    rng = np.random.RandomState(seed)
    flat_dim = int(np.prod(obs_shape))
    w = rng.randn(flat_dim, num_actions).astype(np.float32) / np.sqrt(max(1, flat_dim))
    b = rng.randn(num_actions).astype(np.float32) * 0.01

    def policy(obs: np.ndarray) -> int:
        flat = obs.astype(np.float32).reshape(-1)
        logits = flat @ w + b
        return int(np.argmax(logits))

    return policy


def _first_obs_divergence(obs_ref: Dict[str, np.ndarray], obs_jax: Dict[str, np.ndarray]) -> Optional[str]:
    for agent in ("agent_0", "agent_1"):
        if obs_ref[agent].shape != obs_jax[agent].shape:
            return f"{agent}.shape"
        if not np.array_equal(obs_ref[agent], obs_jax[agent]):
            return f"{agent}.values"
    return None


def run_action_replay(args) -> CanaryResult:
    ref_env = ReferenceEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )
    jax_env = JaxPathEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )

    obs_ref = ref_env.reset(seed=args.seed)
    obs_jax = jax_env.reset(seed=args.seed)
    field = _first_obs_divergence(obs_ref, obs_jax)
    if field is not None:
        return CanaryResult(
            name="action_replay",
            passed=False,
            first_divergence_step=0,
            first_divergence_field=field,
            message="Initial observations diverged",
            details={},
        )

    action_seq = _generate_action_sequence(seed=args.seed + 11, steps=args.steps)
    trace = []
    for t, actions in enumerate(action_seq, start=1):
        obs_ref, sparse_ref, dense_ref, shaped_ref, done_ref, _ = ref_env.step(actions)
        obs_jax, sparse_jax, dense_jax, shaped_jax, done_jax, _ = jax_env.step(actions)

        rec = {
            "t": t,
            "state_hash_ref": _state_hash(ref_env.base_env.state),
            "state_hash_jax": _state_hash(jax_env.env.base_env.state),
            "obs_hash_ref_agent_0": _hash_array(obs_ref["agent_0"]),
            "obs_hash_jax_agent_0": _hash_array(obs_jax["agent_0"]),
            "sparse_ref": sparse_ref,
            "sparse_jax": sparse_jax,
            "dense_ref": dense_ref,
            "dense_jax": dense_jax,
            "shaped_ref": shaped_ref,
            "shaped_jax": shaped_jax,
            "done_ref": done_ref,
            "done_jax": done_jax,
        }
        trace.append(rec)

        checks = [
            ("state_hash", rec["state_hash_ref"] == rec["state_hash_jax"]),
            ("obs_hash_agent_0", rec["obs_hash_ref_agent_0"] == rec["obs_hash_jax_agent_0"]),
            ("sparse_reward", np.isclose(sparse_ref, sparse_jax, atol=1e-6)),
            ("dense_reward", np.isclose(dense_ref, dense_jax, atol=1e-6)),
            ("shaped_reward", np.isclose(shaped_ref, shaped_jax, atol=1e-6)),
            ("done", done_ref == done_jax),
        ]
        for name, ok in checks:
            if not ok:
                return CanaryResult(
                    name="action_replay",
                    passed=False,
                    first_divergence_step=t,
                    first_divergence_field=name,
                    message=f"First divergence at step {t}: {name}",
                    details={"last_record": rec},
                )
        if done_ref or done_jax:
            break

    return CanaryResult(
        name="action_replay",
        passed=True,
        first_divergence_step=None,
        first_divergence_field=None,
        message="No divergence found in replayed rollout",
        details={"steps_checked": len(trace)},
    )


def run_step_hash(args) -> CanaryResult:
    ref_env = ReferenceEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )
    jax_env = JaxPathEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )

    obs_ref = ref_env.reset(seed=args.seed)
    obs_jax = jax_env.reset(seed=args.seed)
    field = _first_obs_divergence(obs_ref, obs_jax)
    if field is not None:
        return CanaryResult(
            name="step_hash",
            passed=False,
            first_divergence_step=0,
            first_divergence_field=field,
            message="Initial observations diverged",
            details={},
        )

    policy_0 = _build_linear_policy(seed=args.seed + 101, obs_shape=obs_ref["agent_0"].shape)
    policy_1 = _build_linear_policy(seed=args.seed + 202, obs_shape=obs_ref["agent_1"].shape)

    for t in range(1, args.steps + 1):
        action_ref = {
            "agent_0": policy_0(obs_ref["agent_0"]),
            "agent_1": policy_1(obs_ref["agent_1"]),
        }
        action_jax = {
            "agent_0": policy_0(obs_jax["agent_0"]),
            "agent_1": policy_1(obs_jax["agent_1"]),
        }
        if action_ref != action_jax:
            return CanaryResult(
                name="step_hash",
                passed=False,
                first_divergence_step=t,
                first_divergence_field="policy_action",
                message=f"Policy selected different actions at step {t}",
                details={"action_ref": action_ref, "action_jax": action_jax},
            )

        obs_ref, sparse_ref, dense_ref, shaped_ref, done_ref, _ = ref_env.step(action_ref)
        obs_jax, sparse_jax, dense_jax, shaped_jax, done_jax, _ = jax_env.step(action_jax)

        checks = [
            ("state_hash", _state_hash(ref_env.base_env.state) == _state_hash(jax_env.env.base_env.state)),
            ("obs_agent_0", np.array_equal(obs_ref["agent_0"], obs_jax["agent_0"])),
            ("obs_agent_1", np.array_equal(obs_ref["agent_1"], obs_jax["agent_1"])),
            ("sparse_reward", np.isclose(sparse_ref, sparse_jax, atol=1e-6)),
            ("dense_reward", np.isclose(dense_ref, dense_jax, atol=1e-6)),
            ("shaped_reward", np.isclose(shaped_ref, shaped_jax, atol=1e-6)),
            ("done", done_ref == done_jax),
        ]
        for name, ok in checks:
            if not ok:
                return CanaryResult(
                    name="step_hash",
                    passed=False,
                    first_divergence_step=t,
                    first_divergence_field=name,
                    message=f"First divergence at step {t}: {name}",
                    details={},
                )
        if done_ref or done_jax:
            break

    return CanaryResult(
        name="step_hash",
        passed=True,
        first_divergence_step=None,
        first_divergence_field=None,
        message="No divergence found with fixed deterministic policy",
        details={"steps_checked": args.steps},
    )


def _channel_identity_checks(
    obs: Dict[str, np.ndarray],
    state: Any,
    agent_idx: int,
) -> Optional[str]:
    # For legacy lossless encoding:
    # - channel 0 is "self player location"
    # - channel 1 is "other player location"
    p_self = state.players[agent_idx].position
    p_other = state.players[1 - agent_idx].position

    ch0 = obs["agent_0"][..., 0]
    ch1 = obs["agent_0"][..., 1]
    if ch0[p_self] != 1 or ch0.sum() != 1:
        return "agent_0.self_channel"
    if ch1[p_other] != 1 or ch1.sum() != 1:
        return "agent_0.other_channel"

    p_self_1 = state.players[1 - agent_idx].position
    p_other_1 = state.players[agent_idx].position
    ch0_1 = obs["agent_1"][..., 0]
    ch1_1 = obs["agent_1"][..., 1]
    if ch0_1[p_self_1] != 1 or ch0_1.sum() != 1:
        return "agent_1.self_channel"
    if ch1_1[p_other_1] != 1 or ch1_1.sum() != 1:
        return "agent_1.other_channel"
    return None


def run_encoding_swap(args) -> CanaryResult:
    if not args.use_legacy_encoding:
        return CanaryResult(
            name="encoding_swap",
            passed=False,
            first_divergence_step=0,
            first_divergence_field="config",
            message="encoding_swap currently expects --use-legacy-encoding",
            details={},
        )

    ref_env = ReferenceEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )
    jax_env = JaxPathEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )

    obs_ref = ref_env.reset(seed=args.seed)
    obs_jax = jax_env.reset(seed=args.seed)
    identity_err_ref = _channel_identity_checks(obs_ref, ref_env.base_env.state, ref_env.agent_idx)
    identity_err_jax = _channel_identity_checks(obs_jax, jax_env.env.base_env.state, jax_env.env.agent_idx)
    if identity_err_ref is not None or identity_err_jax is not None:
        return CanaryResult(
            name="encoding_swap",
            passed=False,
            first_divergence_step=0,
            first_divergence_field=identity_err_ref or identity_err_jax,
            message="Identity channels invalid at reset",
            details={"ref_agent_idx": ref_env.agent_idx, "jax_agent_idx": jax_env.env.agent_idx},
        )

    action_seq = _generate_action_sequence(seed=args.seed + 33, steps=args.steps)
    for t, actions in enumerate(action_seq, start=1):
        obs_ref, _, _, _, done_ref, _ = ref_env.step(actions)
        obs_jax, _, _, _, done_jax, _ = jax_env.step(actions)

        mismatch_field = _first_obs_divergence(obs_ref, obs_jax)
        if mismatch_field is not None:
            return CanaryResult(
                name="encoding_swap",
                passed=False,
                first_divergence_step=t,
                first_divergence_field=mismatch_field,
                message=f"Observation tensor mismatch at step {t}",
                details={},
            )

        identity_err_ref = _channel_identity_checks(obs_ref, ref_env.base_env.state, ref_env.agent_idx)
        identity_err_jax = _channel_identity_checks(obs_jax, jax_env.env.base_env.state, jax_env.env.agent_idx)
        if identity_err_ref is not None or identity_err_jax is not None:
            return CanaryResult(
                name="encoding_swap",
                passed=False,
                first_divergence_step=t,
                first_divergence_field=identity_err_ref or identity_err_jax,
                message=f"Identity-channel mismatch at step {t}",
                details={},
            )
        if done_ref or done_jax:
            break

    return CanaryResult(
        name="encoding_swap",
        passed=True,
        first_divergence_step=None,
        first_divergence_field=None,
        message="Observation swap/channel semantics remained aligned",
        details={"steps_checked": args.steps},
    )


def _build_linear_value_fn(seed: int, obs_shape: Tuple[int, ...]):
    rng = np.random.RandomState(seed)
    flat_dim = int(np.prod(obs_shape))
    w = rng.randn(flat_dim).astype(np.float32) / np.sqrt(max(1, flat_dim))
    b = np.float32(rng.randn() * 0.01)

    def value_fn(obs: np.ndarray) -> float:
        flat = obs.astype(np.float32).reshape(-1)
        return float(flat @ w + b)

    return value_fn


def _compute_gae_with_masks(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_mask: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        next_value = float(last_value) if t == n - 1 else float(values[t + 1])
        nonterminal = float(bootstrap_mask[t])
        delta = float(rewards[t]) + gamma * next_value * nonterminal - float(values[t])
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        adv[t] = np.float32(last_gae)
    returns = adv + values.astype(np.float32)
    return adv, returns


def _first_array_divergence(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> Optional[int]:
    if a.shape != b.shape:
        return 0
    diff = np.where(~np.isclose(a, b, atol=atol))[0]
    if diff.size == 0:
        return None
    return int(diff[0])


def run_training_buffer(args) -> CanaryResult:
    ref_env = ReferenceEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )
    jax_env = JaxPathEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=args.reward_shaping_factor,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=args.use_phi,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )

    obs_ref = ref_env.reset(seed=args.seed)
    obs_jax = jax_env.reset(seed=args.seed)
    obs_field = _first_obs_divergence(obs_ref, obs_jax)
    if obs_field is not None:
        return CanaryResult(
            name="training_buffer",
            passed=False,
            first_divergence_step=0,
            first_divergence_field=obs_field,
            message="Initial observations diverged",
            details={},
        )

    policy_0 = _build_linear_policy(seed=args.seed + 401, obs_shape=obs_ref["agent_0"].shape)
    policy_1 = _build_linear_policy(seed=args.seed + 402, obs_shape=obs_ref["agent_1"].shape)
    value_fn = _build_linear_value_fn(seed=args.seed + 499, obs_shape=obs_ref["agent_0"].shape)

    fields_ref: Dict[str, List[float]] = {
        "value_preds": [],
        "rewards": [],
        "dones": [],
        "episode_starts": [],
        "timeout_done": [],
        "terminal_mask": [],
        "bootstrap_mask": [],
    }
    fields_jax: Dict[str, List[float]] = {k: [] for k in fields_ref}

    ep_start_ref = 1.0
    ep_start_jax = 1.0
    ep_t_ref = 0
    ep_t_jax = 0
    reset_count = 0

    for t in range(args.steps):
        val_ref = value_fn(obs_ref["agent_0"])
        val_jax = value_fn(obs_jax["agent_0"])

        action_ref = {"agent_0": policy_0(obs_ref["agent_0"]), "agent_1": policy_1(obs_ref["agent_1"])}
        action_jax = {"agent_0": policy_0(obs_jax["agent_0"]), "agent_1": policy_1(obs_jax["agent_1"])}
        if action_ref != action_jax:
            return CanaryResult(
                name="training_buffer",
                passed=False,
                first_divergence_step=t + 1,
                first_divergence_field="policy_action",
                message=f"Policy selected different actions at step {t + 1}",
                details={"action_ref": action_ref, "action_jax": action_jax},
            )

        obs_ref, _, _, shaped_ref, done_ref, _ = ref_env.step(action_ref)
        obs_jax, _, _, shaped_jax, done_jax, _ = jax_env.step(action_jax)

        ep_t_ref += 1
        ep_t_jax += 1
        timeout_ref = bool(done_ref and ep_t_ref >= args.horizon)
        timeout_jax = bool(done_jax and ep_t_jax >= args.horizon)
        terminal_mask_ref = 0.0 if done_ref else 1.0
        terminal_mask_jax = 0.0 if done_jax else 1.0
        bootstrap_mask_ref = 1.0 if (not done_ref or (timeout_ref and args.bootstrap_timeout)) else 0.0
        bootstrap_mask_jax = 1.0 if (not done_jax or (timeout_jax and args.bootstrap_timeout)) else 0.0

        fields_ref["value_preds"].append(val_ref)
        fields_ref["rewards"].append(float(shaped_ref))
        fields_ref["dones"].append(float(done_ref))
        fields_ref["episode_starts"].append(float(ep_start_ref))
        fields_ref["timeout_done"].append(float(timeout_ref))
        fields_ref["terminal_mask"].append(float(terminal_mask_ref))
        fields_ref["bootstrap_mask"].append(float(bootstrap_mask_ref))

        fields_jax["value_preds"].append(val_jax)
        fields_jax["rewards"].append(float(shaped_jax))
        fields_jax["dones"].append(float(done_jax))
        fields_jax["episode_starts"].append(float(ep_start_jax))
        fields_jax["timeout_done"].append(float(timeout_jax))
        fields_jax["terminal_mask"].append(float(terminal_mask_jax))
        fields_jax["bootstrap_mask"].append(float(bootstrap_mask_jax))

        ep_start_ref = 1.0 if done_ref else 0.0
        ep_start_jax = 1.0 if done_jax else 0.0
        if done_ref:
            ep_t_ref = 0
            reset_count += 1
            obs_ref = ref_env.reset(seed=args.seed + 10000 + reset_count)
        if done_jax:
            ep_t_jax = 0
            obs_jax = jax_env.reset(seed=args.seed + 10000 + reset_count)

    for key in fields_ref:
        arr_ref = np.asarray(fields_ref[key], dtype=np.float32)
        arr_jax = np.asarray(fields_jax[key], dtype=np.float32)
        idx = _first_array_divergence(arr_ref, arr_jax, atol=1e-6)
        if idx is not None:
            return CanaryResult(
                name="training_buffer",
                passed=False,
                first_divergence_step=idx + 1,
                first_divergence_field=key,
                message=f"Buffer field diverged: {key} at step {idx + 1}",
                details={"ref": float(arr_ref[idx]), "jax": float(arr_jax[idx])},
            )

    last_val_ref = value_fn(obs_ref["agent_0"])
    last_val_jax = value_fn(obs_jax["agent_0"])

    adv_ref, ret_ref = _compute_gae_with_masks(
        rewards=np.asarray(fields_ref["rewards"], dtype=np.float32),
        values=np.asarray(fields_ref["value_preds"], dtype=np.float32),
        bootstrap_mask=np.asarray(fields_ref["bootstrap_mask"], dtype=np.float32),
        last_value=last_val_ref,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    adv_jax, ret_jax = _compute_gae_with_masks(
        rewards=np.asarray(fields_jax["rewards"], dtype=np.float32),
        values=np.asarray(fields_jax["value_preds"], dtype=np.float32),
        bootstrap_mask=np.asarray(fields_jax["bootstrap_mask"], dtype=np.float32),
        last_value=last_val_jax,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    for name, a_ref, a_jax in (
        ("advantages", adv_ref, adv_jax),
        ("returns", ret_ref, ret_jax),
    ):
        idx = _first_array_divergence(a_ref, a_jax, atol=1e-5)
        if idx is not None:
            return CanaryResult(
                name="training_buffer",
                passed=False,
                first_divergence_step=idx + 1,
                first_divergence_field=name,
                message=f"Derived field diverged: {name} at step {idx + 1}",
                details={"ref": float(a_ref[idx]), "jax": float(a_jax[idx])},
            )

    inv_ref = np.max(np.abs(ret_ref - (adv_ref + np.asarray(fields_ref["value_preds"], dtype=np.float32))))
    inv_jax = np.max(np.abs(ret_jax - (adv_jax + np.asarray(fields_jax["value_preds"], dtype=np.float32))))
    if inv_ref > 1e-6 or inv_jax > 1e-6:
        return CanaryResult(
            name="training_buffer",
            passed=False,
            first_divergence_step=None,
            first_divergence_field="returns_identity",
            message="Invariant returns == advantages + value_preds failed",
            details={"max_ref_error": float(inv_ref), "max_jax_error": float(inv_jax)},
        )

    return CanaryResult(
        name="training_buffer",
        passed=True,
        first_divergence_step=None,
        first_divergence_field=None,
        message="Rollout buffer and GAE-derived fields remained aligned",
        details={"steps_checked": args.steps, "gamma": args.gamma, "gae_lambda": args.gae_lambda},
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    ez = np.exp(z)
    return ez / np.sum(ez)


def _build_linear_logits_fn(seed: int, obs_shape: Tuple[int, ...], num_actions: int = 6):
    rng = np.random.RandomState(seed)
    flat_dim = int(np.prod(obs_shape))
    w = rng.randn(flat_dim, num_actions).astype(np.float32) / np.sqrt(max(1, flat_dim))
    b = rng.randn(num_actions).astype(np.float32) * 0.01

    def logits_fn(obs: np.ndarray) -> np.ndarray:
        flat = obs.astype(np.float32).reshape(-1)
        return flat @ w + b

    return logits_fn


def _select_action_from_logits(logits: np.ndarray, mode: str, rng: np.random.RandomState) -> int:
    if mode == "deterministic":
        return int(np.argmax(logits))
    probs = _softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def _evaluate_ref_style(args, seeds: List[int], action_mode: str) -> List[Dict[str, Any]]:
    eval_env = ReferenceEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=0.0,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=False,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )
    init_obs = eval_env.reset(seed=seeds[0] if seeds else args.seed)
    logits_0 = _build_linear_logits_fn(args.seed + 601, init_obs["agent_0"].shape)
    logits_1 = _build_linear_logits_fn(args.seed + 602, init_obs["agent_1"].shape)
    out = []
    for i, seed in enumerate(seeds):
        obs = eval_env.reset(seed=seed)
        rng = np.random.RandomState(seed + 7000)
        sparse_sum = 0.0
        shaped_sum = 0.0
        steps = 0
        done = False
        while not done:
            a0 = _select_action_from_logits(logits_0(obs["agent_0"]), action_mode, rng)
            a1 = _select_action_from_logits(logits_1(obs["agent_1"]), action_mode, rng)
            obs, sparse_r, _dense_r, shaped_r, done, _ = eval_env.step({"agent_0": a0, "agent_1": a1})
            sparse_sum += float(sparse_r)
            shaped_sum += float(shaped_r)
            steps += 1
            if steps > args.horizon + 2:
                break
        out.append({"episode": i, "seed": seed, "sparse": sparse_sum, "shaped": shaped_sum, "steps": steps})
    return out


def _evaluate_jax_style(args, seeds: List[int], action_mode: str) -> List[Dict[str, Any]]:
    eval_env = JaxPathEnv(
        layout_name=args.layout_name,
        old_dynamics=args.old_dynamics,
        horizon=args.horizon,
        reward_shaping_factor=0.0,
        rew_shaping_params=args.rew_shaping_params,
        use_phi=False,
        use_legacy_encoding=args.use_legacy_encoding,
        fixed_agent_idx=args.fixed_agent_idx,
    )
    init_obs = eval_env.reset(seed=seeds[0] if seeds else args.seed)
    logits_0 = _build_linear_logits_fn(args.seed + 601, init_obs["agent_0"].shape)
    logits_1 = _build_linear_logits_fn(args.seed + 602, init_obs["agent_1"].shape)
    out = []
    for i, seed in enumerate(seeds):
        obs = eval_env.reset(seed=seed)
        rng = np.random.RandomState(seed + 7000)
        sparse_sum = 0.0
        shaped_sum = 0.0
        steps = 0
        done = False
        while not done:
            a0 = _select_action_from_logits(logits_0(obs["agent_0"]), action_mode, rng)
            a1 = _select_action_from_logits(logits_1(obs["agent_1"]), action_mode, rng)
            obs, sparse_r, _dense_r, shaped_r, done, _ = eval_env.step({"agent_0": a0, "agent_1": a1})
            sparse_sum += float(sparse_r)
            shaped_sum += float(shaped_r)
            steps += 1
            if steps > args.horizon + 2:
                break
        out.append({"episode": i, "seed": seed, "sparse": sparse_sum, "shaped": shaped_sum, "steps": steps})
    return out


def run_eval_protocol(args) -> CanaryResult:
    seeds = [args.seed + i for i in range(args.eval_num_games)]
    ref_eps = _evaluate_ref_style(args, seeds=seeds, action_mode=args.eval_action_mode)
    jax_eps = _evaluate_jax_style(args, seeds=seeds, action_mode=args.eval_action_mode)

    for i, (r, j) in enumerate(zip(ref_eps, jax_eps)):
        for field in ("sparse", "shaped", "steps"):
            if not np.isclose(float(r[field]), float(j[field]), atol=1e-6):
                return CanaryResult(
                    name="eval_protocol",
                    passed=False,
                    first_divergence_step=i + 1,
                    first_divergence_field=field,
                    message=f"Evaluator drift at episode {i + 1}, field={field}",
                    details={"ref_episode": r, "jax_episode": j, "mode": args.eval_action_mode},
                )

    mean_sparse_ref = float(np.mean([x["sparse"] for x in ref_eps]))
    mean_sparse_jax = float(np.mean([x["sparse"] for x in jax_eps]))

    return CanaryResult(
        name="eval_protocol",
        passed=True,
        first_divergence_step=None,
        first_divergence_field=None,
        message="TF-style and JAX-style evaluators matched on same policy/seeds",
        details={
            "num_games": args.eval_num_games,
            "action_mode": args.eval_action_mode,
            "mean_sparse_ref": mean_sparse_ref,
            "mean_sparse_jax": mean_sparse_jax,
        },
    )


def _default_rew_shaping_params() -> Dict[str, float]:
    return {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout-level TF-vs-JAX parity canaries")
    parser.add_argument(
        "--mode",
        choices=["action_replay", "step_hash", "encoding_swap", "training_buffer", "eval_protocol", "all"],
        default="all",
        help="Which canary to run",
    )
    parser.add_argument("--layout-name", type=str, default="coordination_ring_legacy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--old-dynamics", action="store_true", default=True)
    parser.add_argument("--use-legacy-encoding", action="store_true", default=True)
    parser.add_argument("--use-phi", action="store_true", default=False)
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument(
        "--bootstrap-timeout",
        action="store_true",
        default=True,
        help="Treat timeout truncations as bootstrappable for GAE masks",
    )
    parser.add_argument("--eval-num-games", type=int, default=10)
    parser.add_argument("--eval-action-mode", choices=["stochastic", "deterministic"], default="stochastic")
    parser.add_argument(
        "--fixed-agent-idx",
        type=int,
        choices=[0, 1],
        default=0,
        help="Fix controlled player index for strict replay parity",
    )
    parser.add_argument(
        "--random-agent-idx",
        action="store_true",
        help="Use randomized controlled player index on reset (closer to training behavior)",
    )
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()
    args.layout_name = _canonicalize_layout(args.layout_name)
    args.rew_shaping_params = _default_rew_shaping_params()
    if args.random_agent_idx:
        args.fixed_agent_idx = None
    return args


def _print_results(results: List[CanaryResult]) -> None:
    print("\nRollout Parity Canary Results")
    print("=" * 40)
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] {res.name}: {res.message}")
        if not res.passed:
            print(
                f"       first_divergence_step={res.first_divergence_step}, "
                f"field={res.first_divergence_field}"
            )
    overall = all(r.passed for r in results)
    print("=" * 40)
    print("overall_passed:", overall)


def main():
    args = parse_args()
    selected = [args.mode] if args.mode != "all" else [
        "action_replay",
        "step_hash",
        "encoding_swap",
        "training_buffer",
        "eval_protocol",
    ]
    runners = {
        "action_replay": run_action_replay,
        "step_hash": run_step_hash,
        "encoding_swap": run_encoding_swap,
        "training_buffer": run_training_buffer,
        "eval_protocol": run_eval_protocol,
    }

    results: List[CanaryResult] = []
    for name in selected:
        results.append(runners[name](args))

    _print_results(results)

    if args.out_json:
        payload = {
            "config": {
                "mode": args.mode,
                "layout_name": args.layout_name,
                "seed": args.seed,
                "steps": args.steps,
                "horizon": args.horizon,
                "old_dynamics": args.old_dynamics,
                "use_legacy_encoding": args.use_legacy_encoding,
                "use_phi": args.use_phi,
                "reward_shaping_factor": args.reward_shaping_factor,
                "fixed_agent_idx": args.fixed_agent_idx,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "bootstrap_timeout": args.bootstrap_timeout,
                "eval_num_games": args.eval_num_games,
                "eval_action_mode": args.eval_action_mode,
            },
            "results": [asdict(r) for r in results],
            "overall_passed": all(r.passed for r in results),
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote report: {args.out_json}")


if __name__ == "__main__":
    main()
