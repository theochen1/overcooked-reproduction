"""Compare TF PPO minibatch artifacts against JAX PPO numerics.

Expected input directory (from legacy TF dump script):
  - tf_minibatch_dump.npz
  - tf_metrics.json
  - tf_param_index.json
  - tf_params_pre.npz
  - tf_raw_grads_pre.npz

This script:
1) Loads TF parameters into the JAX model by explicit layer mapping.
2) Compares per-layer init stats (mean/std/min/max/l2).
3) Recomputes JAX PPO losses/gradients on the same minibatch.
4) Compares JAX per-layer gradient norms to TF raw gradient norms.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from flax.core import freeze, unfreeze

from human_aware_rl_jax_lift.agents.ppo.config import PPOConfig
from human_aware_rl_jax_lift.agents.ppo.model import ActorCriticCNN


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "l2": float(math.sqrt(float(np.sum(np.square(arr))))),
    }


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _tf_name_to_jax_path(tf_name: str) -> Tuple[str, str] | None:
    """Map TF variable names to JAX module path + parameter name."""
    if "conv_initial/kernel" in tf_name:
        return ("Conv_0", "kernel")
    if "conv_initial/bias" in tf_name:
        return ("Conv_0", "bias")
    if "conv_0/kernel" in tf_name:
        return ("Conv_1", "kernel")
    if "conv_0/bias" in tf_name:
        return ("Conv_1", "bias")
    if "conv_1/kernel" in tf_name:
        return ("Conv_2", "kernel")
    if "conv_1/bias" in tf_name:
        return ("Conv_2", "bias")
    if "/dense/kernel" in tf_name and "/dense_" not in tf_name:
        return ("Dense_0", "kernel")
    if "/dense/bias" in tf_name and "/dense_" not in tf_name:
        return ("Dense_0", "bias")
    if "/dense_1/kernel" in tf_name:
        return ("Dense_1", "kernel")
    if "/dense_1/bias" in tf_name:
        return ("Dense_1", "bias")
    if "/dense_2/kernel" in tf_name:
        return ("Dense_2", "kernel")
    if "/dense_2/bias" in tf_name:
        return ("Dense_2", "bias")
    if "/pi/w" in tf_name:
        return ("Dense_3", "kernel")
    if "/pi/b" in tf_name:
        return ("Dense_3", "bias")
    if "/vf/w" in tf_name:
        return ("Dense_4", "kernel")
    if "/vf/b" in tf_name:
        return ("Dense_4", "bias")
    return None


def _build_jax_params_from_tf(
    tf_param_index: dict,
    tf_params_pre_npz: np.lib.npyio.NpzFile,
    obs_shape: tuple[int, int, int],
    seed: int,
) -> tuple[dict, list[str]]:
    """Initialize JAX params and overwrite mapped tensors with TF tensors."""
    model = ActorCriticCNN(num_actions=6, num_filters=25, hidden_dim=32)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,) + obs_shape, dtype=jnp.float32))
    p = unfreeze(params)
    missing = []
    for sanitized, meta in tf_param_index.items():
        tf_name = meta["name"]
        mapping = _tf_name_to_jax_path(tf_name)
        if mapping is None:
            continue
        module_name, param_name = mapping
        if sanitized not in tf_params_pre_npz:
            missing.append(f"missing npz tensor for {tf_name}")
            continue
        if module_name not in p["params"] or param_name not in p["params"][module_name]:
            missing.append(f"missing jax path params/{module_name}/{param_name} for {tf_name}")
            continue
        tf_arr = np.asarray(tf_params_pre_npz[sanitized], dtype=np.float32)
        jax_arr = np.asarray(p["params"][module_name][param_name], dtype=np.float32)
        if tf_arr.shape != jax_arr.shape:
            missing.append(
                f"shape mismatch {tf_name}: tf={tf_arr.shape} jax={jax_arr.shape}"
            )
            continue
        p["params"][module_name][param_name] = jnp.asarray(tf_arr)
    return freeze(p), missing


def _flatten_named(tree: dict) -> Dict[str, np.ndarray]:
    flat = traverse_util.flatten_dict(unfreeze(tree), sep="/")
    out: Dict[str, np.ndarray] = {}
    for k, v in flat.items():
        if not k.startswith("params/"):
            continue
        out[k] = np.asarray(v, dtype=np.float32)
    return out


def _loss_and_grads(
    params: dict,
    obs: np.ndarray,
    actions: np.ndarray,
    old_logp: np.ndarray,
    old_values: np.ndarray,
    returns: np.ndarray,
    cliprange: float,
    ent_coef: float,
    vf_coef: float,
):
    model = ActorCriticCNN(num_actions=6, num_filters=25, hidden_dim=32)
    obs_j = jnp.asarray(obs, dtype=jnp.float32)
    actions_j = jnp.asarray(actions, dtype=jnp.int32)
    old_logp_j = jnp.asarray(old_logp, dtype=jnp.float32)
    old_values_j = jnp.asarray(old_values, dtype=jnp.float32)
    returns_j = jnp.asarray(returns, dtype=jnp.float32)

    def _loss_fn(p):
        logits, values = model.apply(p, obs_j)
        logp_all = jax.nn.log_softmax(logits)
        logp = logp_all[jnp.arange(actions_j.shape[0]), actions_j]
        ratio = jnp.exp(logp - old_logp_j)

        advantages = returns_j - old_values_j
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
        actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        v_clip = old_values_j + jnp.clip(values - old_values_j, -cliprange, cliprange)
        vf_loss1 = jnp.square(values - returns_j)
        vf_loss2 = jnp.square(v_clip - returns_j)
        value_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2).mean()

        entropy = -(jax.nn.softmax(logits) * logp_all).sum(axis=-1).mean()
        approxkl = 0.5 * jnp.mean(jnp.square(old_logp_j - logp))
        clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > cliprange).astype(jnp.float32))
        total = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total, {
            "loss": total,
            "policy_loss": actor_loss,
            "value_loss": value_loss,
            "policy_entropy": entropy,
            "approxkl": approxkl,
            "clipfrac": clipfrac,
        }

    (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)
    del loss
    return metrics, grads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    dump_npz = np.load(in_dir / "tf_minibatch_dump.npz")
    tf_metrics = json.loads((in_dir / "tf_metrics.json").read_text())
    tf_param_index = json.loads((in_dir / "tf_param_index.json").read_text())
    tf_params_pre = np.load(in_dir / "tf_params_pre.npz")
    tf_raw_grads_pre = np.load(in_dir / "tf_raw_grads_pre.npz")

    obs = np.asarray(dump_npz["obs"], dtype=np.float32)
    actions = np.asarray(dump_npz["actions"], dtype=np.int32)
    returns = np.asarray(dump_npz["returns"], dtype=np.float32)
    old_values = np.asarray(dump_npz["old_values"], dtype=np.float32)
    old_neglogpacs = np.asarray(dump_npz["old_neglogpacs"], dtype=np.float32)
    cliprange = float(np.asarray(dump_npz["cliprange"]))
    ent_coef = float(np.asarray(dump_npz["ent_coef"]))
    vf_coef = float(np.asarray(dump_npz["vf_coef"]))

    obs_shape = tuple(int(x) for x in obs.shape[1:])
    jax_params, mapping_warnings = _build_jax_params_from_tf(
        tf_param_index=tf_param_index,
        tf_params_pre_npz=tf_params_pre,
        obs_shape=obs_shape,
        seed=args.seed,
    )

    # Init/stat comparison.
    jax_flat = _flatten_named(jax_params)
    tf_to_jax_stats = {}
    for sanitized, meta in tf_param_index.items():
        tf_name = meta["name"]
        mapping = _tf_name_to_jax_path(tf_name)
        if mapping is None or sanitized not in tf_params_pre:
            continue
        module_name, param_name = mapping
        jax_key = f"params/{module_name}/{param_name}"
        if jax_key not in jax_flat:
            continue
        tf_arr = np.asarray(tf_params_pre[sanitized], dtype=np.float32)
        jax_arr = jax_flat[jax_key]
        tf_to_jax_stats[tf_name] = {
            "jax_path": jax_key,
            "tf": _stats(tf_arr),
            "jax_loaded": _stats(jax_arr),
            "max_abs_diff": float(np.max(np.abs(tf_arr - jax_arr))),
            "mean_abs_diff": float(np.mean(np.abs(tf_arr - jax_arr))),
        }

    # Gradient/loss comparison on same minibatch.
    jax_metrics, jax_grads = _loss_and_grads(
        params=jax_params,
        obs=obs,
        actions=actions,
        old_logp=old_neglogpacs,
        old_values=old_values,
        returns=returns,
        cliprange=cliprange,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
    )

    jax_grad_flat = _flatten_named(jax_grads)
    grad_l2_by_var = {}
    for sanitized, meta in tf_param_index.items():
        tf_name = meta["name"]
        mapping = _tf_name_to_jax_path(tf_name)
        if mapping is None:
            continue
        module_name, param_name = mapping
        jax_key = f"params/{module_name}/{param_name}"
        if jax_key not in jax_grad_flat:
            continue
        tf_grad = np.asarray(tf_raw_grads_pre[sanitized], dtype=np.float32)
        jax_grad = np.asarray(jax_grad_flat[jax_key], dtype=np.float32)
        grad_l2_by_var[tf_name] = {
            "jax_path": jax_key,
            "tf_raw_grad_l2": float(np.sqrt(np.sum(np.square(tf_grad)))),
            "jax_raw_grad_l2": float(np.sqrt(np.sum(np.square(jax_grad)))),
            "grad_l2_abs_diff": float(
                abs(
                    np.sqrt(np.sum(np.square(tf_grad)))
                    - np.sqrt(np.sum(np.square(jax_grad)))
                )
            ),
            "grad_cosine": float(
                np.dot(tf_grad.reshape(-1), jax_grad.reshape(-1))
                / (
                    (np.linalg.norm(tf_grad.reshape(-1)) + 1e-12)
                    * (np.linalg.norm(jax_grad.reshape(-1)) + 1e-12)
                )
            ),
        }

    out = {
        "obs_shape": list(obs_shape),
        "cliprange": cliprange,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "mapping_warnings": mapping_warnings,
        "init_stats_by_var": tf_to_jax_stats,
        "tf_pre_stats": tf_metrics.get("pre_stats", {}),
        "jax_pre_stats": {k: float(v) for k, v in jax_metrics.items()},
        "tf_raw_grad_global_norm": tf_metrics.get("tf_raw_grad_global_norm"),
        "tf_clipped_grad_global_norm": tf_metrics.get("tf_clipped_grad_global_norm"),
        "grad_l2_by_var": grad_l2_by_var,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Saved comparison to: {out_path}")


if __name__ == "__main__":
    main()
