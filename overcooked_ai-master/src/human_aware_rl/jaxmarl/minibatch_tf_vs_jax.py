"""
Run one JAX PPO update on the TF dumped minibatch and compare metrics.

This script does NOT yet port TF weights into JAX; it compares update behavior
using a freshly initialized JAX network on the same batch tensors.
"""

import argparse
import json
import os
import re

import numpy as np


def _to_float(x):
    return float(np.asarray(x))


def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _load_tf_params(in_dir, snapshot="post"):
    index_path = os.path.join(in_dir, "tf_param_index.json")
    if snapshot == "pre":
        npz_path = os.path.join(in_dir, "tf_params_pre.npz")
    elif snapshot == "post":
        # Backward compatible with older dump format.
        preferred = os.path.join(in_dir, "tf_params_post.npz")
        legacy = os.path.join(in_dir, "tf_params.npz")
        npz_path = preferred if os.path.exists(preferred) else legacy
    else:
        raise ValueError("Unknown snapshot '{}'".format(snapshot))
    if not (os.path.exists(index_path) and os.path.exists(npz_path)):
        return None
    with open(index_path, "r") as f:
        index = json.load(f)
    params = {}
    with np.load(npz_path) as data:
        for key, meta in index.items():
            if key in data:
                params[meta["name"]] = data[key]
    return params


def _load_tf_named_arrays(in_dir, npz_filename):
    index_path = os.path.join(in_dir, "tf_param_index.json")
    npz_path = os.path.join(in_dir, npz_filename)
    if not (os.path.exists(index_path) and os.path.exists(npz_path)):
        return None
    with open(index_path, "r") as f:
        index = json.load(f)
    out = {}
    with np.load(npz_path) as data:
        for key, meta in index.items():
            if key in data:
                out[meta["name"]] = data[key]
    return out


def _dense_index_from_name(name):
    m = re.search(r"/dense(?:_(\d+))?/kernel:0$", name)
    if not m:
        return None
    return 0 if m.group(1) is None else int(m.group(1))


def _port_tf_to_jax(jax_params, tf_params, action_dim):
    if tf_params is None:
        return jax_params, {"ported": False, "reason": "missing tf_params.npz/tf_param_index.json"}, {}

    from flax.core import unfreeze, freeze

    p = unfreeze(jax_params)
    issues = []
    tf_to_jax_path = {}

    def get_tf_entry(name_endswith):
        matches = [k for k in tf_params.keys() if k.endswith(name_endswith)]
        if not matches:
            return None, None
        if len(matches) > 1:
            # Choose the shortest match path as a tie-breaker.
            matches = sorted(matches, key=len)
        m = matches[0]
        return m, tf_params[m]

    # Conv layers.
    for jname, tf_suffix in [
        ("Conv_0", "conv_initial/kernel:0"),
        ("Conv_1", "conv_0/kernel:0"),
        ("Conv_2", "conv_1/kernel:0"),
    ]:
        k_name, k = get_tf_entry(tf_suffix)
        b_name, b = get_tf_entry(tf_suffix.replace("kernel:0", "bias:0"))
        if k is None or b is None:
            issues.append("missing {}".format(tf_suffix))
            continue
        p["params"][jname]["kernel"] = np.asarray(k, dtype=np.float32)
        p["params"][jname]["bias"] = np.asarray(b, dtype=np.float32)
        tf_to_jax_path[k_name] = ("params", jname, "kernel")
        tf_to_jax_path[b_name] = ("params", jname, "bias")

    # Hidden dense layers from tf.layers.dense naming.
    dense_k = [(name, arr) for name, arr in tf_params.items() if _dense_index_from_name(name) is not None]
    dense_k = sorted(dense_k, key=lambda x: _dense_index_from_name(x[0]))
    for i in range(3):
        if i >= len(dense_k):
            issues.append("missing dense_{}".format(i))
            continue
        name, kernel = dense_k[i]
        bname = name.replace("/kernel:0", "/bias:0")
        bias = tf_params.get(bname, None)
        if bias is None:
            issues.append("missing {}".format(bname))
            continue
        jname = "Dense_{}".format(i)
        p["params"][jname]["kernel"] = np.asarray(kernel, dtype=np.float32)
        p["params"][jname]["bias"] = np.asarray(bias, dtype=np.float32)
        tf_to_jax_path[name] = ("params", jname, "kernel")
        tf_to_jax_path[bname] = ("params", jname, "bias")

    # Policy head (pi) and value head (vf).
    pi_w = [(k, v) for k, v in tf_params.items() if k.endswith("/pi/w:0") and v.shape[-1] == action_dim]
    pi_b = [(k, v) for k, v in tf_params.items() if k.endswith("/pi/b:0") and v.shape[-1] == action_dim]
    vf_w = [(k, v) for k, v in tf_params.items() if k.endswith("/vf/w:0") and v.shape[-1] == 1]
    vf_b = [(k, v) for k, v in tf_params.items() if k.endswith("/vf/b:0") and v.shape[-1] == 1]
    if pi_w and pi_b:
        pi_w_name, pi_w_arr = pi_w[0]
        pi_b_name, pi_b_arr = pi_b[0]
        p["params"]["Dense_{}".format(3)]["kernel"] = np.asarray(pi_w_arr, dtype=np.float32)
        p["params"]["Dense_{}".format(3)]["bias"] = np.asarray(pi_b_arr, dtype=np.float32)
        tf_to_jax_path[pi_w_name] = ("params", "Dense_3", "kernel")
        tf_to_jax_path[pi_b_name] = ("params", "Dense_3", "bias")
    else:
        issues.append("missing policy head pi/w,b")
    if vf_w and vf_b:
        vf_w_name, vf_w_arr = vf_w[0]
        vf_b_name, vf_b_arr = vf_b[0]
        p["params"]["Dense_{}".format(4)]["kernel"] = np.asarray(vf_w_arr, dtype=np.float32)
        p["params"]["Dense_{}".format(4)]["bias"] = np.asarray(vf_b_arr, dtype=np.float32)
        tf_to_jax_path[vf_w_name] = ("params", "Dense_4", "kernel")
        tf_to_jax_path[vf_b_name] = ("params", "Dense_4", "bias")
    else:
        issues.append("missing value head vf/w,b")

    return freeze(p), {"ported": len(issues) == 0, "issues": issues}, tf_to_jax_path


def _tree_get(tree, path):
    out = tree
    for k in path:
        out = out[k]
    return out


def _value_breakdown_np(values, old_values, returns, cliprange):
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    ov = np.asarray(old_values, dtype=np.float32).reshape(-1)
    r = np.asarray(returns, dtype=np.float32).reshape(-1)
    diff = v - ov
    v_clipped = ov + np.clip(diff, -cliprange, cliprange)
    sq_unclipped = np.square(v - r)
    sq_clipped = np.square(v_clipped - r)
    use_clipped = sq_clipped > sq_unclipped
    sq_max = np.where(use_clipped, sq_clipped, sq_unclipped)
    return {
        "cliprange": float(cliprange),
        "n": int(v.shape[0]),
        "mean_sq_unclipped": float(np.mean(sq_unclipped)),
        "mean_sq_clipped": float(np.mean(sq_clipped)),
        "mean_sq_max": float(np.mean(sq_max)),
        "value_loss": float(0.5 * np.mean(sq_max)),
        "frac_use_clipped_sq": float(np.mean(use_clipped.astype(np.float32))),
        "max_abs_value_delta_from_old": float(np.max(np.abs(diff))),
        "mean_abs_value_delta_from_old": float(np.mean(np.abs(diff))),
    }


def _dict_absdiff(a, b):
    out = {}
    for k, v in a.items():
        if isinstance(v, (int, float)) and k in b and isinstance(b[k], (int, float)):
            out[k] = float(abs(v - b[k]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_filters", type=int, default=25)
    parser.add_argument("--num_conv_layers", type=int, default=3)
    args = parser.parse_args()

    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        from flax import linen as nn
        import optax
    except ImportError as e:
        raise ImportError(
            "JAX/Flax/Optax or jaxmarl PPO module unavailable. "
            "Run this in a JAX-capable environment/container."
        ) from e

    class ActorCritic(nn.Module):
        action_dim: int
        hidden_dim: int = 64
        num_hidden_layers: int = 3
        num_filters: int = 25
        num_conv_layers: int = 3

        @nn.compact
        def __call__(self, x):
            glorot_init = nn.initializers.glorot_uniform()
            ortho_init_small = nn.initializers.orthogonal(scale=0.01)
            ortho_init_default = nn.initializers.orthogonal(scale=1.0)

            for i in range(self.num_conv_layers):
                kernel_size = (5, 5) if i == 0 else (3, 3)
                x = nn.Conv(
                    features=self.num_filters,
                    kernel_size=kernel_size,
                    padding="SAME" if i < self.num_conv_layers - 1 else "VALID",
                    kernel_init=glorot_init,
                    bias_init=nn.initializers.zeros,
                )(x)
                x = nn.leaky_relu(x, negative_slope=0.2)

            x = x.reshape((x.shape[0], -1))
            for _ in range(self.num_hidden_layers):
                x = nn.Dense(
                    self.hidden_dim,
                    kernel_init=glorot_init,
                    bias_init=nn.initializers.zeros,
                )(x)
                x = nn.leaky_relu(x, negative_slope=0.2)

            actor_logits = nn.Dense(
                self.action_dim,
                kernel_init=ortho_init_small,
                bias_init=nn.initializers.zeros,
            )(x)
            critic = nn.Dense(
                1,
                kernel_init=ortho_init_default,
                bias_init=nn.initializers.zeros,
            )(x)
            return actor_logits, jnp.squeeze(critic, axis=-1)

    dump_path = os.path.join(args.in_dir, "tf_minibatch_dump.npz")
    tf_metrics_path = os.path.join(args.in_dir, "tf_metrics.json")
    if not os.path.exists(dump_path):
        raise FileNotFoundError("Missing dump file: {}".format(dump_path))
    if not os.path.exists(tf_metrics_path):
        raise FileNotFoundError("Missing tf metrics file: {}".format(tf_metrics_path))

    with np.load(dump_path) as data:
        obs = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.int32)
        returns = data["returns"].astype(np.float32)
        old_values = data["old_values"].astype(np.float32)
        old_neglogpacs = data["old_neglogpacs"].astype(np.float32)
        adv_norm = data["adv_norm"].astype(np.float32)
        tf_logits = data["logits"].astype(np.float32)
        tf_new_values = data["new_values"].astype(np.float32)
        tf_post_values = data["post_new_values"].astype(np.float32) if "post_new_values" in data else None
        ent_coef_dump = float(data["ent_coef"])
        vf_coef_dump = float(data["vf_coef"])
        cliprange = float(data["cliprange"])
        lr = float(data["lr"])

    with open(tf_metrics_path, "r") as f:
        tf_metrics = json.load(f)
    # Ground-truth TF path for this experiment builds PPO model via create_model(),
    # which does not forward these coefficients to learn(); use TF ppo2 defaults.
    effective_ent_coef = 0.0
    effective_vf_coef = 0.5
    effective_max_grad_norm = 0.5

    tf_params_pre = _load_tf_params(args.in_dir, snapshot="pre")
    tf_params_post = _load_tf_params(args.in_dir, snapshot="post")
    tf_raw_grads_pre = _load_tf_named_arrays(args.in_dir, "tf_raw_grads_pre.npz")
    tf_grads_pre = _load_tf_named_arrays(args.in_dir, "tf_grads_pre.npz")
    tf_updates_post = _load_tf_named_arrays(args.in_dir, "tf_updates_post.npz")

    obs_j = jnp.asarray(obs)
    actions_j = jnp.asarray(actions)
    returns_j = jnp.asarray(returns)
    old_values_j = jnp.asarray(old_values)
    adv_norm_j = jnp.asarray(adv_norm)
    tf_old_log_probs_j = jnp.asarray(-old_neglogpacs)

    action_dim = int(np.max(actions)) + 1
    key = random.PRNGKey(args.seed)
    network = ActorCritic(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_filters=args.num_filters,
        num_conv_layers=args.num_conv_layers,
    )
    params = network.init(key, jnp.zeros((1, *obs.shape[1:]), dtype=jnp.float32))

    # Keep clipping transform separate; optimizer step below matches TF1 Adam math.
    clip_tx = optax.clip_by_global_norm(effective_max_grad_norm)
    params0, port_info_pre, tf_to_jax_path = _port_tf_to_jax(params, tf_params_pre, action_dim)
    params_post_ref, port_info_post, _ = _port_tf_to_jax(params, tf_params_post, action_dim)
    opt_state0 = {
        "m": jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params0),
        "v": jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params0),
        # TF Adam starts with beta powers at 1.0 and multiplies before computing lr_t.
        "beta1_power": jnp.asarray(1.0, dtype=jnp.float32),
        "beta2_power": jnp.asarray(1.0, dtype=jnp.float32),
    }

    def compute_terms(curr_params, old_log_probs):
        logits, values = network.apply(curr_params, obs_j)
        log_probs_all = jax.nn.log_softmax(logits)
        log_probs = log_probs_all[jnp.arange(actions_j.shape[0]), actions_j]
        ratio = jnp.exp(log_probs - old_log_probs)
        actor_loss1 = -adv_norm_j * ratio
        actor_loss2 = -adv_norm_j * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
        policy_loss = jnp.mean(jnp.maximum(actor_loss1, actor_loss2))
        values_clipped = old_values_j + jnp.clip(values - old_values_j, -cliprange, cliprange)
        value_loss = 0.5 * jnp.mean(
            jnp.maximum(jnp.square(values - returns_j), jnp.square(values_clipped - returns_j))
        )
        probs = jax.nn.softmax(logits)
        entropy = -jnp.mean(jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1))
        approxkl = 0.5 * jnp.mean(jnp.square(log_probs - old_log_probs))
        clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > cliprange)
        total = policy_loss - effective_ent_coef * entropy + effective_vf_coef * value_loss
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "policy_entropy": entropy,
            "approxkl": approxkl,
            "clipfrac": clipfrac,
            "total_loss": total,
            "logits": logits,
            "values": values,
            "log_probs": log_probs,
        }

    def tf1_adam_step(clipped_grads, curr_opt_state, learning_rate, beta1=0.9, beta2=0.999, eps=1e-5):
        m_new = jax.tree_util.tree_map(
            lambda m, g: beta1 * m + (1.0 - beta1) * g,
            curr_opt_state["m"],
            clipped_grads,
        )
        v_new = jax.tree_util.tree_map(
            lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g),
            curr_opt_state["v"],
            clipped_grads,
        )
        beta1_power_new = curr_opt_state["beta1_power"] * beta1
        beta2_power_new = curr_opt_state["beta2_power"] * beta2
        lr_t = learning_rate * jnp.sqrt(1.0 - beta2_power_new) / (1.0 - beta1_power_new)
        updates = jax.tree_util.tree_map(
            lambda m, v: -lr_t * m / (jnp.sqrt(v) + eps),
            m_new,
            v_new,
        )
        new_state = {
            "m": m_new,
            "v": v_new,
            "beta1_power": beta1_power_new,
            "beta2_power": beta2_power_new,
        }
        return updates, new_state, lr_t

    def one_step(curr_params, curr_opt_state, old_log_probs):
        def loss_fn(p):
            return compute_terms(p, old_log_probs)["total_loss"]

        grads = jax.grad(loss_fn)(curr_params)
        clipped_grads, _ = clip_tx.update(grads, None, curr_params)
        updates, new_opt_state, lr_t = tf1_adam_step(clipped_grads, curr_opt_state, lr)
        new_params = optax.apply_updates(curr_params, updates)
        return new_params, new_opt_state, grads, clipped_grads, updates, lr_t

    # Anchor A: TF old log-probs from dump.
    pre_tf_anchor = compute_terms(params0, tf_old_log_probs_j)
    params1_tf_anchor, _, grads_tf_anchor, clipped_grads_tf_anchor, updates_tf_anchor, lr_t_tf_anchor = one_step(
        params0, opt_state0, tf_old_log_probs_j
    )
    post_tf_anchor = compute_terms(params1_tf_anchor, tf_old_log_probs_j)

    # Anchor B: JAX self old log-probs from its own initial policy.
    jax_old_log_probs = pre_tf_anchor["log_probs"]
    pre_jax_anchor = compute_terms(params0, jax_old_log_probs)
    params1_jax_anchor, _, _, _, _, _ = one_step(params0, opt_state0, jax_old_log_probs)
    post_jax_anchor = compute_terms(params1_jax_anchor, jax_old_log_probs)

    # Reference terms using directly ported TF post-update params.
    tf_post_port_terms = compute_terms(params_post_ref, tf_old_log_probs_j)

    tf_pre = tf_metrics.get("pre_stats", {})
    tf_post = tf_metrics.get("post_stats_same_minibatch_after_one_update", {})
    tf_total_pre = (
        float(tf_pre["policy_loss"])
        - effective_ent_coef * float(tf_pre["policy_entropy"])
        + effective_vf_coef * float(tf_pre["value_loss"])
    )
    tf_total_post = (
        float(tf_post["policy_loss"])
        - effective_ent_coef * float(tf_post["policy_entropy"])
        + effective_vf_coef * float(tf_post["value_loss"])
    )
    tf_value_breakdown_pre = tf_metrics.get("tf_value_breakdown_pre", None)
    tf_value_breakdown_post = tf_metrics.get("tf_value_breakdown_post", None)

    raw_grad_compare = {}
    clipped_grad_compare = {}
    grad_update_compare = {}
    if tf_grads_pre is not None and tf_updates_post is not None:
        for tf_name, path in tf_to_jax_path.items():
            tf_g_raw = tf_raw_grads_pre.get(tf_name, None) if tf_raw_grads_pre is not None else None
            tf_g = tf_grads_pre.get(tf_name, None)
            tf_u = tf_updates_post.get(tf_name, None)
            jax_g_raw = np.asarray(_tree_get(grads_tf_anchor, path))
            jax_g = np.asarray(_tree_get(clipped_grads_tf_anchor, path))
            jax_u = np.asarray(_tree_get(updates_tf_anchor, path))
            raw_entry = {}
            if tf_g_raw is not None and tf_g_raw.shape == jax_g_raw.shape:
                raw_entry["grad_l2_tf_raw"] = float(np.sqrt(np.sum(np.square(tf_g_raw))))
                raw_entry["grad_l2_jax_raw"] = float(np.sqrt(np.sum(np.square(jax_g_raw))))
                raw_entry["grad_l2_diff_raw"] = float(np.sqrt(np.sum(np.square(tf_g_raw - jax_g_raw))))
                raw_entry["grad_max_abs_diff_raw"] = float(np.max(np.abs(tf_g_raw - jax_g_raw)))
            raw_grad_compare[tf_name] = raw_entry
            clipped_entry = {}
            entry = {}
            if tf_g is not None and tf_g.shape == jax_g.shape:
                clipped_entry["grad_l2_tf_clipped"] = float(np.sqrt(np.sum(np.square(tf_g))))
                clipped_entry["grad_l2_jax_clipped"] = float(np.sqrt(np.sum(np.square(jax_g))))
                clipped_entry["grad_l2_diff_clipped"] = float(np.sqrt(np.sum(np.square(tf_g - jax_g))))
                clipped_entry["grad_max_abs_diff_clipped"] = float(np.max(np.abs(tf_g - jax_g)))
            if tf_u is not None and tf_u.shape == jax_u.shape:
                entry["update_l2_tf"] = float(np.sqrt(np.sum(np.square(tf_u))))
                entry["update_l2_jax"] = float(np.sqrt(np.sum(np.square(jax_u))))
                entry["update_l2_diff"] = float(np.sqrt(np.sum(np.square(tf_u - jax_u))))
                entry["update_max_abs_diff"] = float(np.max(np.abs(tf_u - jax_u)))
            clipped_grad_compare[tf_name] = clipped_entry
            grad_update_compare[tf_name] = entry

    # Aggregate scalar diagnostics.
    raw_grad_l2_diffs = [v.get("grad_l2_diff_raw", 0.0) for v in raw_grad_compare.values() if "grad_l2_diff_raw" in v]
    clipped_grad_l2_diffs = [
        v.get("grad_l2_diff_clipped", 0.0) for v in clipped_grad_compare.values() if "grad_l2_diff_clipped" in v
    ]
    update_l2_diffs = [v.get("update_l2_diff", 0.0) for v in grad_update_compare.values() if "update_l2_diff" in v]
    raw_grad_max_abs_diffs = [
        v.get("grad_max_abs_diff_raw", 0.0) for v in raw_grad_compare.values() if "grad_max_abs_diff_raw" in v
    ]
    clipped_grad_max_abs_diffs = [
        v.get("grad_max_abs_diff_clipped", 0.0)
        for v in clipped_grad_compare.values()
        if "grad_max_abs_diff_clipped" in v
    ]
    update_max_abs_diffs = [v.get("update_max_abs_diff", 0.0) for v in grad_update_compare.values() if "update_max_abs_diff" in v]

    def _tree_global_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        acc = 0.0
        for x in leaves:
            arr = np.asarray(x)
            acc += float(np.sum(np.square(arr)))
        return float(np.sqrt(acc))

    jax_raw_grad_global_norm = _tree_global_norm(grads_tf_anchor)
    jax_clipped_grad_global_norm = _tree_global_norm(clipped_grads_tf_anchor)
    pre_value_breakdown_jax = _value_breakdown_np(pre_tf_anchor["values"], old_values, returns, cliprange)
    post_value_breakdown_jax = _value_breakdown_np(post_tf_anchor["values"], old_values, returns, cliprange)
    tf_post_port_value_breakdown = _value_breakdown_np(tf_post_port_terms["values"], old_values, returns, cliprange)
    tf_dump_value_breakdown_from_arrays = _value_breakdown_np(tf_new_values, old_values, returns, cliprange)
    tf_dump_post_value_breakdown_from_arrays = (
        _value_breakdown_np(tf_post_values, old_values, returns, cliprange)
        if tf_post_values is not None
        else None
    )

    out = {
        "meta": {
            "seed": args.seed,
            "num_samples": int(obs.shape[0]),
            "obs_shape": list(obs.shape[1:]),
            "action_dim": action_dim,
            "tf_weight_port_pre": port_info_pre,
            "tf_weight_port_post": port_info_post,
        },
        "tf_reference": {
            "pre": {
                **{k: float(tf_pre[k]) for k in ("policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac")},
                "total_loss": tf_total_pre,
            },
            "post_same_minibatch": {
                **{
                    k: float(tf_post[k])
                    for k in ("policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac")
                },
                "total_loss": tf_total_post,
            },
        },
        "jax_tf_anchor": {
            "pre": {
                k: _to_float(pre_tf_anchor[k])
                for k in ("policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac", "total_loss")
            },
            "post_same_minibatch": {
                k: _to_float(post_tf_anchor[k])
                for k in ("policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac", "total_loss")
            },
        },
        "jax_self_anchor": {
            "pre": {
                k: _to_float(pre_jax_anchor[k])
                for k in ("policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac", "total_loss")
            },
            "post_same_minibatch": {
                k: _to_float(post_jax_anchor[k])
                for k in ("policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac", "total_loss")
            },
        },
        "diagnostics": {
            "pre_logits_l2_vs_tf_logits": _to_float(
                jnp.sqrt(jnp.mean(jnp.square(pre_tf_anchor["logits"] - jnp.asarray(tf_logits))))
            ),
            "pre_values_l2_vs_tf_values": _to_float(
                jnp.sqrt(jnp.mean(jnp.square(pre_tf_anchor["values"] - jnp.asarray(tf_new_values))))
            ),
            "post_logits_l2_jax_step_vs_tf_post_port": _to_float(
                jnp.sqrt(jnp.mean(jnp.square(post_tf_anchor["logits"] - tf_post_port_terms["logits"])))
            ),
            "post_values_l2_jax_step_vs_tf_post_port": _to_float(
                jnp.sqrt(jnp.mean(jnp.square(post_tf_anchor["values"] - tf_post_port_terms["values"])))
            ),
            "post_total_loss_absdiff_jax_step_vs_tf_post_port": abs(
                _to_float(post_tf_anchor["total_loss"]) - _to_float(tf_post_port_terms["total_loss"])
            ),
            "raw_grad_compare_max_l2_diff": max(raw_grad_l2_diffs) if raw_grad_l2_diffs else None,
            "clipped_grad_compare_max_l2_diff": max(clipped_grad_l2_diffs) if clipped_grad_l2_diffs else None,
            "update_compare_max_l2_diff": max(update_l2_diffs) if update_l2_diffs else None,
            "raw_grad_compare_max_abs_diff": max(raw_grad_max_abs_diffs) if raw_grad_max_abs_diffs else None,
            "clipped_grad_compare_max_abs_diff": max(clipped_grad_max_abs_diffs) if clipped_grad_max_abs_diffs else None,
            "update_compare_max_abs_diff": max(update_max_abs_diffs) if update_max_abs_diffs else None,
            "pre_value_loss_absdiff_jax_vs_tf_breakdown": (
                abs(pre_value_breakdown_jax["value_loss"] - tf_value_breakdown_pre["value_loss"])
                if tf_value_breakdown_pre is not None
                else None
            ),
            "post_value_loss_absdiff_jax_vs_tf_breakdown": (
                abs(post_value_breakdown_jax["value_loss"] - tf_value_breakdown_post["value_loss"])
                if tf_value_breakdown_post is not None
                else None
            ),
            "jax_raw_grad_global_norm": jax_raw_grad_global_norm,
            "jax_clipped_grad_global_norm": jax_clipped_grad_global_norm,
            "jax_grad_clip_coef": float(
                min(1.0, effective_max_grad_norm / (jax_raw_grad_global_norm + 1e-12))
            ),
            "jax_tf1_adam_lr_t_step1": float(np.asarray(lr_t_tf_anchor)),
            "tf_raw_grad_global_norm": tf_metrics.get("tf_raw_grad_global_norm", None),
            "tf_clipped_grad_global_norm": tf_metrics.get("tf_clipped_grad_global_norm", None),
            "tf_grad_clip_coef": tf_metrics.get("tf_grad_clip_coef", None),
        },
        "raw_grad_compare_by_param": raw_grad_compare,
        "clipped_grad_compare_by_param": clipped_grad_compare,
        "update_compare_by_param": grad_update_compare,
        "grad_update_compare_by_param": grad_update_compare,
        "value_breakdown": {
            "jax_pre_tf_anchor": pre_value_breakdown_jax,
            "jax_post_tf_anchor": post_value_breakdown_jax,
            "jax_tf_post_port_reference": tf_post_port_value_breakdown,
            "tf_dump_pre_from_arrays": tf_dump_value_breakdown_from_arrays,
            "tf_dump_post_from_arrays": tf_dump_post_value_breakdown_from_arrays,
            "tf_metrics_pre": tf_value_breakdown_pre,
            "tf_metrics_post": tf_value_breakdown_post,
            "pre_absdiff_jax_vs_tf_metrics": (
                _dict_absdiff(pre_value_breakdown_jax, tf_value_breakdown_pre)
                if tf_value_breakdown_pre is not None
                else None
            ),
            "post_absdiff_jax_vs_tf_metrics": (
                _dict_absdiff(post_value_breakdown_jax, tf_value_breakdown_post)
                if tf_value_breakdown_post is not None
                else None
            ),
        },
        "tf_post_port_reference": {
            k: _to_float(tf_post_port_terms[k])
            for k in ("policy_loss", "value_loss", "policy_entropy", "approxkl", "clipfrac", "total_loss")
        },
    }
    out["meta"]["effective_ent_coef"] = effective_ent_coef
    out["meta"]["effective_vf_coef"] = effective_vf_coef
    out["meta"]["effective_max_grad_norm"] = effective_max_grad_norm
    out["meta"]["dumped_ent_coef"] = ent_coef_dump
    out["meta"]["dumped_vf_coef"] = vf_coef_dump

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print("Wrote JAX-vs-TF minibatch comparison:", args.out_json)


if __name__ == "__main__":
    main()
