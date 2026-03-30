"""
Dump a single TF PPO minibatch and loss terms for parity debugging.

Run from the repository root:
    python -m human_aware_rl.ppo.debug_tf_dump_minibatch --out_dir /tmp/parity
"""

import argparse
import json
import os
import re

import numpy as np
import tensorflow as tf

from baselines.ppo2.runner import Runner
try:
    from human_aware_rl.baselines_utils import create_model, get_vectorized_gym_env
    from human_aware_rl.utils import reset_tf, set_global_seed
except ImportError:
    from baselines_utils import create_model, get_vectorized_gym_env
    from utils import reset_tf, set_global_seed
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _value_breakdown(values, old_values, returns, cliprange):
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


def build_default_params(args):
    batch_size = args.total_batch_size // args.sim_threads
    return {
        "RUN_TYPE": "ppo",
        "SEEDS": [args.seed],
        "LOCAL_TESTING": False,
        "GPU_ID": 0,
        "PPO_RUN_TOT_TIMESTEPS": args.total_batch_size,
        "mdp_params": {
            "layout_name": args.layout,
            "start_order_list": None,
            "rew_shaping_params": {
                "PLACEMENT_IN_POT_REW": 3,
                "DISH_PICKUP_REWARD": 3,
                "SOUP_PICKUP_REWARD": 5,
                "DISH_DISP_DISTANCE_REW": 0,
                "POT_DISTANCE_REW": 0,
                "SOUP_DISTANCE_REW": 0,
            },
        },
        "env_params": {"horizon": args.horizon},
        "sim_threads": args.sim_threads,
        "TOTAL_BATCH_SIZE": args.total_batch_size,
        "BATCH_SIZE": batch_size,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "LR": args.lr,
        "LR_ANNEALING": 1,
        "VF_COEF": args.vf_coef,
        "ENTROPY": args.entropy_coef,
        "STEPS_PER_UPDATE": args.noptepochs,
        "MINIBATCHES": args.nminibatches,
        "CLIPPING": args.cliprange,
        "GAMMA": args.gamma,
        "LAM": args.lam,
        "NUM_HIDDEN_LAYERS": args.num_hidden_layers,
        "SIZE_HIDDEN_LAYERS": args.hidden_dim,
        "NUM_FILTERS": args.num_filters,
        "NUM_CONV_LAYERS": args.num_conv_layers,
        "NETWORK_TYPE": "conv_and_mlp",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--layout", type=str, default="random0")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim_threads", type=int, default=30)
    parser.add_argument("--total_batch_size", type=int, default=12000)
    parser.add_argument("--nminibatches", type=int, default=6)
    parser.add_argument("--noptepochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cliprange", type=float, default=0.05)
    parser.add_argument("--entropy_coef", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.98)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_filters", type=int, default=25)
    parser.add_argument("--num_conv_layers", type=int, default=3)
    parser.add_argument(
        "--minibatch_selection_seed",
        type=int,
        default=12345,
        help="Seed used only for selecting minibatch indices from rollout batch.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    reset_tf()
    set_global_seed(args.seed)
    params = build_default_params(args)

    try:
        mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Layout '{}' not found in legacy overcooked_ai data/layouts. "
            "Try one of: random0, random1, random2, random3, corridor, five_by_five.".format(
                args.layout
            )
        ) from e
    env = OvercookedEnv(mdp, **params["env_params"])
    gym_env = get_vectorized_gym_env(
        env, "Overcooked-v0", featurize_fn=lambda s: mdp.lossless_state_encoding(s), **params
    )
    gym_env.self_play_randomization = 1.0
    gym_env.trajectory_sp = False
    gym_env.update_reward_shaping_param(1.0)

    model = create_model(gym_env, "ppo_agent_debug", **params)
    runner = Runner(
        env=gym_env,
        model=model,
        nsteps=params["BATCH_SIZE"],
        gamma=params["GAMMA"],
        lam=params["LAM"],
    )
    obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()

    nbatch = obs.shape[0]
    mbatch = nbatch // params["MINIBATCHES"]
    rng = np.random.RandomState(args.minibatch_selection_seed)
    perm = rng.permutation(nbatch)
    mbinds = perm[:mbatch]

    mb_obs = obs[mbinds]
    mb_returns = returns[mbinds]
    mb_masks = masks[mbinds]
    mb_actions = actions[mbinds]
    mb_values = values[mbinds]
    mb_neglogpacs = neglogpacs[mbinds]
    mb_advs_raw = mb_returns - mb_values
    mb_advs_norm = (mb_advs_raw - mb_advs_raw.mean()) / (mb_advs_raw.std() + 1e-8)

    td_map = {
        model.train_model.X: mb_obs,
        model.A: mb_actions,
        model.ADV: mb_advs_norm,
        model.R: mb_returns,
        model.LR: args.lr,
        model.CLIPRANGE: args.cliprange,
        model.OLDNEGLOGPAC: mb_neglogpacs,
        model.OLDVPRED: mb_values,
    }
    if states is not None:
        td_map[model.train_model.S] = states
        td_map[model.train_model.M] = mb_masks

    pre_stats = model.sess.run(model.stats_list, td_map)
    pre_logits, pre_values, pre_probs, pre_new_neglogpac = model.sess.run(
        [
            model.train_model.pi,
            model.train_model.vf,
            model.train_model.action_probs,
            model.train_model.pd.neglogp(model.A),
        ],
        td_map,
    )
    pre_value_breakdown = _value_breakdown(pre_values, mb_values, mb_returns, args.cliprange)

    # Export TF trainable parameters before the minibatch update.
    tf_trainable = tf.trainable_variables(model.scope + "/ppo2_model")
    tf_values_pre = model.sess.run(tf_trainable)
    tf_param_map_pre = {v.name: arr for v, arr in zip(tf_trainable, tf_values_pre)}
    tf_param_sanitized_pre = {}
    tf_param_index = {}
    for name, arr in tf_param_map_pre.items():
        key = _sanitize(name)
        tf_param_sanitized_pre[key] = arr.astype(np.float32)
        tf_param_index[key] = {"name": name, "shape": list(arr.shape)}
    tf_params_pre_npz = os.path.join(args.out_dir, "tf_params_pre.npz")
    np.savez(tf_params_pre_npz, **tf_param_sanitized_pre)

    # Rebuild PPO loss to extract raw (pre-clip) gradients for parity diagnostics.
    neglogpac_dbg = model.train_model.pd.neglogp(model.A)
    entropy_dbg = tf.reduce_mean(model.train_model.pd.entropy())
    vpred_dbg = model.train_model.vf
    vpred_clipped_dbg = model.OLDVPRED + tf.clip_by_value(
        model.train_model.vf - model.OLDVPRED, -model.CLIPRANGE, model.CLIPRANGE
    )
    vf_losses1_dbg = tf.square(vpred_dbg - model.R)
    vf_losses2_dbg = tf.square(vpred_clipped_dbg - model.R)
    vf_loss_dbg = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1_dbg, vf_losses2_dbg))
    ratio_dbg = tf.exp(model.OLDNEGLOGPAC - neglogpac_dbg)
    pg_losses_dbg = -model.ADV * ratio_dbg
    pg_losses2_dbg = -model.ADV * tf.clip_by_value(ratio_dbg, 1.0 - model.CLIPRANGE, 1.0 + model.CLIPRANGE)
    pg_loss_dbg = tf.reduce_mean(tf.maximum(pg_losses_dbg, pg_losses2_dbg))
    total_loss_dbg = pg_loss_dbg - entropy_dbg * args.entropy_coef + vf_loss_dbg * args.vf_coef
    tf_raw_grads_ops = tf.gradients(total_loss_dbg, tf_trainable)
    tf_raw_global_norm_op = tf.global_norm(tf_raw_grads_ops)
    tf_raw_grads_clipped_ops, _ = tf.clip_by_global_norm(tf_raw_grads_ops, args.max_grad_norm)
    tf_clipped_global_norm_op = tf.global_norm(tf_raw_grads_clipped_ops)

    tf_raw_grads, tf_raw_global_norm, tf_clipped_global_norm = model.sess.run(
        [tf_raw_grads_ops, tf_raw_global_norm_op, tf_clipped_global_norm_op], td_map
    )
    tf_raw_grad_sanitized_pre = {}
    tf_raw_grad_norms = {}
    for var, grad in zip(tf_trainable, tf_raw_grads):
        key = _sanitize(var.name)
        if grad is None:
            grad_arr = np.zeros_like(tf_param_map_pre[var.name], dtype=np.float32)
        else:
            grad_arr = np.asarray(grad, dtype=np.float32)
        tf_raw_grad_sanitized_pre[key] = grad_arr
        tf_raw_grad_norms[var.name] = float(np.sqrt(np.sum(np.square(grad_arr))))
    tf_raw_grads_pre_npz = os.path.join(args.out_dir, "tf_raw_grads_pre.npz")
    np.savez(tf_raw_grads_pre_npz, **tf_raw_grad_sanitized_pre)

    # Export TF clipped gradients (same tensors used by train_op).
    tf_grads = model.sess.run(model.grads, td_map)
    tf_grad_sanitized_pre = {}
    tf_grad_norms = {}
    for var, grad in zip(tf_trainable, tf_grads):
        key = _sanitize(var.name)
        if grad is None:
            grad_arr = np.zeros_like(tf_param_map_pre[var.name], dtype=np.float32)
        else:
            grad_arr = np.asarray(grad, dtype=np.float32)
        tf_grad_sanitized_pre[key] = grad_arr
        tf_grad_norms[var.name] = float(np.sqrt(np.sum(np.square(grad_arr))))
    tf_grads_pre_npz = os.path.join(args.out_dir, "tf_grads_pre.npz")
    np.savez(tf_grads_pre_npz, **tf_grad_sanitized_pre)

    model.train(args.lr, args.cliprange, mb_obs, mb_returns, mb_masks, mb_actions, mb_values, mb_neglogpacs)
    post_stats = model.sess.run(model.stats_list, td_map)
    post_logits, post_values, post_probs, post_new_neglogpac = model.sess.run(
        [
            model.train_model.pi,
            model.train_model.vf,
            model.train_model.action_probs,
            model.train_model.pd.neglogp(model.A),
        ],
        td_map,
    )
    post_value_breakdown = _value_breakdown(post_values, mb_values, mb_returns, args.cliprange)

    # Export TF trainable parameters after the minibatch update.
    tf_values_post = model.sess.run(tf_trainable)
    tf_param_map_post = {v.name: arr for v, arr in zip(tf_trainable, tf_values_post)}
    tf_param_sanitized_post = {}
    tf_update_sanitized_post = {}
    tf_update_norms = {}
    for name, arr in tf_param_map_post.items():
        key = _sanitize(name)
        arr_post = arr.astype(np.float32)
        arr_pre = tf_param_map_pre[name].astype(np.float32)
        tf_param_sanitized_post[key] = arr_post
        update = arr_post - arr_pre
        tf_update_sanitized_post[key] = update
        tf_update_norms[name] = float(np.sqrt(np.sum(np.square(update))))
    tf_params_post_npz = os.path.join(args.out_dir, "tf_params_post.npz")
    np.savez(tf_params_post_npz, **tf_param_sanitized_post)
    tf_updates_post_npz = os.path.join(args.out_dir, "tf_updates_post.npz")
    np.savez(tf_updates_post_npz, **tf_update_sanitized_post)
    # Backward-compatible alias for previous tooling.
    tf_params_npz = os.path.join(args.out_dir, "tf_params.npz")
    np.savez(tf_params_npz, **tf_param_sanitized_post)
    tf_params_index_json = os.path.join(args.out_dir, "tf_param_index.json")
    with open(tf_params_index_json, "w") as f:
        json.dump(tf_param_index, f, indent=2, sort_keys=True)

    out_npz = os.path.join(args.out_dir, "tf_minibatch_dump.npz")
    np.savez(
        out_npz,
        obs=mb_obs.astype(np.float32),
        actions=mb_actions.astype(np.int32),
        returns=mb_returns.astype(np.float32),
        old_values=mb_values.astype(np.float32),
        old_neglogpacs=mb_neglogpacs.astype(np.float32),
        adv_raw=mb_advs_raw.astype(np.float32),
        adv_norm=mb_advs_norm.astype(np.float32),
        logits=pre_logits.astype(np.float32),
        probs=pre_probs.astype(np.float32),
        new_values=pre_values.astype(np.float32),
        new_neglogpacs=pre_new_neglogpac.astype(np.float32),
        post_logits=post_logits.astype(np.float32),
        post_probs=post_probs.astype(np.float32),
        post_new_values=post_values.astype(np.float32),
        post_new_neglogpacs=post_new_neglogpac.astype(np.float32),
        lr=np.array(args.lr, dtype=np.float32),
        cliprange=np.array(args.cliprange, dtype=np.float32),
        ent_coef=np.array(args.entropy_coef, dtype=np.float32),
        vf_coef=np.array(args.vf_coef, dtype=np.float32),
    )

    loss_names = model.loss_names
    pre_stats_dict = {k: float(v) for k, v in zip(loss_names, pre_stats)}
    post_stats_dict = {k: float(v) for k, v in zip(loss_names, post_stats)}
    summary = {
        "layout": args.layout,
        "seed": args.seed,
        "nbatch": int(nbatch),
        "minibatch_size": int(mbatch),
        "loss_names": loss_names,
        "pre_stats": pre_stats_dict,
        "post_stats_same_minibatch_after_one_update": post_stats_dict,
        "tf_value_breakdown_pre": pre_value_breakdown,
        "tf_value_breakdown_post": post_value_breakdown,
        "dump_file": out_npz,
        "tf_params_file": tf_params_npz,
        "tf_params_pre_file": tf_params_pre_npz,
        "tf_params_post_file": tf_params_post_npz,
        "tf_raw_grads_pre_file": tf_raw_grads_pre_npz,
        "tf_grads_pre_file": tf_grads_pre_npz,
        "tf_updates_post_file": tf_updates_post_npz,
        "tf_param_index_file": tf_params_index_json,
        "tf_raw_grad_l2_by_var": tf_raw_grad_norms,
        "tf_grad_l2_by_var": tf_grad_norms,
        "tf_update_l2_by_var": tf_update_norms,
        "tf_raw_grad_global_norm": float(tf_raw_global_norm),
        "tf_clipped_grad_global_norm": float(tf_clipped_global_norm),
        "tf_grad_clip_coef": float(
            min(1.0, args.max_grad_norm / (float(tf_raw_global_norm) + 1e-12))
        ),
        "num_episode_infos_in_rollout": len(epinfos),
    }

    out_json = os.path.join(args.out_dir, "tf_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Saved minibatch dump to:", out_npz)
    print("Saved TF metrics to:", out_json)
    print("Saved TF params (pre) to:", tf_params_pre_npz)
    print("Saved TF params (post) to:", tf_params_post_npz)
    print("Saved TF raw grads (pre) to:", tf_raw_grads_pre_npz)
    print("Saved TF grads (pre) to:", tf_grads_pre_npz)
    print("Saved TF updates (post-pre) to:", tf_updates_post_npz)


if __name__ == "__main__":
    main()
