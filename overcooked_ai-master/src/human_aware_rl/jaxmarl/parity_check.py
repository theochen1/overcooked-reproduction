"""
Minibatch PPO parity checker using dumped TF tensors.

This compares TensorFlow PPO scalar losses to an independent recomputation
of the same equations (using the dumped minibatch tensors).
"""

import argparse
import json
import os

import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    dump_path = os.path.join(args.in_dir, "tf_minibatch_dump.npz")
    tf_metrics_path = os.path.join(args.in_dir, "tf_metrics.json")

    if not os.path.exists(dump_path):
        raise FileNotFoundError("Missing dump file: {}".format(dump_path))
    if not os.path.exists(tf_metrics_path):
        raise FileNotFoundError("Missing tf metrics file: {}".format(tf_metrics_path))

    with np.load(dump_path) as data:
        actions = data["actions"]
        returns = data["returns"]
        old_values = data["old_values"]
        old_neglogpacs = data["old_neglogpacs"]
        adv_norm = data["adv_norm"]
        logits = data["logits"]
        probs = data["probs"]
        new_values = data["new_values"]
        new_neglogpacs = data["new_neglogpacs"]
        cliprange = float(data["cliprange"])
        ent_coef = float(data["ent_coef"])
        vf_coef = float(data["vf_coef"])

    # Use dumped probs if present; fallback to logits softmax.
    if probs.size == 0:
        probs = softmax(logits)

    ratio = np.exp(old_neglogpacs - new_neglogpacs)
    pg_losses = -adv_norm * ratio
    pg_losses2 = -adv_norm * np.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
    policy_loss = float(np.mean(np.maximum(pg_losses, pg_losses2)))

    vpredclipped = old_values + np.clip(new_values - old_values, -cliprange, cliprange)
    vf_losses1 = np.square(new_values - returns)
    vf_losses2 = np.square(vpredclipped - returns)
    value_loss = float(0.5 * np.mean(np.maximum(vf_losses1, vf_losses2)))

    entropy = float(-np.mean(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)))
    approxkl = float(0.5 * np.mean(np.square(new_neglogpacs - old_neglogpacs)))
    clipfrac = float(np.mean(np.abs(ratio - 1.0) > cliprange))
    total_loss = float(policy_loss - ent_coef * entropy + vf_coef * value_loss)

    with open(tf_metrics_path, "r") as f:
        tf_metrics = json.load(f)

    tf_pre = tf_metrics.get("pre_stats", {})
    tf_total = float(
        tf_pre["policy_loss"] - ent_coef * tf_pre["policy_entropy"] + vf_coef * tf_pre["value_loss"]
    )
    recomputed = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "policy_entropy": entropy,
        "approxkl": approxkl,
        "clipfrac": clipfrac,
        "total_loss": total_loss,
    }
    tf_reference = {
        "policy_loss": float(tf_pre["policy_loss"]),
        "value_loss": float(tf_pre["value_loss"]),
        "policy_entropy": float(tf_pre["policy_entropy"]),
        "approxkl": float(tf_pre["approxkl"]),
        "clipfrac": float(tf_pre["clipfrac"]),
        "total_loss": tf_total,
    }

    abs_diff = {k: abs(recomputed[k] - tf_reference[k]) for k in recomputed.keys()}
    max_abs_diff = max(abs_diff.values()) if abs_diff else 0.0
    passed = max_abs_diff < 1e-5

    out = {
        "dump_file": dump_path,
        "tf_metrics_file": tf_metrics_path,
        "settings": {
            "cliprange": cliprange,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "num_samples": int(actions.shape[0]),
        },
        "recomputed": recomputed,
        "tf_reference": tf_reference,
        "abs_diff": abs_diff,
        "max_abs_diff": max_abs_diff,
        "passed": passed,
        "note": (
            "This validates PPO minibatch scalar equations against TF outputs. "
            "If this passes, remaining TF-vs-JAX mismatch is likely outside scalar loss algebra "
            "(e.g., env dynamics, rollout pipeline, network init/architecture, optimizer state handling)."
        ),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("Wrote parity report:", args.out_json)
    print("passed:", passed, "max_abs_diff:", max_abs_diff)


if __name__ == "__main__":
    main()
