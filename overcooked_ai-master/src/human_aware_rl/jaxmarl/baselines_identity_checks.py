"""
Baselines PPO2 identity checks against JAX PPO implementation.

This is a targeted semantic audit for training-loop differences that can
explain long-run drift even when env/rollout parity passes.
"""

import argparse
import json
import os
import re
from typing import Any, Dict


def _contains(src: str, pattern: str) -> bool:
    return re.search(pattern, src, flags=re.MULTILINE) is not None


def run_checks() -> Dict[str, Any]:
    import human_aware_rl.jaxmarl.ppo as jax_ppo
    import inspect

    jax_src = inspect.getsource(jax_ppo)
    jax_update_src = inspect.getsource(jax_ppo.PPOTrainer._make_jit_update_fn)
    jax_train_src = inspect.getsource(jax_ppo.PPOTrainer.train)

    start_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(start_dir, "../../../../"),
        os.path.join(start_dir, "../../../../../"),
        os.path.join(start_dir, "../../../../../../"),
    ]
    tf_learn_path = None
    tf_model_path = None
    for c in candidates:
        root = os.path.abspath(c)
        p_learn = os.path.join(root, "human_aware_rl", "baselines", "baselines", "ppo2", "ppo2.py")
        p_model = os.path.join(root, "human_aware_rl", "baselines", "baselines", "ppo2", "model.py")
        if os.path.exists(p_learn) and os.path.exists(p_model):
            tf_learn_path = p_learn
            tf_model_path = p_model
            break
    if tf_learn_path is None or tf_model_path is None:
        raise FileNotFoundError("Could not locate local baselines PPO2 sources")
    with open(tf_learn_path, "r") as f:
        tf_learn_src = f.read()
    with open(tf_model_path, "r") as f:
        tf_model_src = f.read()

    findings: Dict[str, Any] = {}

    findings["jax"] = {
        "adam_eps_1e5": _contains(jax_src, r"optax\.adam\)\(\s*learning_rate=.*eps=1e-5"),
        "uses_value_clipping": _contains(jax_update_src, r"values_clipped\s*=\s*old_values \+ jnp\.clip"),
        "adv_norm_per_minibatch": _contains(
            jax_train_src,
            r"minibatch\[\"advantages\"\]\s*=\s*\(minibatch_advs - minibatch_advs\.mean\(\)\)\s*/\s*\(minibatch_advs\.std\(\)\s*\+\s*1e-8\)",
        ),
        "lr_schedule_present": _contains(jax_train_src, r"if self\.config\.use_lr_annealing"),
        "cliprange_schedule_present": _contains(jax_train_src, r"clip_eps.*progress|use_clip.*anneal"),
        "global_grad_clip": _contains(jax_src, r"optax\.clip_by_global_norm"),
        "reduction_mean_policy": _contains(jax_update_src, r"actor_loss.*\.mean\(\)"),
        "reduction_mean_value": _contains(jax_update_src, r"critic_loss.*\.mean\(\)"),
    }

    findings["baselines_ppo2"] = {
        "source_path_ppo2": tf_learn_path,
        "source_path_model": tf_model_path,
        "learn_has_lr_callable": _contains(tf_learn_src, r"lr\s*=\s*constfn\(lr\)|if callable\(lr\)"),
        "learn_has_cliprange_callable": _contains(tf_learn_src, r"cliprange\s*=\s*constfn\(cliprange\)|if callable\(cliprange\)"),
        "uses_update_frac": _contains(tf_learn_src, r"frac\s*=\s*1\.0\s*-\s*\(update\s*-\s*1\.0\)\s*/\s*nupdates"),
        "adv_norm_in_model_train": _contains(tf_model_src, r"advs\s*=\s*\(advs - advs\.mean\(\)\)\s*/\s*\(advs\.std\(\) \+ 1e-8\)"),
        "value_clip_formula": _contains(tf_model_src, r"vpredclipped\s*=\s*OLDVPRED \+ tf\.clip_by_value"),
        "adam_eps_1e5": _contains(tf_model_src, r"epsilon=1e-5"),
    }

    findings["delta_flags"] = {
        "potential_cliprange_schedule_mismatch": (
            findings["baselines_ppo2"]["learn_has_cliprange_callable"]
            and findings["baselines_ppo2"]["uses_update_frac"]
            and not findings["jax"]["cliprange_schedule_present"]
        ),
        "potential_lr_schedule_mismatch": (
            findings["baselines_ppo2"]["learn_has_lr_callable"]
            and findings["baselines_ppo2"]["uses_update_frac"]
            and not findings["jax"]["lr_schedule_present"]
        ),
        "adv_norm_scope_mismatch_unlikely": (
            findings["jax"]["adv_norm_per_minibatch"] and findings["baselines_ppo2"]["adv_norm_in_model_train"]
        ),
        "adam_eps_mismatch_unlikely": (
            findings["jax"]["adam_eps_1e5"] and findings["baselines_ppo2"]["adam_eps_1e5"]
        ),
    }

    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=str, required=True)
    args = parser.parse_args()

    report = run_checks()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Wrote identity check report:", args.out_json)
    print("Potential delta flags:")
    for k, v in report["delta_flags"].items():
        print(" ", k, "=", v)


if __name__ == "__main__":
    main()
