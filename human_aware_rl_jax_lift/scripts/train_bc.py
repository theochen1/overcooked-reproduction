"""CLI for BC training with legacy-compatible output naming."""

import argparse
import pickle
from pathlib import Path
from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.agents.bc.train import BCTrainConfig, train_bc
from human_aware_rl_jax_lift.reproducibility.seed import set_global_seed
from human_aware_rl_jax_lift.training.checkpoints import load_best_bc_model_paths, save_bc_checkpoint, save_best_bc_model_paths


def _extract_features_labels(payload: Any, layout: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Best-effort parser for pre-featurized BC datasets."""
    if isinstance(payload, dict):
        if "features" in payload and "actions" in payload:
            return np.asarray(payload["features"]), np.asarray(payload["actions"])
        if layout in payload:
            return _extract_features_labels(payload[layout], layout, split)
        if split in payload:
            return _extract_features_labels(payload[split], layout, split)
        if "X" in payload and "y" in payload:
            return np.asarray(payload["X"]), np.asarray(payload["y"])
    raise ValueError(
        "Could not parse BC dataset. Expected pre-featurized payload containing "
        "`features/actions` (or `X/y`), optionally nested by layout/split."
    )


def _extract_from_human_dataframe(data_path: str, layout: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback path: parse legacy human dataframe into BC features/actions."""
    from human_aware_rl.human.process_dataframes import get_trajs_from_data

    trajs = get_trajs_from_data(
        data_path=data_path,
        train_mdps=[layout],
        ordered_trajs=True,
        human_ai_trajs=False,
    )
    obs_flat, acts_flat = [], []
    for ep_obs, ep_acts in zip(trajs["ep_observations"], trajs["ep_actions"]):
        for ob, act in zip(ep_obs, ep_acts):
            obs_flat.append(np.asarray(ob, dtype=np.float32))
            act_idx = int(np.asarray(act).reshape(-1)[0])
            acts_flat.append(act_idx)
    if not obs_flat:
        raise ValueError(f"No BC trajectories found for layout '{layout}' in {data_path}")
    return np.asarray(obs_flat, dtype=np.float32), np.asarray(acts_flat, dtype=np.int32)


def _train_single_seed(
    features: np.ndarray,
    actions: np.ndarray,
    seed: int,
    run_dir: Path,
    epochs: int,
    lr: float,
    adam_eps: float,
) -> dict:
    rng = set_global_seed(int(seed))
    cfg = BCTrainConfig(num_epochs=int(epochs), learning_rate=float(lr), adam_eps=float(adam_eps), batch_size=256)
    out = train_bc(
        features=jnp.asarray(features, dtype=jnp.float32),
        labels=jnp.asarray(actions, dtype=jnp.int32),
        rng=rng,
        config=cfg,
    )
    params = out["state"].params
    logits = out["state"].apply_fn(params, jnp.asarray(features, dtype=jnp.float32))
    acc = float((jnp.argmax(logits, axis=-1) == jnp.asarray(actions)).mean())
    metadata = {
        "bc_params": {
            "layout_name": str(run_dir.name.split("_bc_")[0]),
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "adam_eps": float(adam_eps),
        },
        "train_info": {"train_accuracy": acc, "num_samples": int(features.shape[0])},
    }
    save_bc_checkpoint(params, metadata, run_dir)
    return {"run_dir": str(run_dir), "train_accuracy": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC models in human_aware_rl_jax_lift.")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--save_dir", type=str, default="data/bc_runs")
    args = parser.parse_args()

    with Path(args.data_path).open("rb") as f:
        payload = pickle.load(f)
    try:
        features, actions = _extract_features_labels(payload, args.layout, args.split)
    except ValueError:
        features, actions = _extract_from_human_dataframe(args.data_path, args.layout)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    best_idx = -1
    best_acc = float("-inf")
    for seed_idx, seed in enumerate(args.seeds):
        run_name = f"{args.layout}_bc_{args.split}_seed{seed_idx}"
        run_dir = save_root / run_name
        summary = _train_single_seed(
            features=features,
            actions=actions,
            seed=seed,
            run_dir=run_dir,
            epochs=args.epochs,
            lr=args.lr,
            adam_eps=args.adam_eps,
        )
        summaries.append(summary)
        if summary["train_accuracy"] > best_acc:
            best_acc = summary["train_accuracy"]
            best_idx = seed_idx

    best_paths_file = save_root / "best_bc_model_paths.pkl"
    if best_paths_file.exists():
        best_map = load_best_bc_model_paths(best_paths_file)
    else:
        best_map = {"train": {}, "test": {}}
    best_map.setdefault("train", {})
    best_map.setdefault("test", {})
    best_map[args.split][args.layout] = str(save_root / f"{args.layout}_bc_{args.split}_seed{best_idx}" / "model.pkl")
    save_best_bc_model_paths(best_map, best_paths_file)

    print(
        {
            "layout": args.layout,
            "split": args.split,
            "num_models": len(summaries),
            "best_seed_idx": best_idx,
            "best_train_accuracy": best_acc,
            "best_model_path": best_map[args.split][args.layout],
        }
    )


if __name__ == "__main__":
    main()
