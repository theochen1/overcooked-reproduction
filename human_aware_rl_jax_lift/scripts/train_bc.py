"""CLI for BC training with legacy-compatible output naming."""

import argparse
import pickle
from pathlib import Path
from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np

from human_aware_rl_jax_lift.agents.bc.model import BCPolicy
from human_aware_rl_jax_lift.agents.bc.train import BCTrainConfig, train_bc
from human_aware_rl_jax_lift.config import set_global_seed
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


def _extract_from_human_dataframe(
    data_path: str,
    layout: str,
    split: str,
    num_train_trajs: int | None,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback path: parse legacy human dataframe into split BC features/actions."""
    from human_aware_rl.human.process_dataframes import get_trajs_from_data

    trajs = get_trajs_from_data(
        data_path=data_path,
        train_mdps=[layout],
        ordered_trajs=True,
        human_ai_trajs=False,
    )
    n_eps = len(trajs["ep_observations"])
    if n_eps == 0:
        raise ValueError(f"No BC trajectories found for layout '{layout}' in {data_path}")
    n_train = (n_eps // 2) if num_train_trajs is None else int(num_train_trajs)
    if not (0 < n_train < n_eps):
        raise ValueError(
            f"Invalid split size: num_train_trajs={n_train} with n_episodes={n_eps}. "
            "Expected 0 < num_train_trajs < n_episodes."
        )
    rng = np.random.RandomState(int(split_seed))
    perm = rng.permutation(n_eps)
    train_idx = set(int(i) for i in perm[:n_train])
    use_train = split == "train"

    obs_flat, acts_flat = [], []
    for ep_i, (ep_obs, ep_acts) in enumerate(zip(trajs["ep_observations"], trajs["ep_actions"])):
        in_train = ep_i in train_idx
        if use_train != in_train:
            continue
        for ob, act in zip(ep_obs, ep_acts):
            obs_flat.append(np.asarray(ob, dtype=np.float32))
            act_idx = int(np.asarray(act).reshape(-1)[0])
            acts_flat.append(act_idx)
    if not obs_flat:
        raise ValueError(f"No BC trajectories after split='{split}' for layout '{layout}' in {data_path}")
    return np.asarray(obs_flat, dtype=np.float32), np.asarray(acts_flat, dtype=np.int32)


def _train_single_seed(
    features: np.ndarray,
    actions: np.ndarray,
    seed: int,
    run_dir: Path,
    epochs: int,
    lr: float,
    adam_eps: float,
    batch_size: int,
    model_selection: str = "best_val_loss",
) -> dict:
    rng = set_global_seed(int(seed))
    cfg = BCTrainConfig(
        num_epochs=int(epochs),
        learning_rate=float(lr),
        adam_eps=float(adam_eps),
        batch_size=int(batch_size),
    )
    out = train_bc(
        features=jnp.asarray(features, dtype=jnp.float32),
        labels=jnp.asarray(actions, dtype=jnp.int32),
        rng=rng,
        config=cfg,
    )
    # train_bc returns {"params": best_params, "best_val_loss": float, "final_params": ...}
    if model_selection == "final":
        params = out["final_params"]
    else:
        params = out["params"]  # best_val_loss params
    logits = BCPolicy().apply(params, jnp.asarray(features, dtype=jnp.float32))
    acc = float((jnp.argmax(logits, axis=-1) == jnp.asarray(actions)).mean())
    metadata = {
        "bc_params": {
            "layout_name": str(run_dir.name.split("_bc_")[0]),
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "adam_eps": float(adam_eps),
        },
        "train_info": {
            "train_accuracy": acc,
            "best_val_loss": float(out["best_val_loss"]),
            "num_samples": int(features.shape[0]),
        },
    }
    save_bc_checkpoint(params, metadata, run_dir)
    return {"run_dir": str(run_dir), "train_accuracy": acc, "best_val_loss": float(out["best_val_loss"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC models in human_aware_rl_jax_lift.")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_train_trajs", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data/bc_runs")
    parser.add_argument(
        "--model_selection",
        type=str,
        default="best_val_loss",
        choices=["best_val_loss", "final"],
        help="Which params to use: 'best_val_loss' (default) or 'final' (uses last-epoch model).",
    )
    parser.add_argument(
        "--seed_idx_override",
        type=int,
        default=None,
        help="If set, use this seed index as the 'best' model (overrides automatic selection). "
             "Useful for manual seed selection per layout (e.g. random0 train→2, test→1).",
    )
    args = parser.parse_args()

    with Path(args.data_path).open("rb") as f:
        payload = pickle.load(f)
    try:
        features, actions = _extract_features_labels(payload, args.layout, args.split)
    except ValueError:
        features, actions = _extract_from_human_dataframe(
            args.data_path,
            args.layout,
            args.split,
            args.num_train_trajs,
            args.split_seed,
        )

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    best_idx = -1
    best_val_loss = float("inf")
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
            batch_size=args.batch_size,
            model_selection=args.model_selection,
        )
        summaries.append(summary)
        if summary["best_val_loss"] < best_val_loss:
            best_val_loss = summary["best_val_loss"]
            best_idx = seed_idx

    if args.seed_idx_override is not None:
        n = len(args.seeds)
        if not (0 <= args.seed_idx_override < n):
            raise ValueError(f"--seed_idx_override={args.seed_idx_override} out of range [0, {n})")
        best_idx = args.seed_idx_override

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
            "best_train_accuracy": summaries[best_idx]["train_accuracy"],
            "best_val_loss": summaries[best_idx]["best_val_loss"],
            "best_model_path": best_map[args.split][args.layout],
        }
    )


if __name__ == "__main__":
    main()
