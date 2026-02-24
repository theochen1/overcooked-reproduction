"""Checkpoint and metadata persistence helpers."""

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_ppo_checkpoint(params: dict, path: str | Path) -> None:
    out_dir = Path(path)
    _ensure_dir(out_dir)
    with (out_dir / "params.pkl").open("wb") as f:
        pickle.dump({"params": params}, f)


def load_ppo_checkpoint(path: str | Path) -> dict:
    in_dir = Path(path)
    with (in_dir / "params.pkl").open("rb") as f:
        payload = pickle.load(f)
    return payload["params"] if isinstance(payload, dict) and "params" in payload else payload


def save_bc_checkpoint(params: dict, bc_metadata: Dict[str, Any], path: str | Path) -> None:
    out_dir = Path(path)
    _ensure_dir(out_dir)
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump({"params": params}, f)
    with (out_dir / "bc_metadata.pkl").open("wb") as f:
        pickle.dump(bc_metadata, f)


def load_bc_checkpoint(path: str | Path) -> Tuple[dict, Dict[str, Any]]:
    in_dir = Path(path)
    with (in_dir / "model.pkl").open("rb") as f:
        params_payload = pickle.load(f)
    with (in_dir / "bc_metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)
    params = params_payload["params"] if isinstance(params_payload, dict) and "params" in params_payload else params_payload
    return params, metadata


def save_training_info(info: Dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    _ensure_dir(out_path.parent)
    with out_path.open("wb") as f:
        pickle.dump(info, f)


def load_training_info(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_best_bc_model_paths(paths_map: Dict[str, Dict[str, str]], path: str | Path) -> None:
    out_path = Path(path)
    _ensure_dir(out_path.parent)
    with out_path.open("wb") as f:
        pickle.dump(paths_map, f)


def load_best_bc_model_paths(path: str | Path) -> Dict[str, Dict[str, str]]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
