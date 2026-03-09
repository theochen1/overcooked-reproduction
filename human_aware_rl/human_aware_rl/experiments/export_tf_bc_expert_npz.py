#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from human_aware_rl.human.process_dataframes import get_trajs_from_data, save_npz_file


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--data_path", required=True, help="e.g. human_aware_rl/human_aware_rl/data/human/clean_train_trials.pkl")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--ordered_trajs", action="store_true", default=True)
    ap.add_argument("--human_ai_trajs", action="store_true", default=False)
    args = ap.parse_args()

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    trajs = get_trajs_from_data(
        data_path=args.data_path,
        train_mdps=[args.layout],
        ordered_trajs=bool(args.ordered_trajs),
        human_ai_trajs=bool(args.human_ai_trajs),
    )
    save_npz_file(trajs, str(out_npz))

    payload = np.load(out_npz, allow_pickle=True)
    report = {
        "layout": args.layout,
        "data_path": args.data_path,
        "out_npz": str(out_npz),
        "sha256": _sha256_file(out_npz),
        "keys": sorted(list(payload.keys())),
    }

    if "obs" in payload:
        report["obs_shape"] = list(payload["obs"].shape)
        report["obs_dtype"] = str(payload["obs"].dtype)
    if "actions" in payload:
        a = np.asarray(payload["actions"]).astype(np.int64).reshape(-1)
        u, c = np.unique(a, return_counts=True)
        report["n_samples"] = int(a.shape[0])
        report["actions_hist"] = {str(int(ui)): int(ci) for ui, ci in zip(u, c)}

    out_json = out_npz.with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wrote_npz": str(out_npz), "wrote_json": str(out_json)}))


if __name__ == "__main__":
    main()
