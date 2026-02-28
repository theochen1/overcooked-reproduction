"""Summarize stage-1 JAX sweep runs from training_info.pkl files."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


def _safe_last(values, default=0.0):
    return float(values[-1]) if values else float(default)


def _safe_at(values, idx, default=0.0):
    return float(values[idx]) if len(values) > idx else float(default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for pkl_path in sorted(args.sweep_dir.glob("**/training_info.pkl")):
        with pkl_path.open("rb") as f:
            logs = pickle.load(f)
        run_dir = pkl_path.parent.parent
        rows.append(
            {
                "run_dir": str(run_dir),
                "final_true_eprew": _safe_last(logs.get("ep_sparse_rew_mean", [])),
                "u50_true_eprew": _safe_at(logs.get("ep_sparse_rew_mean", []), 49),
                "u100_true_eprew": _safe_at(logs.get("ep_sparse_rew_mean", []), 99),
                "final_entropy": _safe_last(logs.get("policy_entropy", [])),
                "final_approxkl": _safe_last(logs.get("approxkl", [])),
                "final_clipfrac": _safe_last(logs.get("clipfrac", [])),
            }
        )

    rows.sort(key=lambda r: (r["u100_true_eprew"], r["final_true_eprew"]), reverse=True)
    for i, r in enumerate(rows, start=1):
        print(
            f"{i:2d}. u100={r['u100_true_eprew']:.3f} final={r['final_true_eprew']:.3f} "
            f"ent={r['final_entropy']:.3f} kl={r['final_approxkl']:.6f} "
            f"clip={r['final_clipfrac']:.3f} :: {r['run_dir']}"
        )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_dir"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
