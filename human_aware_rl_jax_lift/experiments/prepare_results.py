"""Aggregate per-layout eval fragments into results_figure4a.json.

This script is called automatically by the Figure 4a Slurm pipeline after all
per-layout evaluation jobs finish. You can also invoke it manually:

    python prepare_results.py \
        --aggregate \
        --eval_dir  eval_results \
        --out       results_figure4a.json

Then plot:
    python figure4a.py \
        --results_path results_figure4a.json \
        --output figure_4a.png

Pipeline overview
─────────────────
  run_eval_figure4a.sh
    └─ sbatch slurm/05_eval_figure4a.slurm (array over 5 layouts)
         └─ writes experiments/eval_results/results_{layout}.json
  prepare_results.py --aggregate
    └─ merges the 5 fragments → experiments/results_figure4a.json
  figure4a.py --results_path experiments/results_figure4a.json
    └─ renders experiments/figure_4a.png
"""

import argparse
import json
from pathlib import Path

# Short layout name (dir convention) → figure4a.py LAYOUT_ORDER key
# NOTE: This mapping must match the paper layout ordering used by plotting.
LAYOUT_MAP = {
    "simple":    "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random1":   "coordination_ring",
    "random0":   "forced_coordination",
    "random3":   "counter_circuit",
}


def aggregate(eval_dir: Path, out_path: Path) -> None:
    """Merge per-layout JSON fragments into a single results file."""
    merged: dict = {}
    missing: list = []

    for dir_layout in LAYOUT_MAP:
        frag = eval_dir / f"results_{dir_layout}.json"
        if not frag.exists():
            missing.append(str(frag))
            continue
        with open(frag, encoding="utf-8") as f:
            merged.update(json.load(f))

    if missing:
        raise FileNotFoundError(
            "The following per-layout result files are missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nRun 'bash run_eval_figure4a.sh' to generate them."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"✓ Merged {len(merged)} layouts into: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-layout eval JSON fragments into results_figure4a.json."
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Run the aggregation step.",
    )
    parser.add_argument(
        "--eval_dir", type=str, default="eval_results",
        help="Directory with results_{layout}.json files (default: eval_results).",
    )
    parser.add_argument(
        "--out", type=str, default="results_figure4a.json",
        help="Output path for the merged JSON (default: results_figure4a.json).",
    )
    args = parser.parse_args()

    if args.aggregate:
        aggregate(Path(args.eval_dir), Path(args.out))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
