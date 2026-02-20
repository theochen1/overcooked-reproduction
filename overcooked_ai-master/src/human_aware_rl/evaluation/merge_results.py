import argparse
import json
import os
from glob import glob
from typing import Dict, List


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def merge_layout_results(files: List[str]) -> Dict:
    merged = {}
    for path in files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        # Expect top-level dict keyed by layout(s)
        if not isinstance(data, dict):
            print(f"Skipping {path}: not a dict")
            continue

        # Merge keys; later files overwrite earlier if duplicate
        merged.update(data)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge per-layout eval JSON files")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing run3_results_<layout>*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write merged JSON",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "run3_results_*.json")
    files = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    print(f"Merging {len(files)} files from {args.input_dir}")
    merged = merge_layout_results(files)

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged results written to {args.output}")


if __name__ == "__main__":
    main()
