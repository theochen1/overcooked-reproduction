"""Unified paper reproduction runner with train/eval/export/plot subcommands."""

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.evaluation.run_paper_evaluation import run_evaluation
from human_aware_rl.imitation.train_bc_models import train_all_models
from human_aware_rl.ppo.configs.paper_configs import PAPER_LAYOUTS
from human_aware_rl.ppo.train_ppo_bc import train_all_layouts as train_ppo_bc_all_layouts
from human_aware_rl.ppo.train_ppo_hp import train_all_layouts as train_ppo_hp_all_layouts
from human_aware_rl.ppo.train_ppo_sp import train_all_layouts as train_ppo_sp_all_layouts
from human_aware_rl.visualization.plot_paper_figures import (
    plot_figure_5,
    plot_figure_6,
    plot_figure_7,
)
from human_aware_rl.visualization.plot_results import plot_figure_4


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()[:12]
    except Exception:
        return "unknown"


def default_bundle_dir() -> str:
    date_tag = datetime.utcnow().strftime("%Y%m%d")
    return os.path.join(DATA_DIR, "paper_reproduction", _git_sha(), date_tag)


def _flatten_figure_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for figure_key in ("figure_4a", "figure_4b"):
        fig_data = results.get(figure_key, {})
        for layout, layout_payload in fig_data.items():
            for config_name, config_payload in layout_payload.items():
                for order_key in ("order_0", "order_1"):
                    order_payload = config_payload.get(order_key, {})
                    if "error" in order_payload:
                        rows.append(
                            {
                                "figure": figure_key,
                                "layout": layout,
                                "config": config_name,
                                "order": order_key,
                                "error": order_payload.get("error"),
                            }
                        )
                        continue
                    rows.append(
                        {
                            "figure": figure_key,
                            "layout": layout,
                            "config": config_name,
                            "order": order_key,
                            "mean_reward": order_payload.get("mean_reward"),
                            "std_reward": order_payload.get("std_reward"),
                            "stderr_reward": order_payload.get("stderr_reward"),
                            "num_games": order_payload.get("num_games"),
                            "num_seeds": order_payload.get("num_seeds"),
                        }
                    )
    return rows


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def command_train(args: argparse.Namespace) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.bundle_dir, exist_ok=True)
    # BC + HP models first (paper prerequisite).
    train_all_models(layouts=PAPER_LAYOUTS, verbose=not args.quiet, evaluate=not args.no_eval_bc)
    # PPO stages.
    train_ppo_sp_all_layouts(
        seeds=seeds,
        layouts=PAPER_LAYOUTS,
        ppo_data_dir=args.ppo_data_dir,
        verbose=not args.quiet,
    )
    train_ppo_bc_all_layouts(
        seeds=seeds,
        layouts=PAPER_LAYOUTS,
        ppo_data_dir=args.ppo_data_dir,
        partner_type="bc_train",
        verbose=not args.quiet,
    )
    train_ppo_hp_all_layouts(
        seeds=seeds,
        layouts=PAPER_LAYOUTS,
        ppo_data_dir=args.ppo_data_dir,
        verbose=not args.quiet,
    )


def command_eval(args: argparse.Namespace) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    results_file = args.results_file or os.path.join(args.bundle_dir, "paper_results.json")
    run_evaluation(
        figure=args.figure,
        layouts=PAPER_LAYOUTS,
        seeds=seeds,
        num_games=args.num_games,
        output_file=results_file,
        paper_strict=True,
        verbose=not args.quiet,
    )


def command_export(args: argparse.Namespace) -> None:
    results_file = args.results_file or os.path.join(args.bundle_dir, "paper_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file, "r") as f:
        results = json.load(f)

    export_dir = os.path.join(args.bundle_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, "paper_results_export.json"), "w") as f:
        json.dump(results, f, indent=2)

    rows = _flatten_figure_rows(results)
    _write_csv(os.path.join(export_dir, "figure4_summary.csv"), rows)

    manifest = {
        "bundle_dir": args.bundle_dir,
        "git_sha": _git_sha(),
        "export_time_utc": datetime.utcnow().isoformat(),
        "results_file": results_file,
        "rows_exported": len(rows),
    }
    with open(os.path.join(export_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def command_plot(args: argparse.Namespace) -> None:
    results_file = args.results_file or os.path.join(args.bundle_dir, "paper_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file, "r") as f:
        results = json.load(f)

    figures_dir = os.path.join(args.bundle_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Figure 4 from evaluation outputs.
    plot_figure_4(results=results, output_dir=figures_dir, fmt=args.format, show=False)

    # Optional Figures 5-7 from exported analysis payloads.
    analysis_file = args.analysis_file or os.path.join(args.bundle_dir, "exports", "analysis_metrics.json")
    planning_file = args.planning_file or os.path.join(args.bundle_dir, "exports", "planning_metrics.json")
    if os.path.exists(planning_file):
        with open(planning_file, "r") as f:
            planning_data = json.load(f)
        plot_figure_5(planning_data, os.path.join(figures_dir, f"figure5.{args.format}"))
    if os.path.exists(analysis_file):
        with open(analysis_file, "r") as f:
            analysis_data = json.load(f)
        plot_figure_6(analysis_data, os.path.join(figures_dir, f"figure6.{args.format}"))
        plot_figure_7(analysis_data, os.path.join(figures_dir, f"figure7.{args.format}"))

    # Overleaf bundle (side-by-side figures + tabular export files).
    overleaf_dir = os.path.join(args.bundle_dir, "overleaf_bundle")
    os.makedirs(overleaf_dir, exist_ok=True)
    for name in os.listdir(figures_dir):
        if name.endswith(f".{args.format}"):
            shutil.copy2(os.path.join(figures_dir, name), os.path.join(overleaf_dir, name))
    exports_dir = os.path.join(args.bundle_dir, "exports")
    if os.path.isdir(exports_dir):
        for name in os.listdir(exports_dir):
            if name.endswith(".csv") or name.endswith(".json"):
                shutil.copy2(os.path.join(exports_dir, name), os.path.join(overleaf_dir, name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified paper reproduction runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle_dir", type=str, default=default_bundle_dir())
    parser.add_argument("--ppo_data_dir", type=str, default=os.path.join(DATA_DIR, "ppo_runs"))
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--seeds", type=str, default="0,10,20,30,40")
    p_train.add_argument("--no_eval_bc", action="store_true")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--seeds", type=str, default="0,10,20,30,40")
    p_eval.add_argument("--num_games", type=int, default=10)
    p_eval.add_argument("--figure", type=str, choices=["4a", "4b", "all"], default="all")

    sub.add_parser("export")

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--format", type=str, choices=["pdf", "png", "svg"], default="pdf")
    p_plot.add_argument("--analysis_file", type=str, default=None)
    p_plot.add_argument("--planning_file", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.bundle_dir, exist_ok=True)
    if args.command == "train":
        command_train(args)
    elif args.command == "eval":
        command_eval(args)
    elif args.command == "export":
        command_export(args)
    elif args.command == "plot":
        command_plot(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

