# Reproduction Runbook (Figure 4-7)

## Prerequisites

- Install module dependencies (JAX/PyTorch stack as documented in module README).
- Ensure human data provenance is recorded per `data_provenance.md`.
- Work from a clean, known commit SHA when generating final artifacts.

## Canonical End-to-End Command

```bash
python -m human_aware_rl.evaluation.run_paper_evaluation \
  --all \
  --paper_strict \
  --train_missing \
  --seeds 0,10,20,30,40 \
  --results_file paper_results.json \
  --output_dir figures
```

Behavior:
- trains missing BC/HP and PPO_SP/PPO_BC/PPO_HP artifacts under canonical run registry
- evaluates Figure 4 configs in strict mode
- exports `paper_results.json` and Figure 4 plots

## Strict Evaluation Only (No Training)

```bash
python -m human_aware_rl.evaluation.run_paper_evaluation \
  --all \
  --paper_strict \
  --check_only
```

Use this to validate artifact presence before long evaluation jobs.

## Figure 5-7 Pipeline Entry

For Figure 5-7 artifact generation, use the unified reproduction entry points:

```bash
python -m human_aware_rl.reproduce.paper_reproduce train
python -m human_aware_rl.reproduce.paper_reproduce eval
python -m human_aware_rl.reproduce.paper_reproduce export
python -m human_aware_rl.reproduce.paper_reproduce plot
```

## Canonical Output Contract

- PPO runs: `DATA_DIR/ppo_runs/<run>/seed<seed>/<agent>/checkpoint_*`
- Evaluation JSON: `paper_results.json` (or configured path)
- Figures: `<output_dir>/`
- Run provenance: `run_manifest.json` in training outputs

## Common Failure Modes

- `FileNotFoundError` in strict mode:
  - verify expected canonical run/seed/agent directory exists
  - do not rely on `results/*` fallback artifacts
- strict mode override errors:
  - remove non-canonical run-template/agent-dir overrides
  - keep strict defaults for paper runs
- paper mode training errors:
  - remove ablation flags (`--fast`, `--local`, custom timesteps) or switch to `--not_paper`
