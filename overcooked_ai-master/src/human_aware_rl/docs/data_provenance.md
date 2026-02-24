# Data Provenance

This document records provenance requirements for human-data-dependent paper
reproduction (especially Figures 5-7).

## Data Source Status

Current repository status:

- `src/human_aware_rl/static/human_data/` contains documentation and dummy CSVs.
- Full paper-grade human trial artifacts are not bundled in this repository.
- Legacy notebook references indicate historical local paths and MTurk-derived
  records; those must be re-supplied for exact reproduction.

## Expected Data Inputs

Required datasets include:

- Human-human trial trajectories
- Human-AI trial trajectories
- Processed outputs used to build:
  - `humanai_performance`
  - `hh_performance`
  - held-out trajectories for off-distribution loss/accuracy analyses

Schema reference for raw/processed columns is documented in:

- `src/human_aware_rl/static/human_data/README.md`

## Preprocessing Pipeline

Primary preprocessing code:

- `src/human_aware_rl/human/process_dataframes.py`
  - `csv_to_df_pickle(...)`
  - `format_trials_df(...)`
  - `train_test_split(...)` (default 70/30 split)
- `src/human_aware_rl/human/data_processing_utils.py`
- `src/human_aware_rl/human/process_human_trials.py` (legacy forward-port logic)

## Reproducibility Metadata To Capture

For every reproduction run, record:

- Data source URI or transfer location
- File hashes (SHA256) for each raw and processed file
- Preprocessing code commit SHA
- Exact preprocessing parameters and filters
- Train/test split policy and random seed

Suggested metadata table:

| Artifact | Path | SHA256 | Source URI | Notes |
| --- | --- | --- | --- | --- |
| human_human_raw | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| human_ai_raw | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| clean_hh_trials | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| clean_hai_trials | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## Figure Dependency Notes

- Figure 4 can be reproduced from model checkpoints and evaluator outputs.
- Figures 5-7 require real human-study artifacts; dummy files are insufficient.
- If canonical human data is unavailable, document the gap explicitly and treat
  Figures 5-7 as blocked in acceptance reporting.
