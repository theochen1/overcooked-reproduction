# Reproduction Acceptance Criteria

This document defines quantitative acceptance criteria for reproducing Figures
4-7 with the JAX/PyTorch pipeline.

## Global Protocol

- Layouts: `cramped_room`, `asymmetric_advantages`, `coordination_ring`,
  `forced_coordination`, `counter_circuit`
- Seeds: `0, 10, 20, 30, 40`
- Agent order: evaluate both `order_0` and `order_1` (switched indices)
- Reporting: mean and standard error over the same seed/layout scope

## Figure 4 Acceptance

For each layout and condition in Figure 4(a)/(b):

- **Mean tolerance**: reproduced mean reward must be within max(`2 * stderr_ref`,
  `15 reward points`) of the paper reference value.
- **Trend/order tolerance**: relative ranking of primary methods must match:
  - Figure 4(a): `PPO_HP+HP` >= `PPO_BC+HP` >= `BC+HP` and `SP+HP` > `SP+SP`
  - Figure 4(b): `PPO_HP+HP` >= `PPO_BC+HP` >= `BC+HP` and `PBT+HP` > `PBT+PBT`
- **Error-bar overlap check**: reproduced 95% CI should overlap the reference CI
  for at least 80% of layout-condition cells.

## Figure 5 Acceptance (Planning Comparison)

For planning bars (for available layouts and data):

- Mean tolerance: within max(`2 * stderr_ref`, `20 reward points`)
- Ordering must hold:
  - `P_HProxy + HProxy` is the upper-bound reference line
  - `CP+CP` above `CP+HProxy` on layouts where planner assumptions hold
  - `P_BC+HProxy` and `BC+HProxy` preserve the paper trend direction

## Figure 6 Acceptance (Off-Distribution Loss)

For cross-entropy loss on held-out human trajectories:

- Mean tolerance: within `10%` relative error or `0.1` absolute loss
  (whichever is larger)
- Trend/order: lower is better; the relative order of methods per layout must
  match the original figure.

## Figure 7 Acceptance (Off-Distribution Accuracy)

For held-out action prediction accuracy:

- Mean tolerance: within `5` absolute percentage points
- Trend/order: higher is better; per-layout method ordering must match
  reference trends.

## Missing-Data Policy

- Figures 5-7 depend on human-study artifacts (`humanai_performance`,
  `hh_performance`, and derived trajectory sets).
- If only dummy data is available, the run is considered:
  - **Figure 4**: evaluable
  - **Figures 5-7**: blocked pending real data provenance confirmation
