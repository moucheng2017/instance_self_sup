# Changelog - Branch 57

Branch: `57-selective-reseeding`

Base: `56-depth-annealing` (adds staircase depth annealing:
`train.depth_annealing_epochs_per_level` activates one extra supervised tree
level every N epochs via `set_active_depth`; levels beyond the active depth are
built but produce no loss).

Baseline comparison: branch-56 hierarchical balanced vMF self-labeling behavior
plus warm-started tree EM, node-health bookkeeping, selective re-seeding, and
epoch-level monitoring SVGs from this branch.

## Added

- Added `tree_warm_start` (model config, default `False`) for temporal
  continuity of the pseudo-label tree:
  - `_fit_binary_ot` accepts `init_mu`; warm-started nodes initialize their
    2-way Sinkhorn/vMF EM from the node's stored prototypes instead of
    re-deriving farthest-point seeds from the data at every refresh/batch build
  - new `node_fitted` buffer; never-fitted or degenerate nodes still cold-seed
- Added per-node health bookkeeping (logging stage, no tree mutation):
  - `_forward_embeddings` scatter-adds per-node branch correct/count during the
    prediction loss (active depth-annealing levels only, by construction)
  - `finalize_node_stats()` (called once per epoch by the trainer) reports
    `tree_node_acc_overall`, per-level `tree_node_acc_level{i}`,
    `tree_nodes_visited`, and `tree_reseed_candidates`; updates low-accuracy
    streaks against `reseed_acc_threshold` with a `reseed_min_node_samples`
    visit floor; resets accumulators
  - new buffers: `node_correct_sum`, `node_count`, `node_low_acc_streak`,
    `node_last_acc`, `node_levels`
- Added selective re-seeding (live stage, default off):
  - `select_reseed_nodes()` cold-restarts persistently unlearnable nodes:
    candidates with streak >= `reseed_patience`, worst last-accuracy first,
    capped at `reseed_budget_fraction` of measured nodes per epoch
  - a selected node and its whole subtree get `node_fitted=False` (next build
    re-derives them from data) and their stats/streaks reset
  - `reseed_enabled=True` requires `tree_warm_start=True` and a non-null
    `reseed_acc_threshold` (constructor `ValueError` otherwise)
  - pure helpers `select_reseed_indices` / `_subtree_node_ids` for isolated
    testing of the budget and subtree logic
- Added `tools/monitor_plots.py` (torch-free) and trainer wiring that refreshes
  two SVGs in the run's log dir every epoch (non-fatal on plot errors):
  - `monitor_training.svg`: learning rate, all loss components, kNN accuracy
  - `monitor_tree_health.svg`: per-level + overall node branch accuracy against
    the 0.5 chance line, sample-level path/branch accuracy, depth-annealing
    progress and re-seeding activity, leaf balance / tree occupancy
- Added model config keys to
  `configs/hierarchical_balanced_vmf_cifar_colab.yaml` (all default-off):
  `tree_warm_start`, `reseed_acc_threshold`, `reseed_patience`,
  `reseed_min_node_samples`, `reseed_enabled`, `reseed_budget_fraction`.
- Added `.agents/design_choice_depth_annealing_tree_reseeding.md` documenting
  the failure-mode-to-mechanism mapping, trigger/decision-rule rationale,
  mechanism interactions, and the monitoring/diagnosis order.
- Added tests: warm-start polarity/degenerate-fallback/node-marking, node-stat
  accumulation and per-level finalize, streak increment/reset/min-sample
  gating, re-seed selection order and budget, subtree coverage, constructor
  validation, subtree cold-restart integration
  (`tests/test_hierarchical_balanced_vmf_self_labeling.py`), and SVG writers
  including empty/missing-series tolerance (`tests/test_monitor_plots.py`).

## Changed

- Updated `notebooks/hierarchical-balanced-vmf-cifar10-ssl.ipynb`:
  - clone branch is now `57-selective-reseeding`
  - new override variables for depth annealing
    (`depth_annealing_epochs_per_level = 100`, `depth_annealing_initial_depth = 1`),
    warm start (`tree_warm_start = True`), and re-seeding (calibration defaults:
    `reseed_acc_threshold = 0.65`, `reseed_patience = 3`,
    `reseed_min_node_samples = 64`, `reseed_enabled = False`,
    `reseed_budget_fraction = 0.1`), all wired into the `overrides` dict and the
    experiment tag
- `maybe_refresh_hierarchical_assignments` now returns the refresh stats dict
  (used by the tree-health SVG); behavior otherwise unchanged.
- The trainer logs tree-health scalars into the epoch logger (per-level node
  accuracies, `tree_reseed_candidates`, `tree_reseeded_nodes`) and prints a
  one-line health summary per epoch plus a notice when re-seeding fires.
- The trainer accumulates per-epoch means of all batch scalars and appends
  them, with refresh and tree-health stats, to an epoch history consumed by the
  two monitoring SVGs.
- State-dict note: the new registered buffers change the model's state-dict
  keys; backbone-only checkpoint loading (`load_backbone_weights`) is
  unaffected.

## Validation

- Sandbox-verified (no torch/pytest available there):

```bash
python -m compileall models/hierarchical_balanced_vmf_self_labeling_net.py models/__init__.py main.py tools/monitor_plots.py tests/
```

- Pure helpers `select_reseed_indices` / `_subtree_node_ids` executed against
  the full test assertion set (patience filtering, worst-first ordering, budget
  caps, subtree coverage at depths 2/3): all passed.
- `tools/monitor_plots.py` executed for real: both SVGs render with a fake
  5-epoch history including a missing series, and with empty history.
- Config YAML parse-checked; new keys default to disabled.
- Pending locally (torch required):

```bash
PYTHONPATH=/Users/xmc28/Desktop/projects/code/my_simsiam conda run -n ssl_local pytest -q tests/test_hierarchical_balanced_vmf_self_labeling.py tests/test_monitor_plots.py
```

Expected: `27 passed` (24 in the vMF self-labeling file: 11 pre-existing,
4 from branch 56 depth annealing, 9 from this branch; 3 in monitor plots).

## Suggested rollout

1. Calibrate: `tree_warm_start: True`, `reseed_acc_threshold: 0.65`,
   `reseed_enabled: False`, depth annealing on — inspect the per-level panel of
   `monitor_tree_health.svg`; the node-accuracy distribution should be bimodal
   (learnable vs. augmentation-unstable cuts).
2. Go live: set the threshold in the gap, flip `reseed_enabled: True`, keep
   `reseed_budget_fraction: 0.1`.
