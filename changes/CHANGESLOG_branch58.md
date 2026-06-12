# Changelog - Branch 58

Branch: `58-nmi-metrics`

Base: `57-selective-reseeding` (tree warm start, node-health bookkeeping,
selective re-seeding, monitoring SVGs).

Baseline comparison: branch-57 behavior plus tree-structure-vs-ground-truth
diagnostics from this branch.

## Motivation

The project's goal is discovering latent semantic structure, but nothing in the
existing monitoring measures whether the discovered tree aligns with real
classes — learnability, balance, and kNN are all label-structure-blind. This
branch adds per-level purity and NMI between tree path prefixes and
ground-truth labels, turning "the hierarchy preserves semantic meaning" from a
hope into a measured quantity. Diagnostic only: labels never feed training.

## Added

- Added `tools/tree_metrics.py` (torch-free):
  - `prefix_cluster_ids(paths, level)`: integer cluster ids from the first
    `level + 1` branch decisions; level numbering matches the node-accuracy
    stats (level 0 = root decision, 2 clusters; level l = 2^(l+1) clusters)
  - `purity(clusters, labels)`: majority-label purity
  - `nmi(clusters, labels)`: normalized mutual information with
    arithmetic-mean normalization (sklearn default); returns 0.0 for constant
    inputs; clamped at 0
  - `prefix_label_metrics(paths, labels)`: per-level
    `tree_purity_level{l}` / `tree_nmi_level{l}` dict (rounded to 4 dp);
    returns {} on unusable inputs
  - `source_pool_true_labels(dataset, source_indices)`: ground-truth labels
    per source-pool position; unwraps Subset-like wrappers, supports
    `.targets` (CIFAR) and `.labels` (STL10); returns None when unavailable
    instead of raising
- Added structure diagnostics to the full-pool refresh in `main.py`:
  after `refresh_assignments(...)`, hierarchical modes compute
  `prefix_label_metrics` over the stored `assignment_paths` and merge them
  into the refresh stats — so they flow into the epoch logger, the printed
  refresh summary, and the tree-health SVG with no further wiring.
- Added two panels to `monitor_tree_health.svg` (grid 2x2 -> 3x2): tree
  purity per level and tree NMI per level vs ground-truth labels. Reading
  guide: purity rises with depth by construction (more, smaller clusters);
  NMI peaks where tree granularity matches label granularity — for CIFAR-10
  with depth 6, expect the peak around levels 3-4 if the discovered structure
  is semantic.
- Added `tests/test_tree_metrics.py`: prefix-id construction, purity
  (perfect/mixed/empty), NMI (perfect, independent, constant, hand-computed
  value), per-level metrics keys + over-split behavior, and label extraction
  through direct datasets and Subset wrappers.
- Extended the fake history in `tests/test_monitor_plots.py` with purity/NMI
  series.

## Changed

- `tools/monitor_plots.py`: `_level_acc_keys` generalized to
  `_level_keys(history, prefix)`; tree-health figure is now 3x2.
- No training-behavior changes: metrics are computed under the existing
  no-grad refresh pass and add one O(pool x depth) pure-Python pass per
  refresh (~0.3 s for 50k x 6, once per epoch).
- Batch-local mode (`batch_self_labeling=True`) has no full-pool refresh and
  therefore no purity/NMI series; the panels show "no data yet".

## Validation

- Sandbox-verified, executed for real (module and tests are torch-free):
  all 6 `tests/test_tree_metrics.py` tests ran and passed; the tree-health
  SVG renders with the two new panels populated and tolerates empty history;
  `python -m py_compile` on all touched files.
- Pending locally (torch required only for the end-to-end refresh path):

```bash
PYTHONPATH=/Users/xmc28/Desktop/projects/code/my_simsiam conda run -n ssl_local pytest -q tests/test_tree_metrics.py tests/test_monitor_plots.py
```

Expected: `9 passed` (6 metrics + 3 monitor plots; both files run without a
GPU and without torchvision datasets).

## Reading the new panels (first calibration run)

1. `tree_nmi_level0` is the headline number: does the root split carry label
   information at all? Rising from ~0 = the coarse structure is becoming
   semantic; flat at ~0 with high branch accuracy = the tree found a
   consistent but non-semantic partition (the confirmation-bias signature).
2. Purity at the deepest level vs `1/num_classes`-dominated baseline shows
   over-clustering quality; compare against a `flat_vmf_ot` run at K=2^depth
   for the hierarchical-vs-flat ablation.
