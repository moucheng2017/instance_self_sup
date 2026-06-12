# Changelog - Branch 59

Branch: `59-unbalanced-ot`

Base: `58-nmi-metrics` (per-level purity/NMI tree-structure diagnostics —
the instrument that judges this branch's change).

Baseline comparison: branch-58 behavior plus semi-relaxed unbalanced OT for
the 2-way node splits from this branch.

## Motivation

The 50/50 balance constraint exists for collapse prevention, but natural
semantic hierarchies are not balanced (CIFAR-10's animals/vehicles is 6/4).
The hard constraint forces misassignment at every level to satisfy counting,
distorting the discovered structure — the core tension recorded in
`.agents/designs/natural_splits_ratios_strategy.md`. This branch implements
that document's Approach 1 (semi-relaxed unbalanced OT) with tau-annealing:
start nearly balanced while features are noisy, relax toward natural split
ratios as they consolidate. Collapse prevention moves from a global hard
constraint to a local, explicit guard.

## Added

- Added `_unbalanced_sinkhorn(scores, tau, num_iters=None, target=None)`
  (semi-relaxed scaling form, Chizat et al. 2018): the sample marginal stays
  hard (every returned row sums to 1); the component marginal is KL-penalized
  with strength `tau`, applied as the partial correction
  `q_k <- q_k * (target_k / mass_k)^phi` with `phi = tau / (tau + ot_epsilon)`
  (`tau` shares `ot_epsilon`'s cost units). `tau -> inf` recovers `_sinkhorn`
  to machine precision; `tau << ot_epsilon` lets component masses follow the
  data's natural ratio. `_sinkhorn_2way` dispatches on `ot_unbalanced_tau`.
- Added unbalanced hard splits in `_fit_binary_ot`: `argmax(q)` so the
  transport's natural ratio survives (a median split would silently re-impose
  balance), guarded by `ot_unbalanced_min_split_fraction` (default 0.05) with
  per-node fallback to the balanced median split.
- Added model knobs `ot_unbalanced_tau` (null = legacy exact balance) and
  `ot_unbalanced_min_split_fraction`, with constructor validation and a
  validated `set_ot_unbalanced_tau` setter; wired through `models/__init__.py`.
- Added `compute_unbalanced_tau(epoch, tau_start, tau_final, anneal_epochs)`
  (pure cosine schedule) and trainer hook `maybe_update_unbalanced_tau`,
  driven by `train.unbalanced_tau_start` / `unbalanced_tau_final` /
  `unbalanced_tau_anneal_epochs` (disabled when `unbalanced_tau_final` is
  null). The live tau is logged per epoch.
- Added observability: per-node minority-child fractions reported as
  `tree_min_split_fraction` / `tree_mean_split_fraction` in the tree build
  stats (0.5 = balanced; lower = natural ratios emerging), plotted in the
  tree-health SVG together with the tau curve.
- Added `configs/hierarchical_unbalanced_vmf_cifar_colab.yaml` (tau schedule
  5.0 -> 0.02 over 400 epochs, `tree_warm_start: True`) and
  `notebooks/hierarchical-unbalanced-vmf-cifar10-ssl.ipynb` mirroring the
  balanced notebook's structure (clone branch `59-unbalanced-ot`).
- Added `tests/test_unbalanced_ot.py` (15 tests): tau schedule endpoints/
  midpoint/validation, balanced-limit recovery, natural-ratio recovery on a
  12/4 fixture, row-stochasticity across taus, unbalanced vs legacy hard
  splits, min-split-fraction guard, split-fraction stats, setter/constructor
  validation, forward/backward, and argument-passing checks for `get_model`,
  both configs, and the new notebook.

## Changed

- `configs/hierarchical_balanced_vmf_cifar_colab.yaml` documents the new keys
  with null defaults — balanced behavior is bit-for-bit unchanged when they
  are absent or null.
- Tree-health SVG panel titles/keys: sample-accuracy panel also shows split
  fractions (same [0, 1] scale); the annealing/re-seeding panel also shows
  `ot_unbalanced_tau`.
- Flat modes (`flat_ot`, `flat_vmf_ot`, `vmf`) deliberately keep hard balance:
  the motivation is tree-structure discovery, and per-node relaxation is only
  meaningful in the recursive formulation.

## Empirical calibration (validated numerically in-sandbox)

The balanced->natural transition is sharp and sits at `tau ~ ot_epsilon`:
with eps=0.1 on a 12/4 two-cluster fixture, component masses stay 8.00/8.00
down to tau=0.3, soften at tau=0.05 (10.65/5.35), and reach the natural
12.00/4.00 by tau=0.01; tau=1e6 matches the balanced Sinkhorn to 4e-16.
Schedule defaults follow: start at ~100x eps (effectively balanced), end
below eps/2.

## Validation

- Sandbox-verified, executed for real: the full scaling iteration as a
  pure-Python mirror (all behaviors above); every assertion of
  `test_compute_unbalanced_tau_cosine_anneal` against the real function (AST
  extraction); both config parses incl. legacy-null defaults; notebook token
  and cell-structure parity checks; the tree-health SVG rendering with tau and
  split-fraction series; `python -m py_compile` on all touched files.
- Pending locally (torch required):

```bash
PYTHONPATH=/Users/xmc28/Desktop/projects/code/my_simsiam conda run -n ssl_local pytest -q tests/test_unbalanced_ot.py
```

Expected: `15 passed`. (`test_get_model_passes_unbalanced_args` builds a
resnet18 without pretrained weights; no download or GPU needed.)

## Reading the results

Judge this branch by branch-58's panels: per-level purity/NMI for the
balanced run vs the unbalanced run, alongside `tree_mean_split_fraction`
drifting below 0.5 as tau anneals. The motivating prediction: NMI at coarse
levels rises when the root split is allowed to find ~60/40 (animals/vehicles)
instead of forcing 50/50. If split fractions pin at the guard floor or leaf
occupancy collapses, raise `unbalanced_tau_final` (e.g. 0.02 -> 0.05).
