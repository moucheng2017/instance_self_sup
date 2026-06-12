# Changelog - Branch 61

Branch: `61-recursive-ot`

Base: `59-unbalanced-ot` (semi-relaxed unbalanced OT with tau annealing).

Baseline comparison: branch-59 behavior (binary unbalanced vMF/OT tree) plus a
new, additive K-way rank-annealing model from this branch. No binary code path
changed.

## Motivation

The binary model fixes the branching factor at 2 for every level, but the
branching factor is the single most important prior on the partition: it
decides how many sibling concepts a node may resolve into before any data is
seen. *Hierarchical Refinement: Optimal Transport to Infinity and Beyond*
(Halmos, Gold, Liu, Raphael; ICML 2025; "HiRef") supplies a principled
per-level prior: a rank-annealing schedule `(r_1, ..., r_kappa)` chosen by a
dynamic program (coarse small ranks early, larger ranks later), with each
refinement step solving a balanced low-rank OT subproblem whose uniform inner
marginal `g = (1/r) 1_r` makes the optimal factor a balanced hard partition
(`argmax` Assign rule, HiRef Prop 3.1 / Algorithm 1).

This branch ports the HiRef splitting prior and Assign rule, and replaces
HiRef's Euclidean low-rank OT with vMF fitting on the unit sphere (cosine
cost, normalized mean directions), since our points are L2-normalized backbone
features. The tree is built toward fine resolution, but only the first
`supervised_depth` (early, stable) levels emit pseudo-labels to supervise the
backbone, per the "early stable levels" rule. Full design and HiRef->ours
mapping table: `.agents/designs/hierarchical_kway_vmf_ot_design.md`.

## Added

- Added `tools/rank_annealing.py` (torch-free): `optimal_rank_schedule(n,
  depth, max_rank, base_rank)` — the HiRef E.1 dynamic program minimizing the
  sum of partial products `sum_t prod_{s<=t} r_s` subject to
  `prod_t r_t = n / base_rank` and `r_t <= max_rank` — plus
  `partial_product_sum` and mixed-radix tree index helpers (`level_sizes`,
  `level_offsets`, `num_internal_nodes`, `child_global_id`, `leaf_index`)
  generalizing the binary heap `2j+1 / 2j+2` to per-level branching.
- Added `models/hierarchical_kway_vmf_ot_self_labeling_net.py` with
  `HierarchicalKWayVMFOTSelfLabelingNet` and `resolve_rank_schedule`:
  - K-way recursive tree build (`_build_kway_tree`): per level `t`, every
    node with `>= r_t` points is split into `r_t` children by
    `_fit_kway_vmf_ot` — spherical farthest-point cold seeding (or automatic
    per-node warm start via the `node_fitted` gate, with a degenerate-seed
    cold fallback), EM over cosine-cost Sinkhorn with vMF mean-direction
    updates, and `argmax(q)` hard labels (HiRef Assign; no median split).
  - Balanced mode uses the uniform component marginal `1/r_t` (HiRef's
    uniform `g`); unbalanced mode reuses branch 59's semi-relaxed scaling
    form with `phi = tau / (tau + ot_epsilon)`, plus a K-way collapse guard:
    a node whose smallest child would fall below
    `ot_unbalanced_min_split_fraction` is refit balanced (local, explicit).
  - Schedule knobs: explicit `rank_schedule` list, or null to derive it from
    `num_leaf_clusters` / `rank_schedule_depth` / `rank_schedule_max_rank` /
    `rank_schedule_base_rank` via the DP. Depth is a user budget, not
    auto-minimized, so derived schedules anneal coarse -> fine (e.g. 256
    leaves at depth 4 -> `[2, 2, 4, 16]`).
  - `supervised_depth`: early-levels pseudo-label cutoff. Deeper levels are
    built but produce no gradient; `set_active_depth` clamps to it, so the
    branch-56 staircase depth annealing composes with the cutoff.
  - Padded prototype store `[n_internal, max_rank, embedding_dim]`; per-node
    accuracy bookkeeping + `finalize_node_stats()` (per-level chance is
    `1/r_t`); optional prototype EMA; optional learnable prototypes; optional
    sigmoid image-index regularization; `refresh_assignments` full-pool path;
    `set_ot_unbalanced_tau` for the branch-59 tau schedule. All trainer hooks
    are duck-typed by `main.py` with no trainer logic changes.
- Added `configs/hierarchical_kway_vmf_ot_cifar_colab.yaml`
  (`rank_schedule: [2, 4, 4, 8]`, `supervised_depth: 3`, `sinkhorn_iters: 15`,
  `prototype_ema_momentum: 0.9`, tau schedule 5.0 -> 0.02 over 400 epochs) and
  `notebooks/hierarchical-kway-vmf-ot-cifar10-ssl.ipynb` mirroring the
  unbalanced notebook's structure with the schedule knobs exposed.
- Added `tests/test_rank_annealing.py` (8 tests, torch-free): DP optimality
  vs brute force, factorization/max-rank feasibility, infeasible-n raise,
  base-rank reduction, partial-product level sizes/offsets, binary-heap
  equivalence for an all-2 schedule, K-way child ids, mixed-radix leaf index.
- Added `tests/test_hierarchical_kway_vmf_ot.py` (11 tests, needs torch):
  explicit-schedule validation, DP-derived schedule wiring, depth /
  supervised-depth clamping, node-level buffer layout, balanced K-way leaf
  occupancy on a 27-point `[3,3,3]` fixture, small-node level masking,
  forward/backward with nonzero backbone grads, supervised-depth loss cap,
  unbalanced collapse guard, non-batch-local stored assignments, node-stats
  finalize/reset.
- Added per-level purity/NMI tree-structure diagnostics, end to end:
  - `tools/tree_metrics.py`: `prefix_cluster_ids` / `prefix_label_metrics`
    gained an optional `radices` argument (mixed-radix prefix encoding for
    K-way rank schedules; default stays binary, fully backward compatible).
  - `HierarchicalKWayVMFOTSelfLabelingNet.predict_paths(embeddings)`:
    read-only root-to-leaf argmax descent through the stored prototypes —
    scores the partition the prototype hierarchy predicts without mutating
    any tree state (the builders fit and mark `node_fitted`; this does not).
  - `main.py::maybe_compute_tree_structure_metrics`: once every
    `train.tree_metrics_interval` epochs (0/null disables), encodes up to
    `train.tree_metrics_max_samples` clean memory-loader images, predicts
    paths, and computes `tree_purity_level{t}` / `tree_nmi_level{t}` against
    ground-truth labels. Duck-typed on `predict_paths`; diagnostic only.
  - `tools/monitor_plots.py`: new standalone `monitor_tree_structure.svg`
    (`save_tree_structure_monitor_svg`) with one full-width panel each for
    per-level purity and per-level NMI vs true labels — standalone so trees
    with many levels stay readable; the tree-health SVG keeps its 2x2 layout.
    Written only when `tree_purity_level*` keys exist in the history.
  - Config/notebook expose `tree_metrics_interval: 1` and
    `tree_metrics_max_samples: 10000`.
  - Tests: mixed-radix prefix ids and metrics (torch-free, in
    `tests/test_tree_metrics.py`), `predict_paths` read-only/determinism and
    geometry-following descent (torch, in
    `tests/test_hierarchical_kway_vmf_ot.py`).
- Added `.agents/designs/hierarchical_kway_vmf_ot_design.md` (design + review:
  HiRef->ours mapping, what carries over vs what changes, deferred items) and
  `.agents/hierarchical_kway_vmf_ot_self_labeling.txt` (dataflow workflow
  walk-through mirroring the balanced one).

## Changed

- `models/__init__.py`: import + `get_model` registry branch for
  `hierarchical_kway_vmf_ot_self_labeling_net`, passing the schedule, cutoff,
  OT, prototype, and sigmoid knobs from the config.
- `main.py::build_train_loader`: added the new model name to the
  pseudo-supervised source-pool dataset list.
- `main.py` epoch loop: calls `maybe_compute_tree_structure_metrics` after
  the kNN monitor and folds the purity/NMI scalars into the tree stats /
  history (they flow into the SVGs and the logger automatically).
- `.agents/codebase_structure.md`: branch-61 sections (new model, tools,
  config, notebook, tests, edit map, gotchas) and fixed `.agents/designs/`
  paths for the moved design docs.

## Explicitly Not Done (deferred)

- Full instance-level recursion (`base_rank = 1` to a bijection): for
  pseudo-labels only the coarse tree is supervised; the schedule already
  supports it if needed later.
- Selective re-seeding for the K-way node space (branch-57 machinery is
  binary-only); revisit if K-way node accuracies show lock-in.
- Per-node tau dual ascent; sample-linear cost factorization (Indyk et al.
  2019) — unnecessary at batch/coarse-pool scale.

## Verification

- `tests/test_rank_annealing.py`: 8/8 pass (torch-free, run locally).
- `python -m py_compile` clean on all touched files; YAML parses; DP wiring
  checked for depths 3-5 (products and partial sums verified).
- `tests/test_hierarchical_kway_vmf_ot.py` requires torch — run on Colab/GPU
  with `PYTHONPATH="$PWD" pytest -q tests/test_hierarchical_kway_vmf_ot.py`.

## Observability

- Build stats reuse the branch-59 names (`tree_nonempty_leaves`,
  `tree_min/max_leaf_count`, `tree_min/mean_split_fraction`) — note balanced
  K-way nodes sit at `1/r_t`, not 0.5, in the split-fraction panels.
- `finalize_node_stats` reports `tree_node_acc_level{t}` per level; chance is
  `1/r_t` per level.
- `tree_purity_level{t}` / `tree_nmi_level{t}` per epoch (predicted-path
  prefixes vs true labels over the memory pool), plotted in the standalone
  `monitor_tree_structure.svg` (one full-width panel per metric).

## How to judge the change

Compare against the branch-59 binary unbalanced run at matched leaf count
(e.g. binary depth 8 = 256 leaves vs `[2, 4, 4, 8]` = 256 leaves) on: kNN
accuracy, per-level node accuracy, and the per-level purity/NMI panels in
`monitor_tree_structure.svg`. The hypothesis: a 4-way or 8-way split at the
level where CIFAR-10's natural multi-way structure lives should show higher
level purity/NMI than a chain of forced bisections. Caveat: purity rises
mechanically with cluster count, so compare a level mostly against its own
history and across runs at matched prefix-cluster counts; NMI is the
fairer cross-level number.
