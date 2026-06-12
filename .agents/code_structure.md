# Notes
This md describes ONLY the current structure of the code base used as a starting point. Might need to be updated later.

# Codebase Structure Map

This note is a compact orientation guide for future agents working on this
repository. The active branch adds a HiRef-primed K-way
rank-annealing variant on top of the hierarchical balanced/unbalanced vMF / OT
self-labeling family, with older SimSiam/SimCLR/BYOL-era code still present as
legacy baselines or placeholders.

## Repository Purpose

The active experiment is:

```text
images
  -> backbone
  -> optional projector
  -> L2-normalized embeddings on the unit sphere
  -> batch-local or full-pool OT self-label tree
  -> pseudo-label prediction loss
  -> optional sigmoid image-index regularization
```

For algorithm notes, also read:

- `.agents/hierarchical_balanced_vmf_self_labeling.txt`
- `.agents/hierarchical_kway_vmf_ot_self_labeling.txt` (K-way rank-annealing
  variant, branch 61)
- `.agents/designs/hierarchical_kway_vmf_ot_design.md` (HiRef -> ours mapping,
  design decisions, deferred items)
- `.agents/designs/design_choice_depth_annealing_tree_reseeding.md` (depth
  annealing, tree warm start, selective re-seeding: failure modes and design
  rationale)
- `.agents/designs/natural_splits_ratios_strategy.md`
- `.agents/designs/OT_design_justifications.md`
- `.agents/sigmoid_pseudo_sup.txt`

Per-branch change logs live in `changes/CHANGESLOG_branch*.md`.

## Top-Level Entry Points

### `main.py`

Primary training entry point for CLI and notebooks.

Important functions:

- `build_train_loader(args)`: builds ordinary SSL loaders or pseudo-supervised
  source-pool loaders.
- `build_eval_loader(args, train)`: builds kNN memory/test loaders.
- `maybe_refresh_hierarchical_assignments(...)`: full-pool tree refresh path for
  hierarchical vMF models when `batch_self_labeling=False`; returns the refresh
  stats dict (leaf balance) consumed by the tree-health SVG.
- `maybe_update_depth_annealing(model, epoch, args)`: staircase depth annealing;
  computes `active_depth` from `train.depth_annealing_epochs_per_level` /
  `train.depth_annealing_initial_depth` via `compute_active_depth` and calls
  `model.set_active_depth(...)` once per epoch.
- `maybe_update_unbalanced_tau(model, epoch, args)`: cosine-anneals the
  unbalanced-OT marginal penalty via `compute_unbalanced_tau` from
  `train.unbalanced_tau_start` to `train.unbalanced_tau_final` over
  `train.unbalanced_tau_anneal_epochs`; disabled when `unbalanced_tau_final`
  is null (the model keeps its constructor tau, or stays exactly balanced).
- `maybe_finalize_tree_node_stats(model)`: folds per-node branch-accuracy
  accumulators into epoch stats (`finalize_node_stats`), after which the train
  loop calls `select_reseed_nodes()` for the selective re-seeding trigger and
  logs the current `ot_unbalanced_tau`.
- `maybe_update_sigmoid_regularization_progress(...)`: linearly ramps the
  auxiliary sigmoid loss weight.
- `maybe_compute_tree_structure_metrics(...)`: per-level purity/NMI between
  predicted tree-path prefixes and true labels over the memory loader
  (`train.tree_metrics_interval` / `train.tree_metrics_max_samples`;
  duck-typed on the model's read-only `predict_paths`). Diagnostic only.
- `train_model(...)`: core train loop, kNN monitor, checkpoint save, optional
  linear eval. Accumulates per-epoch means of all batch scalars into a history
  and refreshes up to three monitoring SVGs in the log dir every epoch
  (`monitor_training.svg`, `monitor_tree_health.svg`,
  `monitor_tree_structure.svg` when purity/NMI keys exist, via
  `tools/monitor_plots.py`; non-fatal on plot errors).

Training expects each model forward to return a dict containing at least
`{"loss": tensor}`. Any other returned scalars are logged.

### `arguments.py`

Loads YAML configs and builds runtime args.

Key behavior:

- `build_args(...)` supports CLI, notebook, and Colab callers.
- Nested YAML dicts are wrapped in a custom `Namespace`.
- Config keys must start with a letter or `_`, and may contain letters, digits,
  or `_` (keys like `beta1` and `beta2` are valid; hyphens are rejected).
- Runtime-derived fields are added:
  - `args.aug_kwargs`
  - `args.dataset_kwargs`
  - `args.dataloader_kwargs`
- CLI defaults can come from `DATA`, `LOG`, and `CHECKPOINT` environment vars.
- `debug=True` forces tiny batch/epoch/subset settings.

### `jupyter_utils.py` and `colab_utils.py`

Notebook-friendly wrappers around `build_args` and `train_model`.

- `jupyter_utils.py`: local/remote Jupyter usage.
- `colab_utils.py`: Google Drive paths, Colab-safe defaults, and train/eval
  helpers.
- Both default to `num_workers: 0`, matplotlib logging, and no DataParallel.

### `linear_eval.py`

Trains a linear classifier on a frozen backbone loaded from a checkpoint.

Important details:

- `load_backbone_weights(...)` handles checkpoints that either include
  `backbone.*` prefixes or are mostly backbone weights already.
- Uses the same dataset and augmentation registry as training.

### `fix_notebook2.py`

Old notebook-generator helper. It still references an older branch and should
not be treated as authoritative for current Colab runs.

## Current Notebooks

### `notebooks/hierarchical-balanced-vmf-cifar10-ssl.ipynb`

Colab notebook for the current main hierarchical balanced vMF experiment with
sigmoid regularization enabled.

Main experiment cell:

- config: `configs/hierarchical_balanced_vmf_cifar_colab.yaml`
- pool: CIFAR-10 train pool, usually `source_pool_size=50000`
- batch size: usually `512`
- embedding dim: usually `256`
- `ot_assignment_mode`: usually `hierarchical_vmf_ot`; the notebook exposes
  all current modes (`hierarchical_vmf_ot`, `hierarchical_vmf`, `recursive_ot`,
  `flat_ot`, `flat_vmf_ot`, `vmf`)
- default notebook depth: often overridden to `6`
- current notebook default: `batch_self_labeling=False` with
  `tree_refresh_interval=1` for full-pool relabeling
- current notebook default prototype memory: `prototype_ema_momentum=None`
- sigmoid regularization enabled by `sigmoid_regularization_weight=0.1`
- sigmoid bias initialized as `-log(batch_size - 1)`
- long runs may override `num_epochs`/`stop_at_epoch` to `800`
- clone branch is `57-selective-reseeding`; the overrides cell also exposes
  depth annealing (`depth_annealing_epochs_per_level`,
  `depth_annealing_initial_depth`), tree warm start (`tree_warm_start`), and
  selective re-seeding (`reseed_acc_threshold`, `reseed_patience`,
  `reseed_min_node_samples`, `reseed_enabled`, `reseed_budget_fraction`),
  shipped at calibration-stage defaults (warm start on, threshold set,
  trigger off)

### `notebooks/hierarchical-kway-vmf-ot-cifar10-ssl.ipynb`

Colab notebook for the HiRef-primed K-way rank-annealing experiment (branch 61)
using `configs/hierarchical_kway_vmf_ot_cifar_colab.yaml` and model
`hierarchical_kway_vmf_ot_self_labeling_net`. Exposes `rank_schedule` (explicit
per-level branching list, or `None` to let the HiRef dynamic program derive it
from `num_leaf_clusters` / `rank_schedule_depth` / `rank_schedule_max_rank`),
`supervised_depth` (early-levels pseudo-label cutoff), and the shared
unbalanced-tau schedule knobs.

### `notebooks/hierarchical-unbalanced-vmf-cifar10-ssl.ipynb`

Colab notebook for the binary unbalanced-OT variant (branch 59) using
`configs/hierarchical_unbalanced_vmf_cifar_colab.yaml`.

### `notebooks/flat-vmf-ot-cifar10-ssl.ipynb`

Colab notebook for the flat vMF OT variant using
`configs/flat_vmf_ot_cifar_colab.yaml`.

### `notebooks/flat-vmf-sigmoid-cifar10-ssl.ipynb`

Colab notebook for the adjacent flat vMF / sigmoid experiment family.

### `notebooks/linear_eval/linear-eval-cifar10.ipynb`

Colab-oriented notebook for standalone CIFAR-10 linear evaluation.

### `notebooks/sigmoid-pseudo-supervised-cifar10-ssl.ipynb`

Adjacent notebook for flat pairwise sigmoid image-index pseudo-supervision using
`models/sigmoid_pseudo_supervised_net.py`.

### `notebooks/low-rank-multitarget-cifar10-sweep.ipynb`

Diagnostic notebook for the target-rank / target-sharing hypothesis. It uses
`configs/low_rank_multitarget_cifar_colab.yaml` and
`models/low_rank_multitarget_pseudo_supervised_net.py` to sweep fixed balanced
latent target rank `K`, train each run for 100 epochs, and run linear eval from
each checkpoint.

### `notebooks/topk-categorical-bottleneck-pic-cifar10-sweep.ipynb`

Diagnostic notebook for the PIC-style categorical bottleneck hypothesis. It uses
`configs/topk_categorical_bottleneck_pic_cifar_colab.yaml` and
`models/topk_categorical_bottleneck_pic_net.py` to sweep latent capacity `C`,
train each run for 100 epochs, and run linear eval from each checkpoint.

## Current Configs

Configs live in `configs/` and a few subdirectories. Current checked-in configs
include:

- `configs/hierarchical_balanced_vmf_cifar_colab.yaml`: main active config.
- `configs/hierarchical_unbalanced_vmf_cifar_colab.yaml`: unbalanced-OT
  variant — same model, `ot_unbalanced_tau` set, `tree_warm_start: True`, and
  the tau schedule (`unbalanced_tau_start: 5.0 -> unbalanced_tau_final: 0.02`
  over 400 epochs); launched by
  `notebooks/hierarchical-unbalanced-vmf-cifar10-ssl.ipynb`.
- `configs/hierarchical_kway_vmf_ot_cifar_colab.yaml`: K-way rank-annealing
  variant (branch 61) — model `hierarchical_kway_vmf_ot_self_labeling_net`,
  `rank_schedule: [2, 4, 4, 8]` (or null for the HiRef DP), `supervised_depth:
  3`, unbalanced tau schedule shared with branch 59; launched by
  `notebooks/hierarchical-kway-vmf-ot-cifar10-ssl.ipynb`.
- `configs/flat_vmf_ot_cifar_colab.yaml`: flat vMF OT config.
- `configs/sigmoid_pseudo_supervised_cifar_colab.yaml`: pairwise sigmoid
  image-index baseline.
- `configs/low_rank_multitarget_cifar_colab.yaml`: fixed balanced
  multi-target latent-label diagnostic; launched by
  `notebooks/low-rank-multitarget-cifar10-sweep.ipynb`.
- `configs/topk_categorical_bottleneck_pic_cifar_colab.yaml`: PIC-style
  instance classifier through a deterministic soft categorical bottleneck;
  launched by `notebooks/topk-categorical-bottleneck-pic-cifar10-sweep.ipynb`.
- `configs/baselines_ssl/`: legacy Colab SimSiam/SimCLR/STL-10 baselines.
- `configs/python_runs/`: local Python-run SimSiam/SimCLR/BYOL configs.
- `configs/linear_evals/`: standalone linear-evaluation configs.
- `configs/pseudo_sup/`: older pseudo-supervised STL-10 config.

`configs/python_runs/byol_cifar.yaml` is a placeholder; `BYOL.__init__`
currently raises `NotImplementedError`, so it is not runnable as-is.

## Data and Augmentation Layer

### `datasets/__init__.py`

Registry for torchvision datasets:

- `mnist`
- `cifar10`
- `cifar100`
- `stl10`
- `stl10_unlabeled`
- `imagenet`
- `random`

Also supports `debug_subset_size`.

### `datasets/pseudo_supervised.py`

`PseudoSupervisedDataset` is used by:

- `sigmoid_pseudo_supervised_net`
- `low_rank_multitarget_pseudo_supervised_net`
- `topk_categorical_bottleneck_pic_net`
- `hierarchical_balanced_vmf_self_labeling_net`
- `hierarchical_kway_vmf_ot_self_labeling_net`
- legacy `pseudo_supervised_net`
- legacy `hierarchical_pseudo_supervised_net`

It builds a fixed source pool by seed, then each `__getitem__` randomly samples
one source image and returns:

```text
image_tensor, pseudo_label
```

where `pseudo_label` is the image's position inside the source pool, not its
ground-truth CIFAR/STL label.

Important fields:

- `source_indices`
- `num_pseudo_classes`
- `samples_per_epoch`
- `augment_probability`

### `datasets/superimpose.py`

Legacy dataset wrapper that samples two source images, averages their transformed
tensors, and returns two pseudo-labels. The corresponding model/config were
removed, so this file is currently orphaned legacy code.

### `augmentations/__init__.py`

Registry for ordinary SSL augmentations:

- `simsiam`
- `byol`
- `simclr`
- eval transforms through `Transform_single`

The pseudo-supervised source-pool models do not use this training registry. They
use `SuperimposeTransform` directly through their dataset wrappers.

### `augmentations/superimpose_aug.py`

Defines `SuperimposeTransform`:

- `clean(image)`: tensor conversion only
- `augment(image)`: crop, flip, color jitter, grayscale, optional blur
- `postprocess(tensor)`: ImageNet normalization

For CIFAR-10 image size 32, Gaussian blur probability is `0.0`.

## Model Registry

### `models/__init__.py`

Defines:

- `get_backbone(backbone, castrate=True)`
- `get_model(model_cfg, num_classes=None)`

Backbone names are resolved through `_BACKBONE_REGISTRY`, not `eval`. `get_backbone`
sets `backbone.output_dim` from `fc.in_features` and replaces `fc` with
`Identity`.

Registered model names:

- `simsiam`
- `byol` (placeholder; constructor raises `NotImplementedError`)
- `simclr`
- `pseudo_supervised_net`
- `hierarchical_pseudo_supervised_net`
- `sigmoid_pseudo_supervised_net`
- `low_rank_multitarget_pseudo_supervised_net`
- `topk_categorical_bottleneck_pic_net`
- `hierarchical_balanced_vmf_self_labeling_net`
- `hierarchical_kway_vmf_ot_self_labeling_net` (branch 61, K-way
  rank-annealing variant)
- `swav` branch exists in `get_model`, but intentionally raises
  `NotImplementedError`

`models/aug_binary.py`, `datasets/superimpose.py`, and a few adjacent legacy
modules remain on disk but no remaining config targets them.

## Current Main Model: Hierarchical Balanced vMF Self-Labeling

File:

```text
models/hierarchical_balanced_vmf_self_labeling_net.py
```

Class:

```text
HierarchicalBalancedVMFSelfLabelingNet
```

### Two Distinct Types of Pseudo-Label

There are two separate pseudo-label concepts in this model. They serve different
losses and come from different sources.

**1. Image-index pseudo-labels** (`pseudo_labels` argument in `forward`)

- Assigned by `PseudoSupervisedDataset`: each image's position in the fixed source
  pool.
- Static throughout training.
- Used by the sigmoid image-index regularization loss.

**2. OT-discovered pseudo-labels** (tree paths or flat cluster assignments)

- Built at runtime from current detached embeddings (`z.detach()`), or refreshed
  periodically over the full source pool.
- Used by the OT prediction loss.

In `forward`, `pseudo_labels` is passed in from the dataloader. OT labels are
computed internally when `batch_self_labeling=True`, or looked up from stored
buffers when `batch_self_labeling=False`. The two label spaces must not be mixed.

### Constructor Knobs

- `num_classes`: source-pool size; required.
- `backbone`: usually `resnet18`.
- `embedding_dim`: optional projection dimension; default is backbone output dim.
- `depth`: binary tree depth; leaves are up to `2 ** depth`.
- `kappa`: prediction-logit sharpness.
- `ot_epsilon`: entropy temperature for OT fitting.
- `sinkhorn_iters`: binary Sinkhorn iterations.
- `em_iters`: alternating assignment/prototype update iterations.
- `ot_assignment_mode`: one of `hierarchical_vmf_ot`, `hierarchical_vmf`,
  `recursive_ot`, `flat_ot`, `flat_vmf_ot`, or `vmf`.
- `flat_vmf_num_components`: optional K for `flat_vmf_ot` / `vmf`; when unset,
  flat modes use `2 ** depth`.
- `batch_self_labeling`: build labels inside minibatch vs use refreshed buffers.
- `learnable_vmf_prototypes`: makes stored prototypes `nn.Parameter`.
- `prototype_ema_momentum`: optional EMA update from batch-fitted prototypes.
- `sigmoid_regularization_weight`: max auxiliary sigmoid loss weight.
- `sigmoid_init_temperature`, `sigmoid_init_bias`: pairwise sigmoid calibration.
- `ot_unbalanced_tau`: None = exact balanced Sinkhorn (legacy). A float
  switches the 2-way node assignments to semi-relaxed unbalanced Sinkhorn
  (sample marginal hard, component marginal KL-penalized with strength tau).
  tau shares `ot_epsilon`'s cost units; the balanced->natural transition
  happens around tau ~ ot_epsilon. Usually driven per epoch by the trainer's
  tau schedule.
- `ot_unbalanced_min_split_fraction`: collapse guard for unbalanced hard
  splits — a node whose argmax split would starve a child below this fraction
  falls back to the balanced median split locally.
- `tree_warm_start`: initialize each node's 2-way EM from the node's stored
  prototypes (temporal continuity) instead of farthest-point data seeds;
  never-fitted or degenerate nodes still cold-seed. Default False.
- `reseed_acc_threshold`, `reseed_patience`, `reseed_min_node_samples`:
  per-node branch-accuracy streak tracking (null threshold disables).
- `reseed_enabled`, `reseed_budget_fraction`: live selective re-seeding
  trigger; requires `tree_warm_start=True` and a non-null threshold
  (constructor `ValueError` otherwise). Default off.
- Batch-local warm start guard: `batch_self_labeling=True` with
  `tree_warm_start=True`, `learnable_vmf_prototypes=False`, and
  `prototype_ema_momentum=None` raises, because fitted batch prototypes would
  not be persisted between batches.

### Main Forward Flow

```text
images
  -> encode(images)
  -> backbone
  -> projector or Identity
  -> F.normalize(..., dim=1)
  -> z
```

Then:

- `flat_ot`: fit or use a K-way balanced OT assignment where `K = 2 ** depth`.
- `flat_vmf_ot`: fit or use a K-way balanced spherical/vMF OT assignment, where
  `K = flat_vmf_num_components` when set, otherwise `2 ** depth`.
- `vmf`: fit or use an unconstrained K-way vMF assignment.
- `hierarchical_vmf_ot`: recursively fit balanced two-way splits on the sphere.
- `hierarchical_vmf`: recursively fit spherical/vMF splits with unconstrained
  hard branch assignments.
- `recursive_ot`: recursively fit balanced two-way splits using Euclidean
  distances and centroids.

For batch-local modes, all tree/OT fitting happens on `z.detach()`. Gradients
flow through the prediction loss that asks live `z` to predict those fitted
labels.

### Tree / OT Helpers

- `_sinkhorn(scores, num_iters=None)`: balanced entropy-regularized transport.
  - `num_iters=None` uses `self.sinkhorn_iters`.
  - `num_iters=N` uses exactly `N` iterations.
  - `_FLAT_SINKHORN_MIN_ITERS = 50` sets the floor for flat K-way OT.
- `_unbalanced_sinkhorn(scores, tau, ...)`: semi-relaxed unbalanced Sinkhorn
  (scaling form): hard sample marginal, KL-penalized component marginal with
  partial-correction exponent `phi = tau / (tau + ot_epsilon)`. `tau -> inf`
  recovers `_sinkhorn` exactly. `_sinkhorn_2way` dispatches between the two
  based on `ot_unbalanced_tau`. In unbalanced mode the hard split is
  `argmax(q)` (a median split would silently re-impose balance), guarded by
  `ot_unbalanced_min_split_fraction` with per-node balanced fallback; per-node
  minority-child fractions are reported as `tree_min/mean_split_fraction`.
- `compute_unbalanced_tau(epoch, tau_start, tau_final, anneal_epochs)`: pure
  cosine tau schedule (module-level, torch-free); `set_ot_unbalanced_tau` is
  the validated setter.
- `_fit_binary_ot(z, fallback_mu=None, init_mu=None)`: two-component EM +
  Sinkhorn split. `init_mu` warm-starts the EM (used when `tree_warm_start`
  and the node was fitted before); otherwise `_cold_seed_pair(z)` derives
  farthest-point seeds from the data. `_is_degenerate_pair` guards both paths.
- `_build_tree_from_embeddings(...)`: recursive tree fit, returns prototypes,
  paths, node ids, masks, and stats; marks `node_fitted` per node and chooses
  warm vs cold init per node.
- `_build_flat_from_embeddings(...)`: flat K-way OT.
- `_build_vmf_from_embeddings(...)`: flat K-way unconstrained vMF fitting.
- `refresh_assignments(embeddings)`: full-pool refresh API called by `main.py`.

### Depth Annealing and Node Health / Re-seeding

- `compute_active_depth(epoch, depth, epochs_per_level, initial_depth)`:
  pure staircase schedule (module-level, torch-free).
- `set_active_depth(n)`: validated setter; `active_depth` is a plain int
  attribute (not a buffer) so state-dict keys are unchanged.
- The `_forward_embeddings` loss loop is clamped to
  `range(min(depth, active_depth))`; inactive levels are built but produce no
  gradient. `active_depth` is included in the returned metrics.
- Per-node bookkeeping: `node_correct_sum` / `node_count` are scatter-added
  inside the loss (no_grad, active levels only); `finalize_node_stats()` folds
  them into per-level/overall stats once per epoch, updates
  `node_low_acc_streak` / `node_last_acc`, and zeroes the accumulators.
- `select_reseed_nodes()`: live trigger — candidates with streak >=
  `reseed_patience`, worst `node_last_acc` first, capped at
  `reseed_budget_fraction` of measured nodes; a selected node and its whole
  subtree get `node_fitted=False` plus stats reset. Pure helpers
  `select_reseed_indices` / `_subtree_node_ids` hold the selection/subtree
  logic (torch-free, tested in isolation).
- Rationale and failure-mode mapping:
  `.agents/design_choice_depth_annealing_tree_reseeding.md`.

### Prediction Losses

Hierarchical modes:

- `_forward_embeddings(...)`
- cross entropy at each active tree level
- averages over active levels
- returns `loss`, `acc`, and `acc_branch`

Flat modes:

- `_forward_flat_embeddings(...)`
- soft cross entropy against Sinkhorn target probabilities for `flat_ot` /
  `flat_vmf_ot`, or one-hot argmax targets for `vmf`
- returns `loss`, `acc`, and `acc_branch`

### Sigmoid Image-Index Regularization

If `sigmoid_regularization_weight > 0`:

```text
pseudo_labels
  -> sigmoid_label_embeddings
  -> normalized label embeddings
z @ label_embeddings.T
  -> scale by exp(sigmoid_logit_scale)
  -> add sigmoid_logit_bias
  -> B x B pairwise logits
same index: +1 target
different index: -1 target
loss = -mean(logsigmoid(target * logit))
```

`main.py` ramps its active weight through
`set_sigmoid_regularization_progress(...)`.

## Branch-61 Model: Hierarchical K-way vMF/OT Self-Labeling

File:

```text
models/hierarchical_kway_vmf_ot_self_labeling_net.py
```

Class:

```text
HierarchicalKWayVMFOTSelfLabelingNet
```

Additive sibling of the binary model (none of the binary code paths changed).
The branching factor per level comes from a HiRef rank-annealing schedule
`(r_1, ..., r_kappa)` instead of being fixed at 2; splits are K-way spherical
vMF/OT fits; hard labels use HiRef's `argmax` Assign rule (no median split);
only the first `supervised_depth` levels emit pseudo-labels.

Key constructor knobs beyond the shared ones (`kappa`, `ot_epsilon`,
`sinkhorn_iters`, `em_iters`, `ot_unbalanced_tau`,
`ot_unbalanced_min_split_fraction`, `batch_self_labeling`,
`learnable_vmf_prototypes`, `prototype_ema_momentum`, sigmoid knobs):

- `rank_schedule`: explicit per-level branching list (every rank >= 2), or
  null to derive it from the DP.
- `num_leaf_clusters`, `rank_schedule_depth`, `rank_schedule_max_rank`,
  `rank_schedule_base_rank`: DP inputs when `rank_schedule` is null; the
  derived schedule satisfies `prod(r_t) == num_leaf_clusters / base_rank`
  with `r_t <= max_rank` at the user-budget depth.
- `supervised_depth`: early-levels pseudo-label cutoff; `set_active_depth`
  is clamped to it, so depth annealing composes with the cutoff.

Structure notes:

- `depth = len(rank_schedule)`; level `t` holds `rho_t = prod(r_s, s<t)` nodes.
- Node addressing is mixed-radix via `tools/rank_annealing.py`
  (`level_offsets`, `child_global_id`, `leaf_index`) — the generalization of
  the binary heap `2j+1 / 2j+2`.
- Prototypes are stored padded: `[n_internal, max_rank, embedding_dim]`; a
  node at level `t` uses only the first `r_t` rows.
- `_fit_kway_vmf_ot`: farthest-point cold seeding (or warm start from stored
  node prototypes), EM over balanced or semi-relaxed unbalanced Sinkhorn with
  cosine scores and normalized spherical mean updates, `argmax(q)` hard
  labels, per-node balanced refit as the unbalanced collapse guard.
- Trainer hooks mirror the binary model: `refresh_assignments`,
  `set_active_depth`, `set_ot_unbalanced_tau`, `finalize_node_stats`,
  `set_sigmoid_regularization_progress` — duck-typed by `main.py` with no
  trainer changes beyond the `build_train_loader` model-name list and the
  purity/NMI hook below.
- `predict_paths(embeddings)`: read-only root-to-leaf argmax descent through
  the stored prototypes (no fitting, no `node_fitted` mutation). Consumed by
  `main.py::maybe_compute_tree_structure_metrics` for the per-level
  purity/NMI diagnostics (`train.tree_metrics_interval` /
  `train.tree_metrics_max_samples`).
- No `tree_warm_start` flag: warm start is always on per node (`node_fitted`
  gate); EMA remains optional cross-batch memory. No selective re-seeding yet
  (deferred, see the design doc).

Workflow walk-through: `.agents/hierarchical_kway_vmf_ot_self_labeling.txt`.
Design rationale: `.agents/designs/hierarchical_kway_vmf_ot_design.md`.

## Adjacent Models

### `models/sigmoid_pseudo_supervised_net.py`

Flat pairwise image-index sigmoid baseline.

- image embeddings from backbone + optional projector
- label embeddings from an `nn.Embedding`
- B x B image-label logits
- positives are matching pseudo-label IDs in the batch
- negatives are all non-matching pairs

### `models/pseudo_supervised_net.py`

Legacy flat softmax over `num_classes` image-index pseudo-labels. No remaining
config targets it.

### `models/hierarchical_pseudo_supervised_net.py`

Legacy two-level factorized softmax for large image-index spaces. No remaining
config targets it.

### `models/low_rank_multitarget_pseudo_supervised_net.py`

Diagnostic model for the target-sharing hypothesis. Each image-index pseudo-label
maps to a fixed balanced set of `m` latent labels among `K` candidates. The
backbone predicts K-way latent logits and is trained with either
`uniform_multi_ce` (mass on all assigned labels) or `set_ce` (mass on at least
one assigned label). This is not a learned clustering method; the membership
matrix is fixed so rank K is the controlled variable.

Workflow: `.agents/workflows/low_rank_multitarget_flow.txt`.
Design: `.agents/designs/low_rank_multitarget_instance_sweep.md`.

### `models/topk_categorical_bottleneck_pic_net.py`

PIC-style diagnostic model with a deterministic soft categorical bottleneck.
The backbone feature passes through a latent assigner `[D,C]`, softmax produces
a `[B,C]` assignment distribution, optional one-shot column normalization
rescales batch columns, and a linear/small-MLP decoder predicts the original
image-index ID. The loss is instance CE plus optional global latent-usage
balance and per-sample entropy control.

Workflow: `.agents/workflows/topk_categorical_bottleneck_pic_flow.txt`.
Design: `.agents/designs/topk_categorical_bottleneck_pic_plan.md`.

### `models/simsiam.py`, `models/simclr.py`, `models/byol.py`

Older SSL baselines. SimSiam and SimCLR still have configs. BYOL is a placeholder
and raises `NotImplementedError` during construction.

### `models/swav.py`

Placeholder only. `forward` contains a comment and raises `NotImplementedError`.

## Backbones

Directory:

```text
models/backbones/
```

Important exports:

- `resnet18_cifar_variant1`
- `resnet18_cifar_variant2`

`models/__init__.py` also imports torchvision `resnet18` and `resnet50`. The
active config uses `resnet18`; after `get_backbone`, its classifier is replaced
with `Identity` and `output_dim` is set.

## Optimizers and LR

Directory:

```text
optimizers/
```

Registry:

- `sgd`
- `adam`
- `lars`
- `lars_simclr`
- `larc`

`optimizers/__init__.py` creates two parameter groups:

- `base`
- `predictor`

The predictor group matters for SimSiam; most newer models simply have no
`predictor` params.

`optimizers/lr_scheduler.py` implements warmup + cosine decay scheduling and
supports constant predictor LR.

## Tools

Directory:

```text
tools/
```

Key utilities:

- `knn_monitor.py`: feature-bank kNN monitor for backbone quality during SSL.
- `logger.py`: scalar plotting/TensorBoard logging with Colab-friendly fallback.
- `monitor_plots.py`: torch-free SVG monitors written every epoch into the
  run's log dir — `monitor_training.svg` (lr / losses / kNN accuracy),
  `monitor_tree_health.svg` (2x2: per-level + overall node branch accuracy
  against the 0.5 chance line — K-way chance is `1/r_t` per level —
  sample-level path/branch accuracy + split fractions, depth-annealing /
  re-seeding / tau, leaf balance), and `monitor_tree_structure.svg`
  (standalone 2x1: per-level purity and NMI vs true labels, full-width
  panels so deep trees stay readable; written only when the purity/NMI
  diagnostics are enabled). `append_history` aligns series of different
  lifetimes with None padding.
- `average_meter.py`: metric accumulator.
- `rank_annealing.py`: torch-free HiRef rank-annealing schedule
  (`optimal_rank_schedule` dynamic program minimizing the sum of partial
  products subject to a max-rank cap) plus mixed-radix tree index helpers
  (`level_sizes`, `level_offsets`, `num_internal_nodes`, `child_global_id`,
  `leaf_index`). Used by the branch-61 K-way model.
- `tree_metrics.py`: torch-free per-level purity/NMI between tree path
  prefixes and ground-truth labels; `prefix_cluster_ids` /
  `prefix_label_metrics` accept optional `radices` (a K-way rank schedule)
  and default to binary. Wired into training via
  `main.py::maybe_compute_tree_structure_metrics` (every
  `train.tree_metrics_interval` epochs, up to `train.tree_metrics_max_samples`
  memory-loader images, duck-typed on the model's read-only `predict_paths`);
  results plot in the standalone `monitor_tree_structure.svg`. Diagnostic
  only — labels never feed back into training.
- `accuracy.py`, `plotter.py`, `file_exist_fn.py`: smaller helpers.

## Tests

Current focused test file:

```text
tests/test_hierarchical_balanced_vmf_self_labeling.py
```

It verifies:

- Sinkhorn row/column marginal balancing.
- Recursive tree splits produce balanced leaves for power-of-two batches.
- Inactive masks are correct when too few samples remain at deeper levels.
- Flat OT produces balanced soft target probabilities.
- Forward/backward works across:
  - `hierarchical_vmf_ot`
  - `hierarchical_vmf`
  - `recursive_ot`
  - `flat_ot`
  - `flat_vmf_ot`
  - `vmf`
  - learnable vs non-learnable prototypes
  - EMA vs no EMA
  - sigmoid regularization on/off
- Full-pool `refresh_assignments(...)` works for tree and flat modes.
- Depth annealing: staircase schedule math, setter validation, and
  clamp-equals-masking equivalence on `_forward_embeddings`.
- Warm start: `init_mu` polarity is honored, degenerate init falls back to
  cold seeding, `node_fitted` marking, fixed-point behavior on repeat builds.
- Node health: stat accumulation per level, finalize/reset cycle, streak
  increment/reset and min-sample gating.
- Re-seeding: pure selection order/budget, subtree coverage, constructor
  validation, and subtree cold-restart integration.
- Constructor validation for the batch-local warm-start edge case where no
  persisted prototype path exists.

A second torch-free test file covers the SVG monitors:

```text
tests/test_monitor_plots.py
```

Unbalanced OT has its own test file (tau schedule, balanced-limit recovery,
natural-ratio behavior on 12/4 fixtures, split-fraction stats, min-split
guard, setter/constructor validation, forward/backward, get_model and
config/notebook argument passing):

```text
tests/test_unbalanced_ot.py
```

Branch 61 adds two test files:

```text
tests/test_rank_annealing.py
tests/test_hierarchical_kway_vmf_ot.py
```

`test_rank_annealing.py` is torch-free (DP optimality vs brute force,
feasibility/infeasibility, base-rank reduction, mixed-radix index helpers,
binary-heap equivalence for an all-2 schedule). `test_hierarchical_kway_vmf_ot.py`
needs torch (schedule wiring, K-way balanced leaf occupancy, small-node level
masking, `supervised_depth` clamping, unbalanced collapse guard,
forward/backward, non-batch-local stored assignments, node-stats reset).

Tree purity/NMI diagnostics have their own file:

```text
tests/test_tree_metrics.py
```

Run with:

```bash
PYTHONPATH="$PWD" conda run -n ssl_local pytest -q
```

## Common Edit Map

To change the hierarchical vMF algorithm:

- start in `models/hierarchical_balanced_vmf_self_labeling_net.py`
- update or add tests in `tests/test_hierarchical_balanced_vmf_self_labeling.py`
- expose new config fields in `models/__init__.py`
- add defaults/comments in `configs/hierarchical_balanced_vmf_cifar_colab.yaml`
- mirror important notebook hyperparameters in the relevant Colab notebooks

To change the K-way rank-annealing algorithm (branch 61):

- start in `models/hierarchical_kway_vmf_ot_self_labeling_net.py`
- schedule/indexing math lives in `tools/rank_annealing.py` (torch-free)
- update or add tests in `tests/test_hierarchical_kway_vmf_ot.py` and
  `tests/test_rank_annealing.py`
- expose new config fields in `models/__init__.py`
- add defaults/comments in `configs/hierarchical_kway_vmf_ot_cifar_colab.yaml`
- mirror hyperparameters in `notebooks/hierarchical-kway-vmf-ot-cifar10-ssl.ipynb`

To change source-pool sampling or image-index labels:

- start in `datasets/pseudo_supervised.py`
- check `main.py::build_train_loader`
- verify `PseudoSourcePoolDataset` in `main.py` if full-pool refresh is affected

To change augmentations used by pseudo-supervised models:

- start in `augmentations/superimpose_aug.py`
- remember pseudo-supervised models do not go through `get_aug(...)`
  for train-time transforms

To add a model:

- implement it in `models/`
- register it in `models/__init__.py`
- add a config in `configs/`
- update `main.py::build_train_loader` if it needs a special dataset wrapper

To add a dataset:

- register it in `datasets/__init__.py`
- ensure kNN/linear eval can infer labels/classes if evaluation is needed

To change optimizer behavior:

- start in `optimizers/__init__.py`
- check scheduler assumptions in `optimizers/lr_scheduler.py`
- check `main.py::build_optimizer_and_scheduler`

## Important Behavioral Gotchas

- `batch_self_labeling=False` requires `tree_refresh_interval > 0`; `main.py`
  raises a `ValueError` otherwise.
- Full-pool refresh expects one clean embedding per source-pool pseudo class.
- `PseudoSupervisedDataset.__getitem__` ignores the dataloader index and samples
  randomly from the fixed source pool.
- Pseudo labels are source-pool offsets, not CIFAR/STL class labels.
- `source_pool_size` must not exceed the underlying dataset length.
- kNN monitor evaluates the backbone against real dataset labels, not pseudo
  labels.
- `flat_ot` silently enforces at least 50 Sinkhorn iterations for K-way balance.
- `depth > 16` is rejected in the hierarchical vMF model to avoid a huge tree.
- If `learnable_vmf_prototypes=False`, prototypes are buffers and should not get
  gradients.
- If `prototype_ema_momentum=None`, batch-fitted prototypes are not stored by EMA.
- `batch_self_labeling=True` + `tree_warm_start=True` requires either
  `prototype_ema_momentum` or `learnable_vmf_prototypes=True`; otherwise the
  constructor raises to avoid warm-starting from stale stored prototypes.
- The notebooks clone/pull the current branch inside Colab, then install
  `requirements_colab.txt`.
- `pseudo_labels` dual role in `forward`: when `batch_self_labeling=True`, the
  `pseudo_labels` argument is ignored for tree path/OT lookups, but still used
  for sigmoid image-index regularization.
- `recursive_ot` prototype initialisation quirk: `node_prototypes` is always
  initialised as unit vectors. After the first batch update, `recursive_ot`
  prototypes become Euclidean weighted centroids and are no longer unit vectors.
- `reseed_enabled=True` requires `tree_warm_start=True` and a non-null
  `reseed_acc_threshold`; the constructor raises otherwise.
- Depth annealing and the sigmoid rampup are epoch-indexed and restart from 0
  on backbone-only resume (`checkpoint_resume` restores weights only).
- The node-health buffers (`node_fitted`, `node_correct_sum`, `node_count`,
  `node_low_acc_streak`, `node_last_acc`, `node_levels`) change the model's
  state-dict keys vs older checkpoints; backbone-only loading is unaffected.
- Node-stat buffer updates happen inside `forward` and do not synchronize
  across DataParallel replicas (same caveat as prototype EMA;
  `use_data_parallel` defaults to False).
- `tree_warm_start=True` with `reseed_acc_threshold=None` gives the tree
  continuity but no correction mechanism: wrong cuts persist silently. Watch
  `monitor_tree_health.svg` or set the threshold for streak logging.
- Unbalanced OT traps: tau shares `ot_epsilon`'s units and the
  balanced->natural transition is sharp around `tau ~ ot_epsilon` — values
  well above epsilon are effectively balanced. Relaxing Sinkhorn alone is not
  enough: the hard split must be `argmax(q)` (the model handles this), since
  a median split silently re-imposes 50/50. Flat modes always stay balanced.
- K-way model (branch 61) gotchas: a node with fewer points than its level's
  rank `r_t` is not split (its level mask is False for those samples), so
  small batches with large late ranks can leave deep levels entirely
  unsupervised — match `prod(rank_schedule[:supervised_depth])` to the batch
  size. The DP needs `num_leaf_clusters / base_rank` to factor into `depth`
  ranks each `<= max_rank` (e.g. 250 with max_rank 16 at depth 2 raises).
  `set_active_depth` silently clamps to `supervised_depth` — depth annealing
  beyond the cutoff is a no-op by design. There is no
  `tree_warm_start` flag: warm start is per-node automatic; pair
  `batch_self_labeling=True` with `prototype_ema_momentum` for cross-batch
  prototype continuity (the constructor does not force it, unlike the binary
  model).

## Minimal Mental Model For The Active Experiment

1. `PseudoSupervisedDataset` samples augmented CIFAR-10 images from a fixed pool
   and labels each image by its pool offset.
2. `HierarchicalBalancedVMFSelfLabelingNet.encode` maps the image to a normalized
   point on the sphere.
3. In batch-local mode, the model fits a temporary tree or flat OT assignment from
   the current batch's detached embeddings.
4. The same live embeddings are trained to predict those fitted assignments.
5. Optional prototype EMA gives the tree directions memory across batches;
   optional warm start (`tree_warm_start`) additionally makes each node's EM
   re-fit start from its previous solution, so the partition evolves
   continuously instead of being re-derived from scratch.
6. Optional depth annealing supervises the tree coarse-to-fine: only the first
   `active_depth` levels produce loss, one more level activating every
   `depth_annealing_epochs_per_level` epochs.
7. Optional node-health tracking measures each node's branch accuracy per
   epoch; persistently-at-chance nodes (unlearnable, typically
   augmentation-unstable cuts) can be selectively cold-restarted
   (`reseed_enabled`) under a patience + budget rule.
8. Optional sigmoid regularization asks the same embedding to identify image
   indices through pairwise image-label matching, helping fight collapse.
9. kNN monitor reports whether the backbone representation aligns with real
   CIFAR-10 classes; two SVGs in the log dir (`monitor_training.svg`,
   `monitor_tree_health.svg`) track optimization and tree health per epoch.
