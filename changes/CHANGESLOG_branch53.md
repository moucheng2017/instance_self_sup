# Changelog - Branch 53

Branch: `53-recursive-self-labelling-sigmoid-loss`

Baseline comparison: `main...HEAD`, plus the latest local documentation update in
this changelog.

## Added

- Added `models/hierarchical_balanced_vmf_self_labeling_net.py`, the active
  hierarchical balanced vMF / OT self-labeling model with:
  - batch-local and full-pool assignment modes
  - `hierarchical_vmf_ot`, `recursive_ot`, and `flat_ot`
  - optional learnable prototypes
  - optional prototype EMA
  - optional sigmoid image-index regularization
  - full-pool `refresh_assignments(...)`
- Added `configs/hierarchical_balanced_vmf_cifar_colab.yaml` for the active
  CIFAR-10 hierarchical balanced vMF experiment.
- Added `models/sigmoid_pseudo_supervised_net.py` and
  `configs/sigmoid_pseudo_supervised_cifar_colab.yaml` for the pairwise sigmoid
  image-index baseline.
- Added `models/hierarchical_pseudo_supervised_net.py`, a two-level image-index
  softmax baseline kept as legacy code.
- Added Colab notebooks:
  - `notebooks/hierarchical-balanced-vmf-cifar10-ssl.ipynb`
  - `notebooks/hierarchical-balanced-vmf-cifar10-ssl-no-sigmoid.ipynb`
  - `notebooks/sigmoid-pseudo-supervised-cifar10-ssl.ipynb`
- Added a focused test suite in
  `tests/test_hierarchical_balanced_vmf_self_labeling.py` covering:
  - Sinkhorn transport marginal balancing
  - recursive tree splitting and inactive-level masks
  - flat OT soft targets
  - forward/backward behavior across all hierarchical vMF assignment modes
  - learnable vs buffer prototypes
  - prototype EMA behavior
  - sigmoid image-index regularization
  - full-pool assignment refresh for tree and flat modes
- Added agent/design documentation:
  - `.agents/codebase_structure.md`
  - `.agents/hierarchical_balanced_vmf_self_labeling.txt`
  - `.agents/OT_design_justifications.md`
  - `.agents/natural_splits_ratios_strategy.md`
  - `.agents/sigmoid_pseudo_sup.txt`
- Added this branch changelog at `changes/CHANGESLOG_branch53.md`.

## Changed

- Reworked `README.md` from old SimSiam-first documentation into current
  hierarchical balanced vMF self-labeling documentation, including active setup,
  configs, notebooks, tests, and linear-eval guidance.
- Updated `arguments.py`:
  - config keys are validated as Python identifiers, so keys such as `beta1` and
    `beta2` parse correctly
  - debug mode caps pseudo-supervised `source_pool_size` and `samples_per_epoch`
  - missing required paths now raise a clear `ValueError`
  - saved run config now contains the merged config after overrides
  - CLI/build args support `checkpoint_resume`
- Updated `main.py`:
  - routes pseudo-supervised, hierarchical pseudo-supervised, sigmoid
    pseudo-supervised, and hierarchical vMF models through `PseudoSupervisedDataset`
  - adds `PseudoSourcePoolDataset` for clean full-pool assignment refresh
  - supports full-pool hierarchical assignment refresh when
    `batch_self_labeling=False`
  - validates that full-pool mode requires `tree_refresh_interval > 0`
  - supports sigmoid regularization ramp-up
  - supports backbone-only checkpoint initialization with `checkpoint_resume`
  - supports periodic checkpoint tags
  - supports dedicated kNN eval datasets through `dataset.knn_dataset`
  - passes Adam `beta1`/`beta2` to the optimizer registry
- Updated `models/__init__.py`:
  - replaced dynamic backbone `eval(...)` with `_BACKBONE_REGISTRY`
  - registered hierarchical vMF, hierarchical pseudo-supervised, and sigmoid
    pseudo-supervised models
  - removed config-facing registration for augmentation-binary, key-query, and
    superimpose models
- Updated `optimizers/__init__.py` with Adam support and configurable
  `beta1`/`beta2`.
- Updated `datasets/__init__.py` with `stl10_unlabeled` support.
- Updated `datasets/pseudo_supervised.py` to postprocess transformed images in a
  single expression.
- Updated `datasets/superimpose.py` to remove the ignored `augment_probability`
  argument; the legacy dataset always augments.
- Updated `augmentations/superimpose_aug.py`:
  - clean transform now uses tensor conversion only
  - postprocess no longer clamps tensors before normalization
- Updated `linear_eval.py` to weight final accuracy by batch size.
- Updated model constructors in `models/simsiam.py`, `models/simclr.py`,
  `models/pseudo_supervised_net.py`, and `models/aug_binary.py` so backbone
  dependencies are explicit instead of hidden `resnet50()` defaults.
- Updated `models/swav.py` into an explicit placeholder with a not-implemented
  comment and `NotImplementedError`.
- Updated `fix_notebook2.py` to generate a hierarchical balanced vMF Colab
  notebook using `configs/hierarchical_balanced_vmf_cifar_colab.yaml`.
- Updated `.agents/codebase_structure.md` to reflect the current trimmed repo,
  active model, remaining configs, and legacy placeholders.

## Removed

- Removed old planning/change files:
  - `.agent/PLANS.md`
  - `CHANGES.md`
- Removed old/unused configs:
  - `configs/aug_binary_cifar_colab.yaml`
  - `configs/aug_binary_cifar_jupyter.yaml`
  - `configs/key_query_cifar_colab.yaml`
  - `configs/pseudo_supervised_stl10_colab.yaml`
  - `configs/simsiam_cifar_eval_lars.yaml`
  - `configs/simsiam_cifar_eval_sgd.yaml`
  - `configs/simsiam_cifar_quick.yaml`
  - `configs/superimpose_sources_cifar_colab.yaml`
  - `configs/superimpose_sources_stl10_colab.yaml`
- Removed `models/superimpose_net.py`.
- Removed obsolete notebook:
  - `notebooks/pseudo-supervised-image-contrastive-ssl.ipynb`
- Removed stale README references to deleted augmentation-binary configs/helper
  files and to non-existent `configs/simsiam_cifar_eval.yaml`.
- Removed stale superimpose config usage from `fix_notebook2.py`.

## Fixed

- Fixed critical config parsing failure for remaining configs containing
  `beta1` and `beta2`.
- Fixed `tests/test_hierarchical_balanced_vmf_self_labeling.py` Sinkhorn test to
  use the flat K-way Sinkhorn iteration floor instead of the low binary-split
  default.
- Fixed stale agent documentation that referenced removed configs and old model
  registry behavior.
- Fixed the misleading superimpose `augment_probability` API, which accepted a
  value but always augmented.
- Fixed `models/swav.py` syntax validity by giving `forward(...)` an explicit
  not-implemented body.

## Validation

- Verified branch-vs-main coverage with:

```bash
git diff --name-status main...HEAD
git diff --stat main...HEAD
```

- Verified config parsing for:
  - `configs/hierarchical_balanced_vmf_cifar_colab.yaml`
  - `configs/sigmoid_pseudo_supervised_cifar_colab.yaml`
- Verified local path/reference scans for README and `.agents/codebase_structure.md`.
- Verified Python syntax scan after the SwAV placeholder update.
- Verified tests with:

```bash
PYTHONPATH="$PWD" conda run -n ssl_local pytest -q
```

Latest result: `7 passed`.
