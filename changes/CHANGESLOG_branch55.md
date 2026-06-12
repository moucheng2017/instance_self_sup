# Changelog - Branch 55

Branch: `55-add-vmf-assignment-mode`

Baseline comparison: previous hierarchical balanced vMF self-labeling behavior
plus the new assignment-mode changes in this branch.

## Added

- Added `ot_assignment_mode: flat_vmf_ot` for flat spherical/vMF mixture
  self-labeling with balanced Sinkhorn transport:
  - fits a non-recursive vMF mixture on normalized embeddings
  - uses `flat_vmf_num_components` as the user-defined component count
  - reuses the flat K-way Sinkhorn EM path with cosine/vMF scores and normalized
    prototype updates
  - stores balanced soft transport targets for the flat prediction loss
- Added `configs/flat_vmf_ot_cifar_colab.yaml` for the new mode.
- Added `notebooks/flat-vmf-ot-cifar10-ssl.ipynb` with explicit
  `flat_vmf_num_components` overrides.
- Added `notebooks/vmf-sigmoid-cifar10-ssl.ipynb` for flat vMF self-labeling
  without OT balancing and with sigmoid image-index regularization enabled.
- Added `ot_assignment_mode: vmf` for flat spherical/vMF mixture self-labeling:
  - fits `K = 2 ** depth` vMF prototypes with ordinary soft responsibilities
  - assigns samples by unconstrained hard argmax labels
  - stores one-hot flat assignment targets instead of balanced OT transport
- Added `ot_assignment_mode: hierarchical_vmf` for recursive spherical/vMF
  self-labeling:
  - keeps the existing hierarchical vMF fitting path used by
    `hierarchical_vmf_ot`
  - assigns branch labels by unconstrained vMF prototype scores instead of
    forced equal-size splits
- Added `hierachical_vmf` as a typo-tolerant alias for `hierarchical_vmf`.
- Added tests for the new flat vMF OT mode and the unconstrained flat and
  hierarchical assignment modes, including forward/backward coverage with
  learnable prototypes, prototype EMA, sigmoid regularization, and full-pool
  assignment refresh.

## Changed

- Updated `configs/hierarchical_balanced_vmf_cifar_colab.yaml` mode comments to
  document `hierarchical_vmf`, `flat_vmf_ot`, and `vmf`.
- Updated the hierarchical balanced vMF CIFAR-10 Colab notebook to list the
  new `ot_assignment_mode` options and include a `flat_vmf_num_components`
  override:
  - `notebooks/hierarchical-balanced-vmf-cifar10-ssl.ipynb`

## Validation

- Reviewed the implementation diff after tests and kept existing
  `hierarchical_vmf_ot`, `recursive_ot`, and `flat_ot` paths unchanged except
  for shared mode dispatch needed by the new options.
- Verified focused tests with:

```bash
PYTHONPATH=/Users/xmc28/Desktop/projects/code/my_simsiam conda run -n ssl_local pytest -q tests/test_hierarchical_balanced_vmf_self_labeling.py
```

Latest result after adding `flat_vmf_ot`: `11 passed`.

- Verified syntax with:

```bash
conda run -n ssl_local python -m compileall .
```

- Verified the new config builds the requested flat vMF OT prototype shape:

```bash
conda run -n ssl_local python -c "from arguments import build_args; from models import get_model; args = build_args('configs/flat_vmf_ot_cifar_colab.yaml', data_dir='/tmp/data', log_dir='/tmp/logs', ckpt_dir='/tmp/ckpt', create_dirs=False); model = get_model(args.model, num_classes=32); print(model.ot_assignment_mode, model.num_flat_prototypes, tuple(model.flat_prototypes.shape))"
```

Latest result: `flat_vmf_ot 64 (64, 256)`.

- Verified the new flat vMF + sigmoid notebook override pattern builds the
  intended no-OT mode:

```bash
conda run -n ssl_local python -c "from arguments import build_args; from models import get_model; overrides={'model': {'ot_assignment_mode': 'vmf', 'depth': 7, 'sigmoid_regularization_weight': 0.1, 'batch_self_labeling': False, 'learnable_vmf_prototypes': True, 'prototype_ema_momentum': None}}; args = build_args('configs/hierarchical_balanced_vmf_cifar_colab.yaml', overrides=overrides, data_dir='/tmp/data', log_dir='/tmp/logs', ckpt_dir='/tmp/ckpt', create_dirs=False); model = get_model(args.model, num_classes=32); print(model.ot_assignment_mode, model.num_flat_prototypes, float(model.sigmoid_regularization_max_weight))"
```

Latest result: `vmf 128 0.1`.
