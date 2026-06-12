# Changelog - Branch 56

Branch: `56-depth-annealing`

Baseline comparison: branch-55 hierarchical balanced vMF self-labeling behavior
plus the staircase depth-annealing changes in this branch.

## Motivation

With all `model.depth` tree levels supervised from epoch 0, the deep levels are
balanced splits of a near-random feature space — partitions of sampling noise.
The cross-entropy trains the backbone to conform to them, wasting gradient on
targets that churn between refreshes and committing the representation to
arbitrary fine-scale structure before coarse structure exists. Depth annealing
activates levels one at a time, so each new level is first supervised only
after the coarser levels have had time to consolidate.

## Added

- Added staircase depth annealing for the hierarchical prediction loss:
  - `compute_active_depth(epoch, depth, epochs_per_level, initial_depth)`:
    pure schedule function — starting from `initial_depth` active levels, one
    extra level is activated every `epochs_per_level` epochs, capped at
    `depth`; `None` disables annealing (all levels active)
  - `set_active_depth(n)`: validated setter on
    `HierarchicalBalancedVMFSelfLabelingNet`; plain int attribute (not a
    buffer), so checkpoint state-dict keys are unchanged
  - the `_forward_embeddings` loss loop is clamped to
    `range(min(depth, active_depth))`: levels beyond the active depth are
    still built during refresh/batch self-labeling but produce no loss
  - `active_depth` is reported in the per-batch metrics dict for logging
- Added trainer hook `maybe_update_depth_annealing(model, epoch, args)` in
  `main.py` (mirrors the sigmoid-rampup setter pattern), called once per epoch;
  prints on each level activation.
- Added config keys to `configs/hierarchical_balanced_vmf_cifar_colab.yaml`
  (default-disabled, backward compatible):
  - `train.depth_annealing_epochs_per_level: null`
  - `train.depth_annealing_initial_depth: 1`
- Added tests: staircase schedule math and validation errors, setter range
  validation and full-depth default, clamp-equals-masking equivalence on
  `_forward_embeddings`, and forward/backward with `active_depth=1` in batch
  self-labeling mode.

## Changed

- No behavior changes when the config keys are absent or null: the default
  `active_depth = depth` reproduces the previous loss bit-for-bit.

## Notes

- The epoch-based schedule restarts from 0 on backbone-only resume (same
  semantics as `sigmoid_regularization_rampup_epochs`).
- Inactive-level prototypes still EMA-track batch fits under
  `batch_self_labeling=True`; they are unsupervised, kept for diff minimality.
- Design rationale and interaction with warm start / selective re-seeding
  (branch 57): `.agents/design_choice_depth_annealing_tree_reseeding.md`.

## Validation

- Sandbox-verified (no torch/pytest available there):

```bash
python -m compileall models/hierarchical_balanced_vmf_self_labeling_net.py main.py tests/
```

- `compute_active_depth` executed standalone against the full staircase
  assertion set (disabled schedule, level activations at 0/50/.../250 for
  depth 6, initial-depth offsets, ValueError cases): all passed.
- Config YAML parse-checked; new keys default to null/1.
- Pending locally (torch required):

```bash
PYTHONPATH=/Users/xmc28/Desktop/projects/code/my_simsiam conda run -n ssl_local pytest -q tests/test_hierarchical_balanced_vmf_self_labeling.py
```

Expected at this branch's head: `15 passed` (11 pre-existing + 4 new).

## Suggested run

depth=6, `depth_annealing_epochs_per_level: 100` for an 800-epoch run (levels
activate at epochs 0/100/200/300/400/500); monitor per-level `acc_branch` and
the logged `active_depth`.
