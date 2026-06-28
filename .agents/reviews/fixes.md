# Fix Report — `2-vicreg-barlow-twins-baselines`

Scope for this run was explicitly expanded by the user beyond the normal CRITICAL-only mandate
(the Reviewer found **zero CRITICAL** findings). Authorized targets: **R1 (MAJOR)** plus the cheap
safe MINOR/NIT items **R3** and **R6**. No other changes were made. All edits are additive and
honor the §8 binding decisions and the same-data / raw-feature invariants.

---

## R1 — `extract_features` shuffle-detection was sampler-name-based (MAJOR)

- **Files changed:** `analysis/spectral.py`, `tests/test_spectral_diagnostics.py`
- **Fix:** Replaced the name-based check (`sampler.__class__.__name__ == "RandomSampler"`) with a
  **structural** guard. `_loader_is_shuffling` now treats a loader as non-shuffling **only** when its
  sampler is a plain `torch.utils.data.SequentialSampler` (the default produced by `shuffle=False`).
  It also inspects `loader.batch_sampler`: for the standard `BatchSampler` it requires the wrapped
  inner sampler to be sequential, and a custom non-`BatchSampler` `batch_sampler` (whose order cannot
  be confirmed) is rejected. Added a new helper `_sampler_is_sequential`. The error message now
  states the expectation (SequentialSampler / shuffle=False).
- **Why this matches design intent (§3.4/§3.5):** the diagnostics loader must be non-shuffling so the
  shared N-pool is evaluated in identical order for every method. A name-based check silently passed
  `SubsetRandomSampler`, `WeightedRandomSampler`, and any third-party random sampler; the structural
  check closes that gap while still accepting the one legitimate non-shuffling case (the in-repo
  `build_diagnostics_loader` builds with `shuffle=False`).
- **Verification:**
  - Added regression tests: `test_extract_features_rejects_subset_random_sampler`,
    `test_extract_features_rejects_weighted_random_sampler` (both must now raise), and
    `test_extract_features_accepts_sequential_loader` (the legitimate `shuffle=False` case must pass).
    The existing `test_extract_features_rejects_shuffling_loader` and
    `test_build_diagnostics_loader_reuses_selected_indices` remain valid.
  - Because torch/pytest could not be installed in the sandbox (see Test status), I re-implemented the
    guard against mock sampler classes mirroring how `DataLoader` populates `.sampler`/`.batch_sampler`
    and confirmed: shuffle=False → not shuffling; shuffle=True / SubsetRandomSampler /
    WeightedRandomSampler / custom batch_sampler → shuffling (rejected). All five cases behaved as
    expected.

## R3 — `extract_features` could silently under-fill below `n_samples` (MINOR)

- **Files changed:** `analysis/spectral.py`, `tests/test_spectral_diagnostics.py`
- **Fix:** After truncation, if `F_out.shape[0] < n_samples`, emit a `warnings.warn(...)` explaining the
  loader yielded fewer samples than requested and that the diagnostics loader should cover the full
  N-pool. Chose a warning (not a hard assert) so the legitimate "pool smaller than requested" case is
  not turned into a crash, while still surfacing the same-data-invariant risk for a future mismatched
  caller (design.md §3.4 / §3.5 item 2). Added `import warnings`.
- **Verification:** Added `test_extract_features_warns_on_underfilled_pool` (requests more samples than
  the dataset has and asserts the warning fires and the matrix length equals the dataset size).
  Behavior for the matched-size case (current callers) is unchanged. Could not run under pytest (no
  torch); logic reviewed and syntax-checked.

## R6 — dead `get_aug('pseudo_supervised_net')` branch (NIT)

- **Files changed:** `augmentations/__init__.py`
- **Fix:** Confirmed the branch is genuinely dead — when `model.name == "pseudo_supervised_net"`,
  `main.build_train_loader` enters the branch at `main.py:90` and uses `transform=None`, letting
  `PseudoSupervisedDataset` apply `StrongTransform` internally (`datasets/pseudo_supervised.py:23`);
  the eval/diagnostics paths use `train=False`, which ignores `name`. Rather than removing the branch
  (which would change a future stray call from returning a transform to raising `NotImplementedError`,
  a behavior change), I added a clarifying comment per the Reviewer's "or add a comment" option and
  design.md §3.3 ("Note any change to `pseudo_supervised_net`'s aug mapping"). No behavioral change.
- **Verification:** Verified no caller passes `name='pseudo_supervised_net'` with `train=True`
  (`grep get_aug`: callers are `main.py:133` train-path which the pseudo branch never reaches,
  plus `train=False` callers). `augmentations/__init__.py` parses cleanly.

---

## Test status

**`pytest tests/ -q` could NOT be executed in this environment.** The bash sandbox has no `torch` and
no `pytest`, and `pip install torch pytest --break-system-packages` is blocked by the proxy
(`403 Forbidden`, no network) — identical to the constraint the Reviewer reported. Therefore the full
suite pass/fail is **UNVERIFIED** and must be confirmed by CI / an environment with torch installed.

What I was able to run (torch-free, executed successfully with numpy/PyYAML/ast only):

- `tests/test_baseline_notebooks.py` — all 3 tests **PASS** (executed via a manual harness).
- `tests/test_baseline_configs.py` — the 3 yaml-only tests (`common_fields`, `lars_block`+`sgd`,
  `locked_coefficients`) **PASS**. The 4th test (`test_get_model_and_get_aug_resolve_all_baselines`)
  requires torch and was not run.
- Syntax: `analysis/spectral.py`, `augmentations/__init__.py`, and `tests/test_spectral_diagnostics.py`
  all `ast.parse` cleanly.
- R1 guard logic verified against mock samplers (all 5 cases correct, see R1 above).

**Could NOT run** (require torch): `test_spectral_diagnostics.py` (including my 4 new tests),
`test_vicreg.py`, `test_barlow_twins.py`, `test_subset_n.py`, `test_checkpoint_init.py`,
`test_negatives_ratio.py`, and the torch-dependent `test_baseline_configs.py` case.

> ACTION FOR CI: run `pytest tests/ -q` with torch/torchvision/numpy/pyyaml/pytest installed and
> confirm green, paying attention to the four new tests in `tests/test_spectral_diagnostics.py`.

## Intentionally not fixed

- **R2** — Reviewer's explicit "do not change" note (VICReg variance averaged / covariance summed is
  the correct §3.1 convention). Left untouched.
- **R4** (NIT) and **R5** (NIT) — not in the authorized scope for this run; left for later. R4 is a
  pure documentation nit (order-determinism already holds via `shuffle=False`); R5 is a test-robustness
  nit on an implementation already verified correct.
