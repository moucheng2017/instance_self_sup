# Design Guidance: VICReg & Barlow Twins Baselines

**Branch:** `2-vicreg-barlow-twins-baselines`
**Audience:** a coding agent implementing the Section 2.1 "Feature Decorrelation Methods" go/no-go baseline from `.agents/designs/idea.md`.
**Deliverable:** two new SSL methods (VICReg, Barlow Twins) wired into the existing training stack, plus Google Colab notebooks under `notebooks/` that mirror `notebooks/random-meta-cifar10-ssl.ipynb`, all built test-first.

---

## 1. Scope and intent

`idea.md` Section 2.1 is the **week-1 go/no-go experiment**. It asks: do VICReg and Barlow Twins — methods that explicitly decorrelate features — reach the same *effective rank* and KNN accuracy as `pseudo_sup` in the low-data regime (N ≤ 1000)? Both outcomes are publishable, so the job is to produce a *trustworthy, reproducible* comparison, not to win a benchmark.

This branch is responsible for exactly two things:

1. **Two new pretraining methods** — VICReg and Barlow Twins — that train from random init and plug into the existing `main.train_model` loop with no special-casing beyond model/aug registration.
2. **Colab notebooks** that run those methods on CIFAR-10 (and optionally STL-10) across the N-sweep, then report the Section 2.1 metrics: effective rank of the penultimate-layer feature matrix, KNN accuracy (k=20), and the top-20 singular values.

SimCLR and SimSiam already exist (`models/simclr.py`, `models/simsiam.py`) and are reused as comparison points — do not reimplement them. `pseudo_sup` is the existing `pseudo_supervised_net`. The spectrum-surgery experiments (idea.md Section 2.2) are **out of scope** for this branch, but the spectral-diagnostics module built here is the shared foundation for them, so build it cleanly.

Keep changes additive and minimal. Do not modify the core epoch loop, argument parsing, or existing model implementations. The only existing-code edits allowed are the registration hooks listed below plus the narrowly scoped, tested low-data subset plumbing in `main.build_train_loader` / dataset helpers described in §3.5.

---

## 2. How the existing stack works (the contract you must satisfy)

Read these files before writing code; the new methods must conform to the conventions they establish.

**Model contract.** A two-view SSL model is an `nn.Module` exposing:
- `self.backbone` — the castrated ResNet (its `output_dim` attribute is the penultimate feature width; `resnet18` → 512). `knn_monitor` and `linear_eval` consume `model.backbone` directly, so the attribute name must be exactly `backbone`.
- `forward(x1, x2) -> {"loss": tensor, ...}` — returns a dict with a scalar `loss` key. Extra keys (e.g. per-term losses) are logged automatically and are encouraged for diagnostics. See `models/simclr.py` and `models/simsiam.py`.

**Dispatch.** `main.forward_batch` calls `model.forward(*inputs)` when the dataloader yields a list/tuple of views, otherwise `model.forward(inputs, targets)`. Two-view methods therefore need a dataset whose transform returns `(x1, x2)`. This is already the default non-pseudo path in `main.build_train_loader` (the `else` branch), which calls `get_dataset(transform=get_aug(train=True, **args.aug_kwargs), ...)`.

**Registration points and allowed existing-code edits**:
- `models/__init__.py` → add `elif model_cfg.name == 'vicreg'/'barlow_twins'` branches in `get_model`, importing the new classes.
- `augmentations/__init__.py` → add `name` handling in `get_aug` for the new methods.
- `main.py` / dataset helper code → add only the deterministic `train.subset_n` support described in §3.5; do not special-case VICReg or Barlow Twins in the training loop.
- `configs/` → add new YAML config files (additive; no edits to existing configs).

**Config → args.** `arguments.build_args` loads a YAML, deep-merges `overrides`, and exposes it as nested `Namespace` objects (`args.model.name`, `args.train.base_lr`, …). `args.aug_kwargs` is `{name: model.name, image_size: dataset.image_size}` — so `get_aug` is keyed on `model.name`. This means **`model.name` must equal the `get_aug` key**; plan the names accordingly (`vicreg`, `barlow_twins`).

**Notebook entry point.** Notebooks call `colab_utils.train_from_colab(config_file=..., overrides=..., ...)`, which builds args (forcing notebook-safe defaults: `num_workers=0`, `tensorboard=False`, no DataParallel) and calls `main.train_model`. `train_model` runs the epoch loop, calls `knn_monitor` every `knn_interval`, saves a checkpoint, and (unless `args.eval is False`) runs linear eval. **`train_from_colab` already supports any non-meta model** — VICReg/Barlow Twins need no new colab helper, only a config and a model. Confirm this rather than adding a new wrapper.

**Optimizers.** `optimizers/get_optimizer` supports `sgd`, `lars`, `lars_simclr`, `larc`. LR is scaled `base_lr * batch_size / 256` in `build_optimizer_and_scheduler`. VICReg/Barlow Twins canonically use LARS; for small-N CIFAR runs SGD or LARS are both acceptable — make it a config choice, default to what the existing baselines use for parity.

---

## 3. New components to build

### 3.1 `models/vicreg.py`

A projector + VICReg loss. Follow the BN-MLP projector style already in `models/simsiam.py` (3 linear layers with BatchNorm; expander dim configurable, default 2048 for CIFAR — the paper's 8192 is overkill at this scale).

VICReg loss over two projected batches `z_a, z_b` of shape `[N, D]`:
- **Invariance** `s = mse(z_a, z_b)`.
- **Variance** `v(z) = mean_j relu(gamma - sqrt(var(z[:,j]) + eps))`, with `gamma = 1.0`, `eps = 1e-4`. Apply to both `z_a` and `z_b`, then average the two view penalties: `var_loss = (v(z_a) + v(z_b)) / 2`. This matches the reference VICReg implementation while keeping the default coefficient at `25.0`.
- **Covariance** `c(z) = sum_{i != j} Cov(z)_{ij}^2 / D`, where `Cov(z) = (z - z.mean(0)).T @ (z - z.mean(0)) / (N - 1)`. Apply to both.
- **Total** `loss = sim_coeff * s + std_coeff * var_loss + cov_coeff * (c(z_a) + c(z_b))`, default coefficients `sim_coeff = 25.0`, `std_coeff = 25.0`, `cov_coeff = 1.0`.

Return `{"loss": total, "inv_loss": s, "var_loss": ..., "cov_loss": ...}` so each term is logged. Make coefficients, expander dim, `eps`, and `gamma` constructor arguments read from `model_cfg` in `get_model` (use `getattr` with defaults, matching the `pseudo_supervised_net` pattern). Confirm the exact coefficient values and the variance/covariance formulas against the original paper (Bardes et al., 2022) before finalizing.

### 3.2 `models/barlow_twins.py`

Projector + Barlow Twins loss over `z_a, z_b` of shape `[N, D]`:
- Batch-normalize each projection along the batch dim using the reference convention: either `nn.BatchNorm1d(projector_dim, affine=False)` on each projected batch, or a manual normalization equivalent to `(z - z.mean(0)) / (z.std(0, unbiased=False) + eps)`. Include `eps` if normalizing manually so tiny CPU tests do not produce NaNs.
- Cross-correlation `C = (z_a_norm.T @ z_b_norm) / N`, shape `[D, D]`.
- **Loss** `= sum_i (1 - C_ii)^2 + lambd * sum_{i != j} C_ij^2`, default `lambd = 0.0051`. Do not name a Python parameter `lambda`; use `lambd` or `offdiag_coeff`.
- Use a helper `off_diagonal(C)` to select off-diagonal elements.

Return `{"loss": total, "on_diag": ..., "off_diag": ...}`. Projector: 3-layer BN-MLP, dim configurable (default 2048). `lambd` and projector dim come from `model_cfg`. Confirm against Zbontar et al., 2021.

### 3.3 Augmentation registration

VICReg and Barlow Twins can use the repo's standard SimCLR-style stochastic two-view augmentation for parity with existing baselines (random resized crop, flip, color jitter, grayscale, blur). The repo already has `SimCLRTransform` (`augmentations/simclr_aug.py`) which returns two independently sampled views from the same transform distribution. In `get_aug`, map `name in ('vicreg', 'barlow_twins')` to `SimCLRTransform(image_size)` (or a thin shared alias). Do not write a new augmentation unless you deliberately choose paper-fidelity over repo parity; if you do, document that Barlow Twins uses an asymmetric blur/solarization recipe and test the new transform. Verify `SimCLRTransform.__call__` returns a 2-tuple `(x1, x2)` so it matches the `forward(x1, x2)` contract.

### 3.4 `analysis/spectral.py` — shared diagnostics

Port the reference functions from `idea.md` Section 2.2 "Shared diagnostics" into a tested module. Minimum surface:

```python
def spectral_diagnostics(F):      # F: [N, D] tensor
    # returns (singular_values: np.ndarray, effective_rank: float, explained_variance: np.ndarray)

def effective_rank(F):            # exp(entropy of normalized singular value distribution)

def knn_eval(F, labels, k=20, n_train=None):  # cosine-similarity KNN accuracy within the N samples

def extract_features(backbone, loader, device, n_samples=1000):
    # forward pass of the *penultimate* layer (model.backbone), L2-norm optional, returns (F, labels)
```

`extract_features` must read features from `model.backbone` (the 512-d penultimate layer), not the projector, because idea.md defines the feature matrix `F` as the ResNet-18 penultimate output. Reuse `knn_monitor`'s feature-extraction approach (`tools/knn_monitor.py`) for consistency. Effective rank uses the natural-log entropy formula; document the convention in a docstring.

**Determinism (non-negotiable — every method must be evaluated on the exact same images, in the same order).** The whole point of the Section 2.1 comparison is that pseudo_sup, SimCLR, SimSiam, VICReg, and Barlow Twins are scored on an *identical* feature-evaluation set, so any difference in effective rank or KNN is attributable to the method, not to sampling. `extract_features` must therefore guarantee a reproducible, method-independent feature matrix:

- **`backbone.eval()` + `torch.no_grad()`** before the forward pass. ResNet-18 uses BatchNorm; in train mode it normalizes with *batch* statistics, so a feature would depend on how samples are grouped into batches and would drift run-to-run. Eval mode uses fixed running stats, making each image's feature independent of its batch. This is exactly why `knn_monitor` calls `net.eval()`. Omitting it makes the numbers non-deterministic even with a fixed loader order.
- **`shuffle=False`, `drop_last=False`** on the diagnostics loader. `knn_monitor`/`build_eval_loader` rely on `shuffle=False`: the feature bank is built by iterating in order while labels are read by dataset index, so shuffling silently misaligns features and labels and collapses KNN to chance. Beyond alignment, `shuffle=False` preserves the selected-index order so the shared N-pool is evaluated identically for every method. `drop_last=False` ensures all N rows are present.
- **Collect labels from the loader iteration** (`for data, target in loader: feats.append(net(data)); labels.append(target)`) rather than from `dataset.targets`/`.labels`. Combined with `shuffle=False` this is belt-and-suspenders: features and labels are paired within the same batch, so alignment holds even if the dataset attribute order ever diverges.
- **Deterministic eval transform.** Use `get_aug(train=False, train_classifier=False, ...)` → `Transform_single(train=False)`, which is `Resize → CenterCrop → ToTensor → Normalize` with no random crop/flip (the random ops live only in the `train=True` branch). Do not feed a training/two-view transform into the diagnostics loader.
- **`num_workers=0`** (already the notebook default from `colab_utils`) keeps iteration order deterministic.

The same selected N-sample index list used for training at a given `(subset_n, subset_seed)` must also be used for post-training diagnostics at that N. Under the rules above, that index list yields byte-identical image selection and ordering across all methods, which is the prerequisite for an honest effective-rank / KNN comparison.

For `knn_eval`, default `n_train` to `int(0.8 * len(F))`, matching idea.md's 80/20 split for every N in `{200, 500, 1000, 5000, full}` rather than hard-coding `800`. Use `k = min(k, n_train)` to keep the N=200 case valid.

### 3.5 Low-data N-subset support

idea.md sweeps `N ∈ {200, 500, 1000, 5000, full}` for *both* the training set and the feature/KNN evaluation. Two distinct needs:

> **Core invariant — the training pool is decided by the random seed and nothing else.**
> The set of training images at a given N is a *pure function of `(subset_n, subset_seed)`*. It must not depend on the method, the model architecture, the run order, the device, prior RNG consumption (model init, augmentation sampling), or the wall clock. The direct consequence, which is the property the whole comparison rests on: **runs that share the same seed share the same training pool.** Run pseudo_sup, SimCLR, SimSiam, VICReg, and Barlow Twins with the same `subset_seed` and they are guaranteed to train on the identical images at each N; change the seed and they all move to the same new pool together. This is what makes any measured difference attributable to the method rather than to which images each method happened to see.

1. **Training on N samples.** `get_dataset` only supports `debug_subset_size` (a `range(0, k)` slice — not random, not seeded). Add a clean, seeded subset mechanism rather than abusing debug. Recommended: a config field `train.subset_n` (and `train.subset_seed`) consumed in `main.build_train_loader` that produces one reusable selected-index list before any method-specific dataset wrapping. Keep it additive and default `None` (= full set).

   **Select the pool with a dedicated, locally-seeded RNG**, e.g. `g = np.random.RandomState(subset_seed); indices = sorted(g.choice(len(base), size=subset_n, replace=False))` (or a `torch.Generator().manual_seed(subset_seed)`). Do **not** rely on the global `torch`/`numpy`/`random` state seeded in `arguments.set_deterministic` — that global state is consumed by model initialization, augmentation, and data loading, so selecting the pool from it would make the pool depend on how much randomness was drawn beforehand (i.e. on the method and run order). A private generator keyed only on `subset_seed` is the mechanism that enforces the core invariant above. Sorting the chosen indices makes the pool order-stable and easy to compare across runs.

   For regular two-view methods, wrap the base dataset in a `Subset` built from those indices. Preserve label metadata in subset-local order. It is not enough to copy the full parent `.targets` onto a random `Subset`, because `knn_monitor` assumes feature-bank rows and labels have the same order and length. The wrapped subset should expose:
   - `.classes` copied from the parent dataset when available.
   - `.targets = [parent.targets[i] for i in indices]` for CIFAR-like datasets.
   - `.labels = parent.labels[indices]` or an equivalent list for STL-10-style datasets.

   For `pseudo_supervised_net`, do **not** wrap and then let `PseudoSupervisedDataset` sample again via `source_pool_size`; that would double-sample and can break the same-data invariant. Instead pass the selected indices directly as `explicit_indices=indices` to `PseudoSupervisedDataset`, so its pseudo-labels are the positions within the same shared pool. If `explicit_indices` is used, `source_pool_size` is not required.

   Also guard against empty loaders: because the repo uses `drop_last=True` for training, every N-sweep override must choose `batch_size <= subset_n` (or `drop_last` must be changed deliberately and tested). For the notebooks, set batch size per N as `min(DEFAULT_BATCH_SIZE, N)` or another explicit rule that keeps at least one batch.
2. **Feature-matrix N for diagnostics.** Build the diagnostics loader from the exact same selected indices used for training at that N, in the same sorted order. `extract_features(..., n_samples=N)` is still responsible for truncating to N rows, but it should receive a loader whose dataset is already the shared N-pool. Do not evaluate on the first N rows of the full dataset unless that is explicitly the selected pool.

Implement (1) test-first; the subset must be deterministic given a seed so the N-sweep is reproducible. **Critically, all methods must train on and be diagnosed on the identical N-sample subset**: the selected indices are a pure function of `(subset_n, subset_seed)` and nothing else (not the model, not the run order), so a fixed `subset_seed` (default 42, shared across every baseline config and held constant in the notebook overrides) guarantees pseudo_sup, SimCLR, VICReg, and Barlow Twins all see the same images at each N. Do not derive the seed from anything method-specific. Return or persist the selected indices (for example in the effective config/log dir) so the diagnostics cell can reconstruct the same N-pool instead of silently sampling a different one.

### 3.6 Configs

Add `configs/baselines/vicreg_cifar10.yaml` and `configs/baselines/barlow_twins_cifar10.yaml`, modeled on the structure of `configs/meta_exps/meta_random_config.yaml` but **without** the `meta:` section (these use the plain `train_model` path). Required sections: `name`, `dataset` (cifar10, image_size 32), `model` (name + backbone resnet18 + method hyperparams), `train` (optimizer, LR schedule, epochs, batch_size, `knn_monitor: false` by default unless you intentionally want the separate repo train/test smoke monitor, `subset_n: null`, `subset_seed: 42`), and `eval` (linear-eval block — or set `eval: false` if not needed for the go/no-go).

There are no SimCLR/SimSiam baseline YAMLs in this checkout, so do not say "mirror existing baseline configs" unless you add those configs in the same branch. For an initial fair comparison, use explicit shared defaults across VICReg and Barlow Twins, for example the repo's current pseudo-supervised optimizer schedule (`sgd`, weight decay `0.0005`, momentum `0.9`, `base_lr: 0.03`, `warmup_epochs: 5`) unless you decide to create a full LARS-based baseline suite for all compared SSL methods. Only the `model` block and method-specific knobs should differ between the two new baseline configs.

Provide both CIFAR-10 and (optionally) an STL-10 variant since idea.md uses both.

---

## 4. Notebooks (the primary deliverable)

Create one notebook per method, named to match the reference convention:
- `notebooks/vicreg-cifar10-ssl.ipynb`
- `notebooks/barlow-twins-cifar10-ssl.ipynb`

Each must reuse the setup/training cell structure of `notebooks/random-meta-cifar10-ssl.ipynb`, changing only the training-call and the parameter markdown, then add one diagnostics cell:

1. **Cell 0 (markdown):** title + short description of the method and what the notebook produces (trained backbone + spectral/KNN diagnostics across the N-sweep).
2. **Cell 1 (code):** Google Drive mount (`try: from google.colab import drive ... except ModuleNotFoundError`). Copy verbatim from the reference.
3. **Cell 2 (code):** repo clone/sync with optional `GITHUB_TOKEN`, `pip install -r` requirements. Copy verbatim, keeping `PUBLIC_REPO_URL` and the `BRANCH` handling — set `BRANCH` so the notebook pulls this baselines branch (or `main` after merge; make the intent explicit in a comment).
4. **Cell 3 (code):** CIFAR-10 download into the Drive data dir. Copy verbatim.
5. **Cell 4 (markdown):** parameter table documenting the override keys this notebook exposes (method hyperparameters, `subset_n`, batch size, epochs, knn settings) — same table style as the reference.
6. **Cell 5 (code):** the training call. Use `from colab_utils import train_from_colab` and pass `config_file='configs/baselines/<method>_cifar10.yaml'` with an `overrides` dict, following the exact override-dict shape used in the reference's `meta_train_from_colab` call (top-level `name`, nested `train`, `model` blocks). Expose the N-sweep as a Python list the user can edit; loop over N, calling `train_from_colab` per N and collecting `model_path`, `log_dir`, the selected subset indices/path, and (if `train.knn_monitor=True`) `monitor_accuracy`. Do not label the returned `train_from_colab()["accuracy"]` as the Section 2.1 KNN metric: that built-in monitor uses the repo's train/test eval loaders, while Section 2.1 requires the 80/20 split inside the shared N-pool. Prefer setting `train.knn_monitor: False` in these baseline configs to save compute, or keep it only as a smoke-monitor field named `monitor_accuracy`. For each N, override `train.subset_n`, `train.subset_seed`, and a valid `train.batch_size <= subset_n` so `drop_last=True` cannot produce an empty training loader.
7. **Cell 6 (code, new):** post-training diagnostics — load each checkpoint's backbone, call `analysis.spectral.extract_features` + `spectral_diagnostics` + `knn_eval`, and assemble the Section 2.1 comparison table (effective rank and KNN acc per N) plus a top-20 singular-value plot. Keep this cell importable-function-driven so it is unit-testable. For checkpoint loading, instantiate `get_backbone(args.model.backbone)`, call `linear_eval.load_backbone_weights(backbone, checkpoint_path)`, and pass that backbone to `extract_features`; do not instantiate the full SSL model for diagnostics. Build the diagnostics dataset/loader from the same selected indices used for that N's training run, then run extraction following the determinism rules in §3.4 (`backbone.eval()` + `torch.no_grad()`, `shuffle=False`, `drop_last=False`, labels from the loader, deterministic single-view eval transform, `num_workers=0`). Reuse the same selected-index list and loader order for every method's checkpoint so the feature matrices are computed over an identical image set. For each N, call `knn_eval(..., n_train=int(0.8 * N), k=min(20, int(0.8 * N)))`.

Constraints carried over from the reference: every code cell must be valid Python parseable by `ast.parse` (there is an existing test enforcing this — see §5), notebook-safe defaults come from `colab_utils` so do not re-set `num_workers`, and the notebook must run top-to-bottom on a fresh Colab runtime.

Optionally add a third notebook `notebooks/baselines-spectral-comparison-cifar10.ipynb` that loads checkpoints from all methods (pseudo_sup, SimCLR, SimSiam, VICReg, Barlow Twins) and renders the full idea.md Section 2.1 table and the multi-method singular-value figure. Decide based on effort; the per-method notebooks are the required minimum.

---

## 5. Test-driven approach

Write tests **before** the corresponding implementation. Place them in `tests/`, runnable with `pytest` (the repo already uses pytest; see `tests/test_negatives_ratio.py` for the established style — it tests datasets, config YAML values, and notebook validity together). Mirror that file's patterns.

**`tests/test_vicreg.py`**
- Loss is non-negative and a scalar; `forward` returns a dict containing `loss`.
- Identical inputs (`z_a == z_b`) → invariance term ≈ 0.
- A batch with a near-constant feature dimension drives the variance term up (hinge active); a high-variance batch drives it toward 0.
- Correlated dimensions raise the covariance term vs. decorrelated input.
- Output `loss` is differentiable (`loss.backward()` populates grads on a tiny model).

**`tests/test_barlow_twins.py`**
- `off_diagonal` selects exactly the `D*D - D` off-diagonal entries.
- When `z_a == z_b` and dimensions are decorrelated, the cross-correlation is ≈ identity → on-diagonal loss ≈ 0. Use a deterministic centered/standardized toy matrix rather than random data so this is not flaky.
- Fully correlated dimensions inflate the off-diagonal penalty.
- `forward(x1, x2)` on a tiny `resnet18`-castrated backbone returns `{"loss": ...}` and backpropagates.

**`tests/test_spectral_diagnostics.py`**
- `effective_rank` of an orthonormal `[N, D]` matrix (equal singular values) ≈ D; of a rank-1 matrix ≈ 1.
- `spectral_diagnostics` returns singular values in descending order and `explained_variance` monotonically increasing to 1.0.
- `knn_eval` returns 1.0 on a trivially separable toy set (two well-separated clusters), uses an 80/20 split by default when `n_train=None`, and clamps `k` so small-N cases such as N=200 are valid. Avoid assertions on random-label chance accuracy; those are inherently flaky.
- `extract_features` returns `F` of shape `[n_samples, backbone.output_dim]` and matching label length, reading from `backbone` (assert width == 512 for resnet18).
- **Determinism / same-data guarantees:** calling `extract_features` twice on the same backbone + loader returns identical `F` and identical `labels` (byte-for-byte), and two *different* backbones run over the same loader return the same `labels` and the same image ordering. Include a regression test that puts the backbone in `.train()` mode with a BatchNorm layer and asserts `extract_features` still returns deterministic features (proving it forces `.eval()` internally). Assert the diagnostics loader is constructed with `shuffle=False`, `drop_last=False`, and the exact selected indices from the corresponding training N-pool.

**`tests/test_subset_n.py`**
- `subset_n` produces a dataset of exactly N items; same seed → identical indices; different seed → different indices; `.classes` is preserved and `.targets` / `.labels` are subset-local and length N so `knn_monitor` still works.
- **Seed-only invariant:** the selected indices are a pure function of `(subset_n, subset_seed)`. Assert identical indices when the helper is called (a) repeatedly, (b) after consuming arbitrary amounts of global `torch`/`numpy`/`random` randomness beforehand (simulating different model inits / augmentation draws), and (c) interleaved with other subset calls. This proves the pool uses a private RNG and that two different method runs with the same seed get the same training pool. Different `subset_seed` → different pool; sorting makes the index list order-stable.
- `pseudo_supervised_net` receives the selected N-pool through `PseudoSupervisedDataset(explicit_indices=indices)` rather than through a second `source_pool_size` sample. Assert its `source_indices` exactly equal the shared selected indices and `num_pseudo_classes == subset_n`.
- `build_train_loader` with `subset_n=200` and a notebook-style batch-size override produces at least one batch; if `batch_size > subset_n` with `drop_last=True`, the code should either raise a clear error or the notebook should avoid that configuration.

**`tests/test_baseline_configs.py`**
- Both new YAMLs load, declare the expected `model.name` (`vicreg` / `barlow_twins`), `backbone: resnet18`, and a `subset_n` key; coefficients present and of correct type.
- Baseline configs set `train.knn_monitor: false` by default, or tests/documentation must prove the returned monitor accuracy is named separately from the Section 2.1 within-N KNN metric.
- `get_model` builds each method from its config and `get_aug` resolves each `model.name` without raising.

**`tests/test_baseline_notebooks.py`** (mirror `test_negatives_ratio.py`'s notebook test)
- Each new notebook's code cells all pass `ast.parse`.
- Required strings present: `from colab_utils import train_from_colab`, the correct `config_file=...` path, the N-sweep list variable, `subset_n`, `subset_seed`, selected-index reuse in diagnostics, `monitor_accuracy` (or `knn_monitor: False`), and a per-N batch-size rule.

Keep tests CPU-only and tiny (small batches, 1–2 channels/dims where possible, `resnet18` only where a real backbone is needed) so they run fast without a GPU. Avoid network/dataset downloads in tests — use synthetic tensors and `TinyDataset`-style stubs as in the existing test file.

---

## 6. Recommended implementation order

1. `analysis/spectral.py` + `tests/test_spectral_diagnostics.py` (no dependencies; foundational).
2. Subset-N support in `main.build_train_loader`/dataset helpers, including selected-index persistence and the `pseudo_supervised_net` `explicit_indices` path + `tests/test_subset_n.py`.
3. `models/vicreg.py` + `tests/test_vicreg.py`; register in `models/__init__.py` and `augmentations/__init__.py`.
4. `models/barlow_twins.py` + `tests/test_barlow_twins.py`; register likewise.
5. Configs + `tests/test_baseline_configs.py`.
6. Notebooks + `tests/test_baseline_notebooks.py`.
7. Smoke run: one tiny end-to-end run per method (`debug=True` or 1 epoch, N=200) on CPU/Colab to confirm `train_from_colab` → `train_model` → checkpoint save → shared-index diagnostics works end-to-end. If `train.knn_monitor` is enabled for smoke monitoring, keep its output separate from the Section 2.1 KNN metric.

At each step run `pytest` and keep the suite green before moving on.

---

## 7. Acceptance criteria

- `pytest tests/` passes, including all new tests, with no regressions to `test_negatives_ratio.py`.
- `get_model` and `get_aug` resolve `vicreg` and `barlow_twins`; a 1-epoch `train_from_colab` run completes for each and produces a checkpoint. If the repo train/test KNN monitor is enabled, its number is logged only as `monitor_accuracy`, not as the Section 2.1 within-N KNN metric.
- Each new notebook runs top-to-bottom on a fresh Colab runtime and emits the Section 2.1 metrics (effective rank, KNN acc, top-20 singular values) for at least N ∈ {200, 1000, full}.
- The diagnostics module reproduces the `idea.md` reference formulas (verify effective rank and KNN against a hand-checked toy example).
- **Same-data invariant holds end to end:** at each N, every method trains on the identical `(subset_n, subset_seed)` subset and is evaluated by `extract_features` over the identical, deterministically ordered image set (`backbone.eval()`, `shuffle=False`). Re-running diagnostics on a checkpoint reproduces its effective rank and KNN exactly.
- No edits to `arguments.py`, the core epoch loop, or existing model implementations. Existing-file changes are limited to model/augmentation registration plus the tested subset helper path required for the N-sweep.

## 8. Open questions to confirm before/early in implementation

- **Optimizer for parity:** there are no SimCLR/SimSiam baseline YAMLs in this checkout. Either use the explicit shared SGD defaults in §3.6 for VICReg/Barlow Twins and later add matching SimCLR/SimSiam configs, or build a full LARS-based baseline suite for all compared SSL methods. Decide and document the choice in the configs.
- **Projector/expander dimension** at CIFAR scale (suggested 2048; the papers' 8192 is likely unnecessary and slow) — pick and justify in config comments.
- **Linear eval on/off** for the go/no-go: the gate only needs effective rank + KNN, so `eval` can be disabled to save compute; confirm before committing config defaults.
- **STL-10 variant:** idea.md uses CIFAR-10 *and* STL-10; build CIFAR-10 first, add STL-10 configs/notebooks if in scope for this branch.
- **Reference hyperparameters:** confirm VICReg coefficients (25/25/1, with the two variance terms averaged) and Barlow Twins `lambd` (0.0051) against the original papers before locking defaults.
