# Design Guidance: VICReg & Barlow Twins Baselines

**Branch:** `2-vicreg-barlow-twins-baselines`
**Audience:** a coding agent implementing the Section 2.1 "Feature Decorrelation Methods" go/no-go baseline from `.agents/designs/idea.md`.
**Deliverable:** two new SSL methods (VICReg, Barlow Twins) wired into the existing training stack, plus Google Colab notebooks under `notebooks/` that mirror `notebooks/random-meta-cifar10-ssl.ipynb`, all built test-first.

---

## 1. Scope and intent

`idea.md` Section 2.1 is the **week-1 go/no-go experiment**. It asks: do VICReg and Barlow Twins — methods that explicitly decorrelate features — reach the same *effective rank* and KNN accuracy as `pseudo_sup` in the low-data regime (N ≤ 1000)? Both outcomes are publishable, so the job is to produce a *trustworthy, reproducible* comparison, not to win a benchmark.

This branch is responsible for:

1. **Two new pretraining methods** — VICReg and Barlow Twins — that train from random init and plug into the existing `main.train_model` loop with no special-casing beyond model/aug registration.
2. **A baseline suite of Colab notebooks** — one per compared method (`pseudo_sup`, VICReg, Barlow Twins, SimCLR, SimSiam) — that run the N-sweep and report the Section 2.1 metrics: effective rank of the penultimate-layer feature matrix, KNN accuracy (k=20), and the top-20 singular values. All five methods use SGD, held constant for parity so any spectral/KNN difference is attributable to the method, not the optimizer: all five methods share one identical SGD schedule (`base_lr` 0.05, `warmup_epochs` 10, weight decay 5e-4, `grad_clip` 1.0); `pseudo_sup` differs only in its model recipe (cosine-softmax / l2-norm / negatives-ratio), not the optimizer (see below and §3.6).
3. **A checkpoint-initialization option** exposed by every notebook (and all five configs): start training from a saved checkpoint, loading the backbone by default and optionally the projector and predictor (§3.7). This is what enables idea.md's proposed "SimCLR + pseudo_sup init" method.

SimCLR and SimSiam already exist (`models/simclr.py`, `models/simsiam.py`) and are reused as-is — do not reimplement or modify them; they only get new baseline configs and notebooks. `pseudo_sup` is the existing `pseudo_supervised_net`; it gets a baseline notebook that reproduces **one episode** of `notebooks/random-meta-cifar10-ssl.ipynb` (its native SGD pseudo-supervised recipe), trained on the same shared N-pool as the others so it stays directly comparable. The spectrum-surgery experiments (idea.md Section 2.2) are **out of scope** for this branch, but the spectral-diagnostics module built here is the shared foundation for them, so build it cleanly. STL-10 is **out of scope** for this branch (deferred to separate work); build CIFAR-10 only.

Keep changes additive and minimal. Do not modify the core epoch loop, argument parsing, or existing model implementations. The only existing-code edits allowed are the registration hooks listed below, the narrowly scoped tested low-data subset plumbing in `main.build_train_loader` / dataset helpers (§3.5), and the tested checkpoint-init call added to `main.train_model` (§3.7).

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
- `main.py` / dataset helper code → add only (a) the deterministic `train.subset_n` support described in §3.5 and (b) the checkpoint-init call described in §3.7, invoked right after `get_model`. Do not special-case any method in the training loop.
- `configs/` → add new YAML config files (additive; no edits to existing configs).
- A new helper module (e.g. `models/checkpoint_init.py`) for §3.7 — additive, imported by `train_model`.

**Config → args.** `arguments.build_args` loads a YAML, deep-merges `overrides`, and exposes it as nested `Namespace` objects (`args.model.name`, `args.train.base_lr`, …). `args.aug_kwargs` is `{name: model.name, image_size: dataset.image_size}` — so `get_aug` is keyed on `model.name`. This means **`model.name` must equal the `get_aug` key**; plan the names accordingly (`vicreg`, `barlow_twins`).

**Notebook entry point.** Notebooks call `colab_utils.train_from_colab(config_file=..., overrides=..., ...)`, which builds args (forcing notebook-safe defaults: `num_workers=0`, `tensorboard=False`, no DataParallel) and calls `main.train_model`. `train_model` runs the epoch loop, calls `knn_monitor` every `knn_interval`, saves a checkpoint, and (unless `args.eval is False`) runs linear eval. **`train_from_colab` already supports any non-meta model** — VICReg/Barlow Twins need no new colab helper, only a config and a model. Confirm this rather than adding a new wrapper.

**Optimizers.** `optimizers/get_optimizer` supports `sgd`, `lars`, `lars_simclr`, `larc`. LR is scaled `base_lr * batch_size / 256` in `build_optimizer_and_scheduler`. **Decision (revised — supersedes the original LARS choice): the whole baseline suite uses SGD** (with momentum) — all five methods (VICReg, Barlow Twins, SimCLR, SimSiam, and `pseudo_sup`) use SGD, so the optimizer type is held constant across every method and any spectral/KNN difference is attributable to the method, not the optimizer. All five methods share one identical SGD schedule (`base_lr` 0.05, `warmup_epochs` 10, `final_lr` 0, weight decay 5e-4, momentum 0.9, `grad_clip` 1.0) — the four contrastive configs are byte-identical, and `pseudo_sup` carries the same optimizer/schedule values (it differs only in its model recipe and dataset keys, not the optimizer). **LARS was dropped**: the small-N regime forces small batches (`batch_size <= subset_n`, ≤256), and LARS only helps in the large-batch regime — at this scale it gave no benefit and was observed to diverge, so SGD is the appropriate, more stable choice. A shared `base_lr` (0.05) and gradient-norm clipping (`grad_clip`) are used across all methods for stability and parity.

---

## 3. New components to build

### 3.1 `models/vicreg.py`

A projector + VICReg loss. Follow the BN-MLP projector style already in `models/simsiam.py` (3 linear layers with BatchNorm; expander dim configurable, default 2048 for CIFAR — the paper's 8192 is overkill at this scale).

VICReg loss over two projected batches `z_a, z_b` of shape `[N, D]`:
- **Invariance** `s = mse(z_a, z_b)`.
- **Variance** `v(z) = mean_j relu(gamma - sqrt(var(z[:,j]) + eps))`, with `gamma = 1.0`, `eps = 1e-4`. Apply to both `z_a` and `z_b`, then average the two view penalties: `var_loss = (v(z_a) + v(z_b)) / 2`. This matches the reference VICReg implementation while keeping the default coefficient at `25.0`.
- **Covariance** `c(z) = sum_{i != j} Cov(z)_{ij}^2 / D`, where `Cov(z) = (z - z.mean(0)).T @ (z - z.mean(0)) / (N - 1)`. Apply to both.
- **Total** `loss = sim_coeff * s + std_coeff * var_loss + cov_coeff * (c(z_a) + c(z_b))`, default coefficients `sim_coeff = 25.0`, `std_coeff = 25.0`, `cov_coeff = 1.0`.

Return `{"loss": total, "inv_loss": s, "var_loss": ..., "cov_loss": ...}` so each term is logged. Make coefficients, expander dim, `eps`, and `gamma` constructor arguments read from `model_cfg` in `get_model` (use `getattr` with defaults, matching the `pseudo_supervised_net` pattern). **Locked defaults = the original-paper values (Bardes et al., 2022): `sim_coeff = 25.0`, `std_coeff = 25.0`, `cov_coeff = 1.0`, `gamma = 1.0`, `eps = 1e-4`.** The expander dimension is the one deliberate departure from the paper: use **2048** (a CIFAR-scale choice; the paper's 8192 is unnecessary and slow here) and note this in the config comment.

### 3.2 `models/barlow_twins.py`

Projector + Barlow Twins loss over `z_a, z_b` of shape `[N, D]`:
- Batch-normalize each projection along the batch dim using the reference convention: either `nn.BatchNorm1d(projector_dim, affine=False)` on each projected batch, or a manual normalization equivalent to `(z - z.mean(0)) / (z.std(0, unbiased=False) + eps)`. Include `eps` if normalizing manually so tiny CPU tests do not produce NaNs.
- Cross-correlation `C = (z_a_norm.T @ z_b_norm) / N`, shape `[D, D]`.
- **Loss** `= sum_i (1 - C_ii)^2 + lambd * sum_{i != j} C_ij^2`, default `lambd = 0.0051`. Do not name a Python parameter `lambda`; use `lambd` or `offdiag_coeff`.
- Use a helper `off_diagonal(C)` to select off-diagonal elements.

Return `{"loss": total, "on_diag": ..., "off_diag": ...}`. Projector: 3-layer BN-MLP, dim configurable, default **2048** (deliberate CIFAR-scale choice vs. the paper's 8192; note in config). `lambd` and projector dim come from `model_cfg`. **Locked default = the original-paper value (Zbontar et al., 2021): `lambd = 0.0051`.**

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
    # forward pass of the *penultimate* layer (model.backbone); returns RAW features (no L2-norm) + labels
```

`extract_features` must read features from `model.backbone` (the 512-d penultimate layer), not the projector, because idea.md defines the feature matrix `F` as the ResNet-18 penultimate output. Reuse `knn_monitor`'s extraction loop structure, but **return raw features — do NOT L2-normalize before returning.** Normalizing would distort the singular-value spectrum and corrupt the effective-rank measurement; `knn_eval` applies its own cosine normalization internally, and `knn_monitor`'s `F.normalize` step is specific to its own KNN and must not be copied into `extract_features`. Effective rank uses the natural-log entropy formula; document the convention in a docstring.

**Determinism (non-negotiable — every method must be evaluated on the exact same images, in the same order).** The whole point of the Section 2.1 comparison is that pseudo_sup, SimCLR, SimSiam, VICReg, and Barlow Twins are scored on an *identical* feature-evaluation set, so any difference in effective rank or KNN is attributable to the method, not to sampling. `extract_features` **and the caller that builds its loader** must together guarantee a reproducible, method-independent feature matrix. The split of responsibility:

`extract_features` itself owns:
- **`backbone.eval()` + `torch.no_grad()`** before the forward pass. ResNet-18 uses BatchNorm; in train mode it normalizes with *batch* statistics, so a feature would depend on how samples are grouped into batches and would drift run-to-run. Eval mode uses fixed running stats, making each image's feature independent of its batch. This is exactly why `knn_monitor` calls `net.eval()`. Omitting it makes the numbers non-deterministic even with a fixed loader order.
- **Collecting labels from the loader iteration** (`for data, target in loader: feats.append(net(data)); labels.append(target)`) rather than from `dataset.targets`/`.labels`. Combined with the caller's `shuffle=False` this is belt-and-suspenders: features and labels are paired within the same batch, so alignment holds even if the dataset attribute order ever diverges. Then truncate to `n_samples`.

The caller (cell 6 / §3.5 item 2) owns loader construction, and `extract_features` may defensively assert the loader is non-shuffling:
- **`shuffle=False`, `drop_last=False`** on the diagnostics loader, which is passed *into* `extract_features`. `knn_monitor`/`build_eval_loader` rely on `shuffle=False`: the feature bank is built by iterating in order while labels are read by dataset index, so shuffling silently misaligns features and labels and collapses KNN to chance. Beyond alignment, `shuffle=False` preserves the selected-index order so the shared N-pool is evaluated identically for every method. `drop_last=False` ensures all N rows are present.
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
   - `.labels = parent.labels[indices]` for STL-10-style datasets (not exercised in this CIFAR-only branch, but keep the helper general).

   For `pseudo_supervised_net`, do **not** wrap and then let `PseudoSupervisedDataset` sample again via `source_pool_size`; that would double-sample and can break the same-data invariant. Instead pass the selected indices directly as `explicit_indices=indices` to `PseudoSupervisedDataset`, so its pseudo-labels are the positions within the same shared pool. If `explicit_indices` is used, `source_pool_size` is not required.

   Also guard against empty loaders: because the repo uses `drop_last=True` for training, every N-sweep override must choose `batch_size <= subset_n` (or `drop_last` must be changed deliberately and tested). For the notebooks, set batch size per N as `min(DEFAULT_BATCH_SIZE, N)` or another explicit rule that keeps at least one batch.
2. **Feature-matrix N for diagnostics.** Build the diagnostics loader from the exact same selected indices used for training at that N, in the same sorted order. `extract_features(..., n_samples=N)` is still responsible for truncating to N rows, but it should receive a loader whose dataset is already the shared N-pool. Do not evaluate on the first N rows of the full dataset unless that is explicitly the selected pool.

Implement (1) test-first; the subset must be deterministic given a seed so the N-sweep is reproducible. **Critically, all methods must train on and be diagnosed on the identical N-sample subset**: the selected indices are a pure function of `(subset_n, subset_seed)` and nothing else (not the model, not the run order), so a fixed `subset_seed` (default 42, shared across every baseline config and held constant in the notebook overrides) guarantees pseudo_sup, SimCLR, VICReg, and Barlow Twins all see the same images at each N. Do not derive the seed from anything method-specific. Return or persist the selected indices (for example in the effective config/log dir) so the diagnostics cell can reconstruct the same N-pool instead of silently sampling a different one.

### 3.6 Configs

Add four CIFAR-10 configs under `configs/baselines/`: `vicreg_cifar10.yaml`, `barlow_twins_cifar10.yaml`, `simclr_cifar10.yaml`, `simsiam_cifar10.yaml`. Model them on the structure of `configs/meta_exps/meta_random_config.yaml` but **without** the `meta:` section (these use the plain `train_model` path). Required sections: `name`, `dataset` (cifar10, image_size 32), `model` (name + backbone resnet18 + method hyperparams), `train`, and `eval`.

**Shared across all four (for parity):**
- `train.optimizer`: **SGD** (with momentum), with one identical LR schedule (`base_lr`, `warmup_epochs`, `final_lr`, weight decay) reused verbatim across the four files — the SGD baseline suite decided in §2 (LARS was dropped; see §2). Assert the equality in tests.
- `train.knn_monitor: false` by default (the in-training monitor uses the full train/test loaders, not the Section 2.1 within-N split — keep it off to save compute; if ever enabled, treat its number only as `monitor_accuracy`).
- `train.subset_n: null`, `train.subset_seed: 42`.
- `eval: false` — **linear-eval monitoring is OFF during all training runs** (decided). Linear eval, if wanted, is a separate post-training step (see §4) run on saved checkpoints after the go/no-go.

**Differs per file:** only the `model` block. VICReg and Barlow Twins carry their locked paper coefficients and the **2048** projector/expander dim. SimCLR and SimSiam reference the existing models unchanged — do **not** alter their projector architectures (SimCLR's is 256-d, SimSiam's 2048-d by construction); just set `model.name` to `simclr`/`simsiam` and attach the shared `train`/`eval` blocks.

**Fifth config — `pseudo_sup_cifar10.yaml` (single-episode reproduction; same SGD optimizer/schedule as the rest of the suite, differing only in its model recipe).** `model.name: pseudo_supervised_net`, backbone resnet18, reproducing one episode of `notebooks/random-meta-cifar10-ssl.ipynb`: carry the same single-episode settings the meta path uses (`model.cosine_softmax: true`, `model.l2_norm_backbone_features: false`, `train.negatives_ratio`, `train.augment_probability: 1.0`, and the shared SGD optimizer/schedule used across the suite — `sgd`, momentum `0.9`, weight decay `5e-4`, `warmup_epochs: 10`, `base_lr: 0.05`, `grad_clip: 1.0`). Use the plain `train_model` path (not `meta_train_model`) with `subset_n`/`subset_seed` + the `explicit_indices` wiring from §3.5 so it trains on the exact shared N-pool. **All five methods now use the same SGD optimizer and schedule** (LARS was dropped — see §2); `pseudo_sup` carries no optimizer asymmetry at all — its only distinctions are the model recipe (`cosine_softmax`, `l2_norm_backbone_features`) and dataset keys (`negatives_ratio`, `augment_probability`). Same `eval: false`, `knn_monitor: false`, `subset_seed: 42` as the rest. Document the asymmetry in a config comment.

Every config also exposes the checkpoint-init keys from §3.7 (default `init_checkpoint: null`).

CIFAR-10 only; no STL-10 in this branch.

### 3.7 Checkpoint initialization (all methods)

Every training run can optionally start from a saved checkpoint instead of random init. This is required for idea.md's proposed method — **SimCLR warm-started from `pseudo_sup` weights** — and is also useful for resuming/continuing any method.

**Config keys** (under `model`, consumed in `train_model` right after `get_model`, default to a no-op):
- `init_checkpoint: null` — path to a `.pth` saved by `save_checkpoint` (`{"epoch", "state_dict"}`). `null` = random init.
- `init_load_backbone: true` — **load the backbone by default.**
- `init_load_projector: false` — optionally load the projector.
- `init_load_predictor: false` — optionally load the predictor.

**Helper** `load_init_weights(model, checkpoint_path, load_backbone=True, load_projector=False, load_predictor=False)` in the new `models/checkpoint_init.py`:
- `state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]` (flat module state dict with `backbone.` / projector / predictor / `classifier.` prefixes).
- Load each requested submodule **by prefix into the target submodule**, resolving the attribute name per architecture: backbone is always `backbone`; the projector is `projector` for SimCLR/SimSiam/VICReg/Barlow Twins and `proj` for `pseudo_supervised_net`; the predictor is `predictor` for SimSiam and `pred` for `pseudo_supervised_net` (SimCLR/VICReg/Barlow Twins have none). Use `getattr` with this mapping; reuse the `backbone.`-prefix extraction style from `linear_eval.load_backbone_weights`.
- **Backbone is the cross-architecture case** (all methods share the same castrated resnet18, so `backbone.` keys always match) — this is the primary use. Projector/predictor transfer only makes sense between architectures whose submodule shapes match (e.g. same method, or same projector dims); if a requested submodule is missing on the target or its shapes mismatch, **raise a clear error** rather than silently skipping (a requested-but-unloadable init is almost always a mistake). When `load_*` is `False`, that submodule is simply left at its random init.
- Log exactly which submodules were loaded from which checkpoint for provenance.

Keep it additive: `train_model` calls `load_init_weights(...)` only when `model.init_checkpoint` is set; otherwise behaviour is unchanged.

---

## 4. Notebooks (the primary deliverable)

Create one notebook per method, named to match the reference convention — five in the suite:
- `notebooks/pseudo-sup-cifar10-ssl.ipynb`
- `notebooks/vicreg-cifar10-ssl.ipynb`
- `notebooks/barlow-twins-cifar10-ssl.ipynb`
- `notebooks/simclr-cifar10-ssl.ipynb`
- `notebooks/simsiam-cifar10-ssl.ipynb`

The SimCLR/SimSiam notebooks are structural clones of the VICReg/Barlow Twins ones — only the `config_file` and the method-hyperparameter markdown differ (no new model code, since those models already exist). The **`pseudo_sup` notebook reproduces a single episode** of `notebooks/random-meta-cifar10-ssl.ipynb`: same cell structure and the same per-episode hyperparameters (negatives_ratio, cosine_softmax, l2_norm, SGD recipe), but it calls `train_from_colab` with `configs/baselines/pseudo_sup_cifar10.yaml` (the plain `train_model` path on the shared N-pool) rather than `meta_train_from_colab` — so it behaves like one episode while staying on the suite's shared-N-pool + diagnostics machinery. Each notebook reuses the setup/training cell structure of the reference, changing only the training-call and the parameter markdown, then adds one diagnostics cell:

1. **Cell 0 (markdown):** title + short description of the method and what the notebook produces (trained backbone + spectral/KNN diagnostics across the N-sweep).
2. **Cell 1 (code):** Google Drive mount (`try: from google.colab import drive ... except ModuleNotFoundError`). Copy verbatim from the reference.
3. **Cell 2 (code):** repo clone/sync with optional `GITHUB_TOKEN`, `pip install -r` requirements. Copy verbatim, keeping `PUBLIC_REPO_URL` and the `BRANCH` handling — set `BRANCH` so the notebook pulls this baselines branch (or `main` after merge; make the intent explicit in a comment).
4. **Cell 3 (code):** CIFAR-10 download into the Drive data dir. Copy verbatim.
5. **Cell 4 (markdown):** parameter table documenting the override keys this notebook exposes (method hyperparameters, `subset_n`, batch size, epochs, knn settings, **and the §3.7 checkpoint-init keys**) — same table style as the reference.
6. **Cell 5 (code):** the training call. Use `from colab_utils import train_from_colab` and pass `config_file='configs/baselines/<method>_cifar10.yaml'` with an `overrides` dict, following the exact override-dict shape used in the reference's `meta_train_from_colab` call (top-level `name`, nested `train`, `model` blocks). **Expose the checkpoint-init options** as editable notebook variables (`INIT_CHECKPOINT = None`, `INIT_LOAD_BACKBONE = True`, `INIT_LOAD_PROJECTOR = False`, `INIT_LOAD_PREDICTOR = False`) mapped into `overrides['model']` (`init_checkpoint`, `init_load_backbone`, `init_load_projector`, `init_load_predictor`) — so any run can warm-start from a saved checkpoint (e.g. point the SimCLR notebook at a `pseudo_sup` checkpoint to build idea.md's proposed method). Default `INIT_CHECKPOINT = None` (random init). Expose the N-sweep as a Python list the user can edit; loop over N, calling `train_from_colab` per N and collecting `model_path`, `log_dir`, the selected subset indices/path, and (if `train.knn_monitor=True`) `monitor_accuracy`. Do not label the returned `train_from_colab()["accuracy"]` as the Section 2.1 KNN metric: that built-in monitor uses the repo's train/test eval loaders, while Section 2.1 requires the 80/20 split inside the shared N-pool. Prefer setting `train.knn_monitor: False` in these baseline configs to save compute, or keep it only as a smoke-monitor field named `monitor_accuracy`. For each N, override `train.subset_n`, `train.subset_seed`, and a valid `train.batch_size <= subset_n` so `drop_last=True` cannot produce an empty training loader.
7. **Cell 6 (code, new):** post-training diagnostics — load each checkpoint's backbone, call `analysis.spectral.extract_features` + `spectral_diagnostics` + `knn_eval`, and assemble the Section 2.1 comparison table (effective rank and KNN acc per N) plus a top-20 singular-value plot. Keep this cell importable-function-driven so it is unit-testable. For checkpoint loading, instantiate `get_backbone(args.model.backbone)`, call `linear_eval.load_backbone_weights(backbone, checkpoint_path)`, and pass that backbone to `extract_features`; do not instantiate the full SSL model for diagnostics. Build the diagnostics dataset/loader from the same selected indices used for that N's training run, then run extraction following the determinism rules in §3.4 (`backbone.eval()` + `torch.no_grad()`, `shuffle=False`, `drop_last=False`, labels from the loader, deterministic single-view eval transform, `num_workers=0`). Reuse the same selected-index list and loader order for every method's checkpoint so the feature matrices are computed over an identical image set. For each N, call `knn_eval(..., n_train=int(0.8 * N), k=min(20, int(0.8 * N)))`.

**Linear eval is post-training only.** All five configs set `eval: false`, so no linear probe runs during the go/no-go sweep (the gate needs only effective rank + KNN — recall `train_model` skips inline eval exactly when `args.eval is False`). If you want linear-probe accuracy as a secondary metric, add an optional final cell that runs *after* the sweep — load each saved checkpoint and call `colab_utils.linear_eval_from_colab(...)`. **Important:** `linear_eval.main` reads `args.eval.batch_size`, `args.eval.optimizer`, `args.eval.base_lr`, etc., so a config with `eval: false` has no eval block to read and would raise. The post-hoc cell must therefore pass an `eval` block via `overrides={'eval': { ... }}` (optimizer, base_lr, warmup_epochs, num_epochs, batch_size) to repopulate `args.eval` for that standalone run. Never inline it during training.

Constraints carried over from the reference: every code cell must be valid Python parseable by `ast.parse` (there is an existing test enforcing this — see §5), notebook-safe defaults come from `colab_utils` so do not re-set `num_workers`, and the notebook must run top-to-bottom on a fresh Colab runtime.

Optionally add a sixth, comparison notebook `notebooks/baselines-spectral-comparison-cifar10.ipynb` that loads checkpoints from all methods (pseudo_sup, SimCLR, SimSiam, VICReg, Barlow Twins) and renders the full idea.md Section 2.1 table and the multi-method singular-value figure. Decide based on effort; the per-method notebooks are the required minimum.

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
- **Determinism / same-data guarantees:** calling `extract_features` twice on the same backbone + loader returns identical `F` and identical `labels` (byte-for-byte), and two *different* backbones run over the same loader return the same `labels` and the same image ordering. Include a regression test that puts the backbone in `.train()` mode with a BatchNorm layer and asserts `extract_features` still returns deterministic features (proving it forces `.eval()` internally) and returns **raw, un-normalized** features (e.g. feature norms are not all ≈ 1). Separately, test the diagnostics loader *builder* (caller side): it produces a loader with `shuffle=False`, `drop_last=False`, over the exact selected indices from the corresponding training N-pool; and `extract_features` raises (or its defensive assert fires) if handed a shuffling loader.

**`tests/test_subset_n.py`**
- `subset_n` produces a dataset of exactly N items; same seed → identical indices; different seed → different indices; `.classes` is preserved and `.targets` / `.labels` are subset-local and length N so `knn_monitor` still works.
- **Seed-only invariant:** the selected indices are a pure function of `(subset_n, subset_seed)`. Assert identical indices when the helper is called (a) repeatedly, (b) after consuming arbitrary amounts of global `torch`/`numpy`/`random` randomness beforehand (simulating different model inits / augmentation draws), and (c) interleaved with other subset calls. This proves the pool uses a private RNG and that two different method runs with the same seed get the same training pool. Different `subset_seed` → different pool; sorting makes the index list order-stable.
- `pseudo_supervised_net` receives the selected N-pool through `PseudoSupervisedDataset(explicit_indices=indices)` rather than through a second `source_pool_size` sample. Assert its `source_indices` exactly equal the shared selected indices and `num_pseudo_classes == subset_n`.
- `build_train_loader` with `subset_n=200` and a notebook-style batch-size override produces at least one batch; if `batch_size > subset_n` with `drop_last=True`, the code should either raise a clear error or the notebook should avoid that configuration.

**`tests/test_checkpoint_init.py`**
- `load_init_weights` with `load_backbone=True` (default) loads only backbone weights and leaves projector/predictor at random init; assert backbone params equal the checkpoint's and a projector param does not.
- **Cross-architecture backbone transfer:** save a `pseudo_supervised_net` checkpoint, load its backbone into a freshly built `SimCLR` model; assert every `backbone.*` param matches and the load succeeds (this is the proposed-method path).
- `load_projector=True` / `load_predictor=True` load the correct submodule for each architecture (resolve `projector`/`proj`, `predictor`/`pred`); SimSiam round-trips predictor, `pseudo_supervised_net` round-trips `pred`.
- Requesting a submodule the target lacks (e.g. `load_predictor=True` into SimCLR) or a shape-mismatched projector **raises a clear error**, not a silent skip.
- `init_checkpoint: null` is a no-op: `train_model` builds the model identically to the no-init path.

**`tests/test_baseline_configs.py`**
- All five YAMLs (`pseudo_sup`, `vicreg`, `barlow_twins`, `simclr`, `simsiam`) load and declare the expected `model.name`, `backbone: resnet18`, `train.subset_n`/`train.subset_seed`, `eval: false`, and the `model.init_checkpoint` (default null) + `init_load_*` keys.
- The four contrastive configs (`vicreg`, `barlow_twins`, `simclr`, `simsiam`) declare `train.optimizer` name `sgd` and share an identical `train` optimizer/LR block — assert equality across the four so the SGD schedule is held constant for parity. The `pseudo_sup` config carries the same SGD optimizer/schedule (sgd, weight decay 5e-4, warmup 10, base_lr 0.05) and differs only in its model recipe — assert it declares `sgd` and carries `cosine_softmax`/`l2_norm_backbone_features`.
- VICReg/Barlow Twins configs carry the locked paper coefficients (`sim/std/cov = 25/25/1`, `lambd = 0.0051`) and projector dim `2048`, of correct type.
- `train.knn_monitor: false` in all five (or tests/documentation prove the monitor number is named separately from the Section 2.1 within-N KNN metric).
- `get_model` builds each method from its config and `get_aug` resolves each `model.name` without raising.

**`tests/test_baseline_notebooks.py`** (mirror `test_negatives_ratio.py`'s notebook test)
- Each of the five notebooks' code cells all pass `ast.parse`.
- Required strings present in each: `from colab_utils import train_from_colab`, the correct `config_file=...` path, the N-sweep list variable, `subset_n`, `subset_seed`, the checkpoint-init variables (`INIT_CHECKPOINT`, `INIT_LOAD_BACKBONE`, `INIT_LOAD_PROJECTOR`, `INIT_LOAD_PREDICTOR`), selected-index reuse in diagnostics, `monitor_accuracy` (or `knn_monitor: False`), and a per-N batch-size rule.
- The `pseudo_sup` notebook references `pseudo_sup_cifar10.yaml` and carries the single-episode settings (`cosine_softmax`, `l2_norm_backbone_features`, `negatives_ratio`).

Keep tests CPU-only and tiny (small batches, 1–2 channels/dims where possible, `resnet18` only where a real backbone is needed) so they run fast without a GPU. Avoid network/dataset downloads in tests — use synthetic tensors and `TinyDataset`-style stubs as in the existing test file.

---

## 6. Recommended implementation order

1. `analysis/spectral.py` + `tests/test_spectral_diagnostics.py` (no dependencies; foundational).
2. Subset-N support in `main.build_train_loader`/dataset helpers, including selected-index persistence and the `pseudo_supervised_net` `explicit_indices` path + `tests/test_subset_n.py`.
3. `models/checkpoint_init.py` (`load_init_weights`) + the `train_model` init hook + `tests/test_checkpoint_init.py`.
4. `models/vicreg.py` + `tests/test_vicreg.py`; register in `models/__init__.py` and `augmentations/__init__.py`.
5. `models/barlow_twins.py` + `tests/test_barlow_twins.py`; register likewise.
6. Configs for all five methods (four-method SGD suite sharing one schedule + `pseudo_sup` SGD single-episode config, all `eval: false`, all exposing the init keys) + `tests/test_baseline_configs.py`.
7. Notebooks for all five methods (SimCLR/SimSiam are config-only clones of the VICReg/Barlow Twins notebooks; `pseudo_sup` reproduces one meta episode) + `tests/test_baseline_notebooks.py`.
8. Smoke run: one tiny end-to-end run per method (`debug=True` or 1 epoch, N=200) on CPU/Colab to confirm `train_from_colab` → `train_model` → checkpoint save → shared-index diagnostics works end-to-end, plus one run that warm-starts a SimCLR model from a `pseudo_sup` checkpoint via `init_checkpoint`. If `train.knn_monitor` is enabled for smoke monitoring, keep its output separate from the Section 2.1 KNN metric.

At each step run `pytest` and keep the suite green before moving on.

---

## 7. Acceptance criteria

- `pytest tests/` passes, including all new tests, with no regressions to `test_negatives_ratio.py`.
- `get_model` and `get_aug` resolve `vicreg` and `barlow_twins`; all five baseline configs (`pseudo_sup`, VICReg, Barlow Twins, SimCLR, SimSiam) drive a 1-epoch `train_from_colab` run to a saved checkpoint with `eval: false` — the four contrastive/decorrelation methods under the shared SGD schedule, `pseudo_sup` under its native SGD single-episode recipe (all five use SGD). If the repo train/test KNN monitor is enabled, its number is logged only as `monitor_accuracy`, not as the Section 2.1 within-N KNN metric.
- **Checkpoint init works:** with `init_checkpoint` set, a run loads the backbone by default and optionally projector/predictor; a SimCLR model successfully warm-starts its backbone from a `pseudo_sup` checkpoint (the proposed-method path), and an unloadable requested submodule errors clearly.
- Each of the five notebooks runs top-to-bottom on a fresh Colab runtime and emits the Section 2.1 metrics (effective rank, KNN acc, top-20 singular values) for at least N ∈ {200, 1000, full}.
- The diagnostics module reproduces the `idea.md` reference formulas (verify effective rank and KNN against a hand-checked toy example).
- **Same-data invariant holds end to end:** at each N, every method trains on the identical `(subset_n, subset_seed)` subset and is evaluated by `extract_features` over the identical, deterministically ordered image set (`backbone.eval()`, `shuffle=False`). Re-running diagnostics on a checkpoint reproduces its effective rank and KNN exactly.
- No edits to `arguments.py`, the core epoch loop, or existing model implementations. Existing-file changes are limited to model/augmentation registration, the tested subset helper path, and the tested checkpoint-init call in `train_model`.

## 8. Resolved decisions (binding)

These were open questions; the project owner has decided each. Treat as binding.

- **Optimizer / baseline suite (revised):** build a single **SGD-based** baseline suite covering all five methods — all five (VICReg, Barlow Twins, SimCLR, SimSiam, and `pseudo_sup`) share one identical SGD optimizer/schedule (base_lr 0.05, weight decay 5e-4, warmup 10, grad_clip 1.0); `pseudo_sup` differs only in its model recipe, not the optimizer. **LARS was dropped** (it only helps at large batch; these runs use batch ≤256, where it gave no benefit and diverged), giving full optimizer parity across all five at small batch. SimCLR/SimSiam reuse their existing models (config + notebook only, no model edits). (§2, §3.6, §4)
- **`pseudo_sup` in the suite:** add a fifth notebook/config that reproduces **one episode** of `notebooks/random-meta-cifar10-ssl.ipynb` via the plain `train_model` path on the shared N-pool. It uses the **same SGD optimizer and schedule** as the rest of the suite (all five baselines share `sgd` / base_lr 0.05 / weight decay 5e-4 / warmup 10 / grad_clip 1.0); the only remaining asymmetry is `pseudo_sup`'s model recipe — cosine-softmax, l2-norm, and the negatives-ratio dataset — not the optimizer. (§3.6, §4)
- **Checkpoint initialization (all methods):** every notebook/config can warm-start from a saved checkpoint, loading the **backbone by default** and **optionally the projector and predictor** (§3.7). This enables idea.md's "SimCLR + pseudo_sup init" proposed method. Requested-but-unloadable submodules error rather than silently skip. (§2, §3.7, §4)
- **Projector/expander dimension:** **2048** for the new VICReg/Barlow Twins models. SimCLR/SimSiam keep their existing architecture dims unchanged. (§3.1, §3.2, §3.6)
- **Linear eval:** **off for all training runs** (`eval: false`) to save compute; run linear eval only as a separate post-training step on saved checkpoints. (§3.6, §4)
- **STL-10:** **out of scope** for this branch (deferred to other work). CIFAR-10 only. (§1, §3.6)
- **Reference hyperparameters:** use the **original-paper values** as defaults — VICReg `25/25/1` (two variance terms averaged, `gamma=1`, `eps=1e-4`), Barlow Twins `lambd = 0.0051`. The 2048 projector dim is the only intentional departure (a CIFAR-scale choice). (§3.1, §3.2)
