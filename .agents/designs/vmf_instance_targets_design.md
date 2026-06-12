# Design: Softened Instance Supervision via Per-Instance vMF Targets

Status: proposed (2026-06-12)
Related: `.agents/designs/research_plan_instance_discrimination_limitations.md`
(hypothesis H1: instance CE maximizes within-class scatter; fix family:
"soften the index targets").

## 1. Motivation

Instance-index supervision (one-hot CE or pairwise sigmoid over 50k image
indices) produces features with kNN >> linear probe. The working diagnosis is
that the loss repels *every* same-class pair with full, uniform strength:
classes end up as thousands of tight instance clusters scattered across the
sphere — locally clean, linearly inseparable. The fix family explored here
keeps instance-level supervision but makes the targets *distributions*, so
that ambiguous / near-duplicate / augmentation-unstable instances exert weaker
repulsion, and the inseparability is absorbed by target uncertainty instead of
by backbone geometry.

## 2. Key Observation: Instance CE Is Already a Degenerate vMF Model

Softmax instance classification over cosine logits with temperature tau is
exactly the posterior of a von Mises-Fisher mixture with:

- one component per image,
- uniform mixing weights pi_i = 1/N,
- a single shared concentration kappa_i == 1/tau,
- point estimates for mean directions (the classifier rows / label embeddings).

So "cast indices into vMF distributions" = promote the implicit constants into
per-instance parameters (mu_i, kappa_i). Nothing exotic is being bolted on;
the baseline is recovered as a special case, which makes ablations clean.

## 3. vMF Background

Density on the unit sphere S^{d-1}:

```text
f(z; mu, kappa) = C_d(kappa) * exp(kappa * mu^T z)
C_d(kappa)      = kappa^{d/2-1} / ( (2*pi)^{d/2} * I_{d/2-1}(kappa) )
```

with `I_v` the modified Bessel function of the first kind. Useful quantities:

- Mean resultant: `E[z] = A_d(kappa) * mu`, where
  `A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa)` (monotone in kappa, in [0,1)).
- MLE from V unit samples (Banerjee et al. 2005 approximation):

```text
Rbar      = || (1/V) * sum_v z_v ||
kappa_hat = Rbar * (d - Rbar^2) / (1 - Rbar^2)
```

- KL between two vMFs in the same dimension (closed form):

```text
KL(P1 || P2) = log C_d(k1) - log C_d(k2) + A_d(k1) * (k1 - k2 * mu1^T mu2)
```

- Expected-likelihood kernel (closed form), for distribution-to-distribution
  similarity:

```text
int f(z; mu1,k1) f(z; mu2,k2) dz = C_d(k1) C_d(k2) / C_d(|| k1*mu1 + k2*mu2 ||)
```

## 4. Formulation A — vMF Mixture Classifier (recommended first)

Each index i owns vMF(mu_i, kappa_i). Backbone produces z = normalized
embedding. Loss = CE on the mixture posterior:

```text
p(i | z) = pi_i * C_d(kappa_i) * exp(kappa_i * mu_i^T z)
           / sum_j pi_j * C_d(kappa_j) * exp(kappa_j * mu_j^T z)
L = -log p(y | z)
```

### Why this softens repulsion (gradient analysis)

```text
-dL/dz = kappa_y * mu_y - sum_j p_j * kappa_j * mu_j
```

Each negative j repels with strength `p_j * kappa_j`. A low-concentration
instance repels weakly. kappa_i is the per-instance repulsion dial — the
principled version of label smoothing over indices. With all kappa equal the
gradient reduces exactly to standard instance CE.

### Parameter sources (the important design choice)

Do NOT learn (mu_i, kappa_i) by backprop. Two reasons: (a) the starved-head
problem — 50k rows each visited ~once/epoch never converge (the PIC paper's
core pathology); (b) learned kappa can collapse to dodge the loss (Section 7).
Instead:

- **mu_i: EMA memory bank.** `mu_i <- normalize(m * mu_i + (1-m) * z_i)`
  updated for the in-batch indices each step (InstDisc-style). Targets are
  always fresh; no parametric head.
- **kappa_i: estimated from view scatter.** Maintain per-instance resultant
  statistics over the EMA of augmented-view embeddings and apply the Banerjee
  MLE. Augmentation-unstable images automatically get low kappa => weak
  repulsion => data-driven softening with no new hyperparameter. Cheapest
  implementation: track `Rbar_i` as an EMA of `mu_i^T z_i` per visit
  (cosine of new view to running mean direction is a serviceable proxy for
  resultant length); exact variant: store EMA of the *unnormalized* mean
  `s_i <- m * s_i + (1-m) * z_i` and use `Rbar_i = ||s_i||` — one buffer,
  mu_i = s_i/||s_i|| comes for free. The exact variant is preferred: one
  [N, d] float buffer gives both mu and Rbar.

### Numerics

- `log C_d(kappa)` needs scaled Bessel `ive` (or standard bounds, Kumar &
  Tsvetkov 2019) at d=256. Precompute a 1-D lookup table for
  `log C_d(kappa)` and `A_d(kappa)` over a kappa grid at startup (kappa is
  never backpropagated under the estimation scheme, so a table + linear
  interpolation suffices; torch-free helper, unit-testable against
  scipy.special.ive).
- Compute cost is a non-issue: 512 x 50,000 logits and one [50k, 256] buffer
  fit easily on a Colab GPU.

## 5. Formulation B — Fully Distributional (predict parameters)

The network outputs a distribution per image: mu(x) from the normalized
projection head, kappa(x) from a small scalar head (softplus + floor). Losses:

- **KL to the target vMF** of the image's own index (closed form, Section 3).
- **Cross-instance term**: contrastive over distribution similarities using
  the expected-likelihood kernel, or MC-InfoNCE (Kirchhof et al. 2023) —
  sample z ~ vMF(mu(x), kappa(x)) with the rejection sampler and run InfoNCE
  on samples.

Predicted kappa(x) becomes a per-image confidence; the model can hedge on
ambiguous inputs. This is the probabilistic-embedding line (HIB, PFE,
MCInfoNCE). Higher implementation cost (Bessel terms now need gradients —
use the standard differentiable bound for log C_d, or the ive-based exact
form), so this is the phase-2 variant after Formulation A is characterized.

## 6. Relation to the Existing Codebase

- The hierarchical vMF/OT models already do spherical EM with normalized
  mean-direction updates. Formulation A is the *leaf level* of that tree:
  one component per instance, estimated concentration. Conceptually the full
  model becomes one object — a depth-annealed vMF hierarchy from coarse
  clusters down to softened instances.
- The pairwise **sigmoid image-index regularizer** is the hard +/-1 version of
  exactly this loss. Formulation A is a drop-in successor: replace
  `sigmoid_label_embeddings` (learned nn.Embedding) with the EMA bank `s_i`
  and the BxB sigmoid loss with the vMF posterior CE (or keep the BxN
  in-batch-negatives restriction first for a minimal diff).
- Reuse: `_is_degenerate_pair`-style guards, normalization conventions, and
  the buffer/state-dict patterns from
  `models/hierarchical_balanced_vmf_self_labeling_net.py`.

### Proposed minimal implementation (new, additive — no binary/K-way paths touched)

- `tools/vmf_math.py` (torch-free where possible): `log_c_d(kappa, d)`,
  `a_d(kappa, d)` (table + interpolation), `banerjee_kappa(rbar, d)`,
  `vmf_kl(mu1, k1, mu2, k2)`. Unit tests against scipy.
- `models/vmf_instance_net.py`: backbone + optional projector + normalize;
  buffers `bank_s [N, d]` (EMA unnormalized mean; mu and Rbar derived),
  `bank_count [N]`; forward computes the mixture-posterior CE with
  kappa from `banerjee_kappa(clamp(Rbar))`; knobs:
  `ema_momentum`, `kappa_mode: fixed | estimated`, `kappa_fixed`,
  `kappa_floor`, `kappa_ceil`, `warmup_epochs_fixed_kappa`,
  `negatives: in_batch | full_bank`.
- Register in `models/__init__.py`; config
  `configs/vmf_instance_cifar_colab.yaml`; notebook
  `notebooks/vmf-instance-cifar10-ssl.ipynb`; changelog entry in `changes/`.
- Later (optional): expose the same loss as a regularizer option inside the
  hierarchical models, replacing the sigmoid term.

## 7. Risks and Design Guards

1. **kappa collapse / triviality.** If kappa were learnable, shrinking all
   kappa makes the task easy and uninformative. Guard: estimation-not-learning
   (kappa cannot move to dodge the loss), plus `kappa_floor`/`kappa_ceil`
   clamps.
2. **Feedback loop with estimated kappa.** Scattered views -> low kappa ->
   weak gradient -> stays scattered. Early in training *everything* is
   scattered. Guard: `warmup_epochs_fixed_kappa` — train with fixed kappa
   (= the standard-CE special case) until embeddings stabilize, then switch
   to estimated kappa. The switch epoch is a hyperparameter to sweep.
3. **EMA staleness vs lr.** Bank directions lag the backbone; standard
   InstDisc-era issue. Guard: momentum sweep (0.5–0.99); monitor
   `mean(mu_i^T z_i)` per epoch as a staleness metric.
4. **Bessel numerics at d=256.** Use `ive`-based log C_d and a precomputed
   table; unit-test the table against scipy over the clamped kappa range.
5. **It may still under-perform coarsened labels.** Softened instance targets
   reduce *within-class* repulsion only where augmentation instability or
   near-duplication reveals it; semantically-same-but-visually-distinct pairs
   keep repelling. Expectation: improves the linear/kNN gap, does not fully
   match cluster-level supervision. That itself is an informative result for
   the main research question.

## 8. Evaluation Plan (ties into the main research plan)

Primary metrics: kNN, linear probe, normalized-linear probe, small-MLP probe
(probe ladder from Phase 1), tracked periodically during training.

Runs (Colab single GPU, CIFAR-10, resnet18, 100–200 epochs, matched budget):

1. Baseline: standard instance CE == Formulation A with `kappa_mode: fixed`
   (sanity: must reproduce the ~70/58 inversion).
2. Formulation A, estimated kappa, in-batch negatives.
3. Formulation A, estimated kappa, full-bank negatives.
4. (Optional) kappa-percentile ablation: clamp kappa at the p-th percentile to
   sweep the overall softening strength.

Predictions if the H1 diagnosis is right:

- The kNN-minus-linear gap shrinks for runs 2–3 vs run 1 at matched kNN.
- kappa_i correlates inversely with within-class neighbor distance (ambiguous
  instances are exactly the ones that stop repelling).
- Gains are real but smaller than label-coarsening (Phase 3 sweep), since
  softening only fixes the repulsion the data exposes.

Diagnostics to log per epoch: kappa histogram, mean/min Rbar, staleness
`mean(mu_i^T z_i)`, and the probe ladder every k epochs.

## 9. References

- Wu et al. 2018, InstDisc — non-parametric instance discrimination, memory
  bank (the mu-EMA precedent).
- Cao et al. 2020, PIC — parametric instance classification pathologies and
  fixes (the starved-head argument).
- Banerjee et al. 2005 — vMF mixtures, the kappa MLE approximation.
- Hasnat et al. 2017, von Mises-Fisher mixture loss for face verification.
- Scott et al. 2021, vMF loss: predictions as distributions.
- Oh et al. 2019, HIB; Shi & Jain 2019, PFE — probabilistic embeddings.
- Kirchhof et al. 2023, MCInfoNCE — vMF-likelihood contrastive learning
  recovers aleatoric uncertainty.
- Kumar & Tsvetkov 2019 — differentiable bounds for vMF normalizers.
- Wang & Isola 2020 — alignment/uniformity (framing for what kappa modulates).
