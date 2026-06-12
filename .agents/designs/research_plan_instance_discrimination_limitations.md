# Research Plan: Why Instance-Level Supervision Fails as a Pre-Training Strategy

Status: proposed (2026-06-12)
Constraints: Colab single GPU, CIFAR-10, resnet18. Existing failing-run checkpoint:
`/Users/xmc28/Desktop/projects/checkpoints/pseudo_sup/exps/0524111121_meta_training_cifar10/episode_8/episode_8_0525083402.pth`
(~70% kNN, ~58% linear probe).

## The Core Anomaly

Instance-index supervision produces features where kNN (70%) >> linear probe (58%).
Established SSL (SimCLR/SimSiam/clustering) always shows linear probe >= kNN.
This inversion is the key diagnostic signal: the features have good *local* class
structure (nearest neighbors share class) but classes are not *linearly separable*
(each class occupies a scattered, non-convex region).

## Revised Hypotheses

**H1 — Independent per-instance targets fragment class geometry (primary).**
The failure is not merely that image-index supervision uses hard one-hot labels.
It is that the targets are *independent across images*: two CIFAR-10 images from
the same semantic class are assigned unrelated objectives. Augmentation
invariance can still make each image's views stable, and natural image statistics
can still put visually similar same-class images near each other locally, giving
good kNN. But the training objective provides no force that makes all members of
a semantic class share one convex/linear region. The resulting class support is a
union of many small instance neighborhoods, often tangled with other classes:
locally useful, globally nonlinearly arranged, and therefore weak for a linear
probe. Prediction: a small MLP probe and graph/cluster metrics should recover
substantially more of the kNN signal than a linear probe.

**H2 — The kNN/linear inversion is a geometry problem, not just weak semantics.**
A 70% kNN score means semantic information is present in the representation, but
it may be encoded in neighborhood topology rather than in linearly separable
directions. The key diagnostic is not "does the backbone know classes?" but
"what probe complexity is required to read out classes?" Prediction: normalized
linear < shallow MLP <= kNN, with high per-class component counts and poor Fisher
linear separation.

**H3 — Evaluation asymmetry can exaggerate the gap.**
`tools/knn_monitor.py` L2-normalizes features; `linear_eval.py` probes raw
backbone output. Feature-norm variance, anisotropy, or covariance collapse can
make the linear probe look worse even when angular neighborhoods are decent.
This is an artifact candidate, not the main causal story. Cheap falsification:
linear probe on normalized features, BN/whitened features, and normalized linear
classifier weights.

**H4 — Parametric heads may hide or absorb instance identity.**
With a learnable `nn.Embedding` over indices (sigmoid net) or a wide FC
classifier, the head can absorb much of the arbitrary image-index mapping. The
backbone may only need to expose nuisance/detail directions sufficient for the
head, not a class-aligned representation. Prediction: head weights classify
indices better than backbone geometry predicts; backbone features have strong
anisotropy or dominant nuisance components; removing/limiting the head or adding
a projector changes where the damage appears.

**H5 — Loss placement controls how much instance identity contaminates the probe
features.**
In SimCLR/SimSiam, the SSL loss is applied after a projector, and the backbone
can remain more semantic while the projection head carries loss-specific detail.
If image-index loss is applied directly at the backbone output, instance identity
is burned into the exact features used by linear evaluation. Prediction:
layer-wise probes show earlier/intermediate features with a smaller kNN-linear
gap; adding a projector before the index loss improves backbone linear probing
without necessarily improving index-prediction accuracy.

**H6 — Label granularity is the controlling intervention.**
Coarsening targets should interpolate between the bad instance-index regime
(K=N, unrelated targets per image) and cluster/self-labeling regimes where
multiple images share a target. If the gap shrinks or flips sign as K decreases,
the root cause is target granularity / target independence rather than the exact
loss family. Prediction: kNN-minus-linear is largest at K=N, smaller at
K=10k/1k, and may become normal at semantically meaningful K.

**Deprioritized fix hypothesis — Distributional vMF targets alone are unlikely
to solve the root cause.**
Replacing one-hot index labels with fixed per-index vMF distributions keeps the
same independence structure: each image still has an unrelated target. It is a
clean diagnostic for "hardness of target" but not a strong remedy for the
linear-probe gap. Memory-bank vMF targets are also not appropriate for this
question because they turn the method toward contrastive/moving-prototype SSL.
Therefore the next work should prioritize post-hoc diagnostics and granularity
ablations over new per-instance target parameterizations.

## Phases

### Phase 0 — Literature grounding (no compute)

Position the result against: Exemplar-CNN (Dosovitskiy et al. 2014, small surrogate
class pools work), InstDisc (Wu et al. 2018, works via *non-parametric* memory bank
+ NCE, deliberately avoiding a parametric instance classifier), PIC (Cao et al.
2020, parametric instance classification *can* match contrastive but only with
specific fixes — identify which fixes our setup lacks), Wang & Isola 2020
(alignment/uniformity), neural-collapse literature (CE drives class means to a
simplex ETF — with N=50,000 singleton "classes" this forces uniform instance
scatter). Output: `.agents/designs/lit_notes_instance_discrimination.md` with a
table mapping each prior method's design choice to our hypotheses.

### Phase 1 — Post-hoc diagnostics on the existing checkpoint (cheap, ~1 GPU-hour)

All run from one new notebook + one new tool module; no retraining.

1. **Probe ladder** (decisive for H1): on frozen backbone features, train and
   compare kNN, linear probe, linear probe on normalized features (+BN variant,
   H2), and a 2-layer MLP probe. If MLP ≈ kNN >> linear, H1 is confirmed:
   information is present but nonlinearly arranged. If normalized-linear ≈ kNN,
   the anomaly was largely H2 (evaluation artifact) — important to rule out first.
2. **Class-scatter geometry** (H1): intra-class vs inter-class cosine-distance
   distributions; per-class number of connected components / cluster count of
   instance-clusters (e.g. DBSCAN or graph components on the class submanifold);
   Fisher ratio per class. Compare against a SimCLR or SimSiam checkpoint at
   matched kNN accuracy (train a short baseline if none is saved).
3. **Spectral diagnostics** (H3): eigenspectrum of the feature covariance,
   effective rank, alignment/uniformity metrics (Wang–Isola). Compare with the
   baseline checkpoint.
4. **Layer-wise probing** (H4): linear probe + kNN at each resnet stage. If
   penultimate-stage features probe *better* than final features, the index loss
   is corrupting only the top of the network.

Deliverable: `tools/feature_diagnostics.py` (torch, unit-tested),
`notebooks/diagnostics/feature-diagnostics-cifar10.ipynb`, results table in
`.agents/designs/`. Minimal changes to `linear_eval.py`: optional
`--normalize_features` and `--probe mlp` flags.

### Phase 2 — Controlled comparison at matched budget (~3–4 Colab runs, 100–200 epochs)

Same backbone, augmentation (`SuperimposeTransform`), batch size, epochs, optimizer:

- A: SimCLR (existing config) — reference with normal kNN/linear relationship.
- B: parametric index softmax (`pseudo_supervised_net`) — pure instance CE.
- C: sigmoid index net (`sigmoid_pseudo_supervised_net`) — pairwise variant.
- D (H4 test): B/C with a projector before the index loss, probing the backbone.

Track kNN and (new) periodic linear-probe accuracy during training so the
divergence point is visible, not just final numbers. Minimal change: add an
optional cheap linear-probe monitor to `train_model` alongside the kNN monitor
(reusing the eval hooks already in `main.py`).

### Phase 3 — Label-granularity sweep (H5, the headline experiment; ~4–5 runs)

Replace instance indices with K-means (or spherical k-means, reusing
`_build_flat_from_embeddings`-style fitting) pseudo-labels over the pool,
K ∈ {10, 100, 1000, 10000, 50000}; K=50000 recovers instance discrimination.
Refresh labels every R epochs (DeepCluster-style) or keep fixed from an early
checkpoint — start with fixed for cleanliness. Plot kNN, linear, MLP probe vs K.
Expected result: the kNN-minus-linear gap monotonically shrinks and flips sign as
K decreases, with peak linear accuracy at intermediate K. This (a) explains the
failure as a granularity problem and (b) justifies the hierarchical OT
self-labeling direction as "annealed granularity."

Implementation: small dataset wrapper or a `label_granularity_k` option in
`PseudoSupervisedDataset` (minimal-change: new dataset module, register in
`datasets/__init__.py`, one new config + notebook).

First concrete diagnostic: `low_rank_multitarget_pseudo_supervised_net` sweeps
a fixed balanced target matrix with rank K and membership size m. K=N,m=1
approximates independent instance supervision; K<<N,m=5 forces overclustered
target sharing and tests whether target rank alone changes the kNN-linear gap.

Second concrete diagnostic: `topk_categorical_bottleneck_pic_net` keeps the
original instance-ID CE objective but routes prediction through a sampled top-k
categorical bottleneck. This tests whether PIC-like instance classification
only develops the kNN-linear inversion when the latent bottleneck capacity
(`C`, `k`) becomes large.

### Phase 4 — Synthesis

Write up the causal story with evidence per hypothesis:
`.agents/designs/findings_instance_discrimination_limitations.md`. Expected shape
of the conclusion: instance-level CE is degenerate supervised learning at K=N;
it optimizes alignment (per-instance) and uniformity (over instances) but the
uniformity term acts *within classes*, maximizing within-class scatter; kNN is
robust to this, linear probes are not; methods that work either make targets
non-parametric (InstDisc), coarsen labels (DeepCluster/SwAV/this repo's OT trees),
or add fixes (PIC).

## Order and Decision Points

1. Phase 1.1 first (one notebook, hours): if normalized-linear closes the gap,
   H2 dominates and the framing changes before any training is spent.
2. Phase 0 in parallel (no compute).
3. Phase 2 → Phase 3 sequentially on Colab.
4. After Phase 3, decide whether to extend (e.g., STL-10 replication, projector
   ablation depth) or write up.

## Repo Conventions

- Each code change gets a `changes/CHANGESLOG_branch*.md` entry.
- All experiments run from Colab notebooks under `notebooks/`.
- Design docs and findings live in `.agents/designs/`.
- New tools get tests under `tests/` (torch-free where possible).
