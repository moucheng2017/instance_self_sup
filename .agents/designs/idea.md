# Research Idea: Spectral Geometry Transfer in Self-Supervised Learning

---

## 1. Hypothesis

Instance discrimination pretraining (pseudo_sup, reimplemented from Dosovitskiy et al. 2014) produces feature representations with a **flatter singular value spectrum** than standard contrastive methods (SimCLR, SimSiam) trained from random initialisation. We hypothesise that:

1. **Spectral flatness is causally responsible** for pseudo_sup's advantage in low-data regimes — not an incidental by-product of the training procedure.

2. **Pseudo_sup initialisation transfers this spectral geometry** into subsequent contrastive learning: a SimCLR model warm-started from pseudo_sup weights inherits and preserves a flat spectrum, whereas the same model trained from random initialisation converges to a spectrally concentrated attractor.

3. **This attractor is robust to longer training**: contrastive methods trained from random initialisation in low-data regimes cannot recover pseudo_sup's spectral flatness regardless of training duration. The initialisation geometry acts as a basin-of-attraction selector.

4. **Spectral flatness translates into practical gains in data-scarce domains**: medical imaging (dermoscopy, histopathology, radiology), satellite imagery, and industrial inspection — settings where large-batch contrastive methods structurally struggle.

### Intuition

In a KNN evaluation with N=1000 samples and D=512 feature dimensions, distance metric quality is critical. A spectrally concentrated feature matrix means a handful of principal directions dominate distance computation, biasing nearest-neighbour search toward those axes. A flat spectrum distributes variance evenly across dimensions, enabling more discriminative distance geometry. Pseudo_sup's surrogate-class objective — which does not require large batches or negative pairs — naturally encourages this uniform spreading without explicitly optimising for it.

### Framing for publication

> *"Contrastive learning in low-data regimes fails because random initialisation leads the encoder into a spectrally concentrated basin that longer training cannot escape. Pseudo_sup initialisation conditions the feature geometry before contrastive training begins; this spectral signature persists, compounds, and cannot be recovered by additional compute from random init."*

Target venue: **ICML main track** (mechanistic claim + negative result + application = publishable combination).

---

## 2. Baselines

### 2.1 Feature Decorrelation Methods

#### Motivation and reasoning

VICReg (Bardes et al. 2022) and Barlow Twins (Zbontar et al. 2021) explicitly penalise feature redundancy via a covariance regularisation term. In principle, they should self-correct toward a flat singular value spectrum without requiring any special initialisation. If they do — and if their spectral flatness matches pseudo_sup's at low N — then the paper's central mechanism claim requires reframing.

This is the **go/no-go experiment**: run it in week 1 of month 1, before investing in any application experiments. Two outcomes are both publishable:

- **VICReg/Barlow Twins do NOT match pseudo_sup in effective rank at N=1000** → the original claim holds. Pseudo_sup achieves spectral flatness without large batches; feature decorrelation methods need large batches to make their covariance regularisation effective, and fail in the small-N regime.
- **VICReg/Barlow Twins DO match pseudo_sup in effective rank** → reframe: "spectral flatness is the key variable for low-data SSL; pseudo_sup is a batch-size-efficient way to achieve it." The contrast between pseudo_sup (no large batch needed) and VICReg (degrades at small batch) is still a contribution.

#### Experiment to run

**Setup:** ResNet-18 backbone, CIFAR-10 / STL-10, N ∈ {200, 500, 1000, 5000, full}.

**Methods compared:**
- `pseudo_sup` (reimplemented Exemplar-CNN)
- `SimCLR` (random init)
- `SimSiam` (random init)
- `VICReg` (random init)
- `Barlow Twins` (random init)
- `SimCLR` (pseudo_sup init) — the proposed method

**Metrics at each N:**
- Effective rank of the feature matrix F (shape N × 512) from the ResNet-18 penultimate layer
- KNN accuracy (k=20, 80/20 train/test split within the N samples)
- Top-20 singular values (for SVD plot)

**Expected finding:** VICReg and Barlow Twins have flatter spectra than SimCLR/SimSiam but still less flat than pseudo_sup at N ≤ 1000. The gap between pseudo_sup and VICReg closes as N grows (crossover N is itself a finding — it defines the "low-data regime" boundary).

**Key comparison table:**

| Method | Effective rank (N=1000) | KNN acc (N=1000) | Batch size dependency |
|---|---|---|---|
| pseudo_sup | high (flat) | best | none |
| SimCLR + pseudo_sup init | high (preserved) | best | low |
| VICReg | medium | mid | large |
| Barlow Twins | medium | mid | large |
| SimCLR random init | low | worst | large |
| SimSiam random init | very low (collapse) | worst | medium |

---

### 2.2 Spectrum Surgery Experiments

#### Motivation and reasoning

Showing that pseudo_sup outperforms other methods and has a flatter spectrum is a correlation. To establish that spectral flatness **causes** the low-data KNN advantage — rather than being an incidental property of pseudo_sup's training — we need to manipulate the spectrum directly and measure the downstream effect.

Three complementary surgeries test the causal claim from different directions:

- **Experiment 1 (collapse)** tests the necessary condition: destroying pseudo_sup's flat spectrum should destroy its KNN advantage.
- **Experiment 2 (flatten)** tests the sufficient condition: manufacturing spectral flatness in SimCLR's features should recover KNN performance toward pseudo_sup levels.
- **Experiment 3 (matched rank)** provides the cleanest comparison: controlling for effective rank isolates whether pseudo_sup's advantage persists even when the spectral summary statistic is equalised (implying richer structure beyond rank).

Together, these three experiments make the strongest possible causal argument available without running a full theoretical proof.

#### Conceptual flow

```
Extract features (ResNet-18 penultimate layer, N=1000)
        ↙                    ↓                    ↘
[pseudo_sup init]     [Feature matrix F]     [Random init SimCLR]
        ↘                    ↓                    ↙
              Compute SVD: F = U Σ Vᵀ
              Effective rank = exp(H(σ/‖σ‖₁))
                     ↙         ↓         ↘
              [Exp 1]      [Exp 2]      [Exp 3]
           Collapse       Flatten       Matched
           pseudo_sup     SimCLR        rank
           spectrum       (ZCA)         checkpoints
              ↓              ↓              ↓
        KNN drops?     KNN rises?     pseudo_sup
        → flatness     → flatness     still wins?
          necessary      sufficient   → beyond rank
              ↘             ↓             ↙
                  Causal claim established:
            spectral flatness → low-data KNN gain
            pseudo_sup = one efficient path to it
```

#### Experiment 1: Collapse pseudo_sup's spectrum

**What:** Project pseudo_sup features onto only the top-K principal components, concentrating variance into fewer dimensions. Sweep K ∈ {8, 16, 32, 64, 128, 256, 512}.

**Expected result:** KNN accuracy degrades monotonically as K decreases (spectrum becomes more concentrated). This establishes that spectral flatness is a **necessary condition** for pseudo_sup's advantage — the advantage is not due to some other property of the features.

**Implementation sketch:**
```python
def collapse_spectrum(F, keep_dims=8):
    _, S, Vt = torch.linalg.svd(F, full_matrices=False)
    F_collapsed = F @ Vt[:keep_dims].T @ Vt[:keep_dims]
    return F_collapsed

# Sweep keep_dims, log effective_rank and knn_acc at each point
```

**Figure:** KNN accuracy vs. keep_dims (log scale), with a dashed horizontal line at the intact pseudo_sup baseline. A monotonic drop as dims decrease confirms the claim.

---

#### Experiment 2: Flatten SimCLR's spectrum (ZCA whitening)

**What:** Apply ZCA whitening to SimCLR features at varying strengths (0 = original, 1 = fully flat). ZCA rotates the feature space so every direction has equal variance, directly manufacturing the flatness that pseudo_sup achieves naturally.

**Expected result:** KNN accuracy rises as whitening strength increases, approaching — or reaching — pseudo_sup's performance at strength=1.0. This establishes spectral flatness as a **sufficient condition**. If it only partially closes the gap, there is additional structure in pseudo_sup features beyond the spectrum (e.g. better feature alignment), which is also a finding worth reporting.

**Implementation sketch:**
```python
def zca_whiten(F, eps=1e-5, strength=1.0):
    F = F - F.mean(0)
    U, S, Vt = torch.linalg.svd(F, full_matrices=False)
    S_inv = 1.0 / (S + eps)
    S_target = strength * S_inv + (1 - strength) * torch.ones_like(S_inv)
    F_white = U * S_target.unsqueeze(0)
    F_white = F_white * (S.mean() / F_white.std())
    return F_white

# Sweep strength in [0.0, 0.25, 0.5, 0.75, 1.0]
# Also run on medical datasets (PathMNIST) to confirm generality
```

**Figure:** KNN accuracy vs. whitening strength, showing pseudo_sup accuracy as a dashed line. The gap between SimCLR (strength=0) and pseudo_sup (dashed) should shrink as strength→1.

---

#### Experiment 3: Matched effective rank comparison

**What:** During training, log effective rank and KNN accuracy at every checkpoint (every 50 epochs) for both pseudo_sup-init SimCLR and random-init SimCLR. Find pairs of checkpoints across the two runs where effective rank is approximately equal (within tolerance of ±2.0). Compare KNN accuracy at those matched pairs.

**Expected result (two scenarios):**
- **pseudo_sup still wins at matched rank** → effective rank is not the whole story; pseudo_sup's initialisation encodes richer structure (e.g. better feature-class alignment, or specific singular vector directions) beyond what the scalar effective rank captures. This is a more nuanced and interesting finding.
- **KNN accuracy equalises at matched rank** → effective rank is a sufficient summary statistic. The advantage of pseudo_sup init reduces entirely to getting the model to a higher effective rank faster and keeping it there.

**Implementation sketch:**
```python
def build_trajectory(checkpoint_dir, dataset, device, log_every=50):
    trajectory = []
    for ckpt in sorted(glob.glob(f"{checkpoint_dir}/*.pt")):
        epoch = int(re.search(r'ep(\d+)', ckpt).group(1))
        if epoch % log_every != 0:
            continue
        model = get_resnet18_extractor(ckpt, device)
        F, labels = extract_features(model, dataset, n_samples=1000)
        _, eff_rank, _ = spectral_diagnostics(F)
        acc = knn_eval(F, labels, k=20)
        trajectory.append({'epoch': epoch, 'effective_rank': eff_rank,
                           'knn_acc': acc, 'checkpoint_path': ckpt})
    return trajectory

# Match checkpoints across trajectories where |eff_rank_A - eff_rank_B| < 2.0
```

**Figure:** Two curves (pseudo_sup init, random init) on axes of KNN accuracy vs. effective rank. If the curves diverge at the same effective rank, pseudo_sup is doing something more than just achieving flatness.

---

#### Shared diagnostics (all three experiments)

```python
def spectral_diagnostics(F):
    _, S, _ = torch.linalg.svd(F, full_matrices=False)
    S = S.float()
    p = (S / S.sum()).clamp(min=1e-10)
    effective_rank = torch.exp(-(p * torch.log(p)).sum()).item()
    explained = (S**2).cumsum(0) / (S**2).sum()
    return S.numpy(), effective_rank, explained.numpy()

def knn_eval(F, labels, k=20, n_train=800):
    F_norm = F.float()
    F_norm = F_norm / (F_norm.norm(dim=1, keepdim=True) + 1e-8)
    sim = F_norm[n_train:] @ F_norm[:n_train].T
    topk_labels = labels[:n_train][sim.topk(k, dim=1).indices]
    preds = torch.mode(topk_labels, dim=1).values
    return (preds == labels[n_train:]).float().mean().item()
```

**Combined figure (3-panel, main paper Figure 2):**
- Panel A: KNN acc vs. keep_dims — collapse experiment
- Panel B: KNN acc vs. whitening strength — ZCA experiment
- Panel C: KNN acc vs. effective rank — matched rank trajectories

---

## Notes on execution order

1. **Week 1:** Run VICReg / Barlow Twins baseline (Section 2.1). Gate decision: do they match pseudo_sup in effective rank at N=1000? Adjust framing accordingly.
2. **Weeks 2–3:** Run "longer training" ablation — train SimCLR from random init to 800 epochs at N=1000, logging effective rank every 50 epochs. This is the "cannot escape the attractor" evidence.
3. **Weeks 3–4:** Spectrum surgery experiments (Section 2.2), in order: Exp 1 → Exp 2 → Exp 3.
4. **Month 2:** Application experiments on PathMNIST, ISIC 2019, EuroSAT, DTD.
5. **Month 2 (parallel):** Begin writing introduction and related work.
6. **Month 3:** Full draft, revision, submission.

---

*Generated from research panel discussion — June 2026*