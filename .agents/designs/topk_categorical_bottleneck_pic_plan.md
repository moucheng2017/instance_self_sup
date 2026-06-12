# Plan: Soft Categorical Bottleneck PIC

Status: implemented draft (2026-06-12)
Related: `.agents/designs/research_plan_instance_discrimination_limitations.md`

## Motivation

The working hypothesis is that full-capacity instance supervision fragments
semantic class geometry: instance IDs are locally recoverable, so kNN can be
high, but the representation is not linearly separable by true CIFAR classes.

PIC is a full parametric instance classifier:

```text
image -> backbone -> feature z [B,D]
z -> learned instance matrix [D,N]
instance logits [B,N]
CE(instance_id)
```

This experiment inserts a deterministic soft categorical bottleneck before instance
classification:

```text
image -> backbone -> feature z [B,D]
z -> latent assigner [D,C]
q(c|x) [B,C]
q -> decoder / classifier [C,N]
instance logits [B,N]
CE(instance_id)
```

It should be framed as a diagnostic bridge between PIC and SwAV-like prototype
assignment, not as a standalone claim of a novel SSL algorithm.

## Core Questions

1. Does constraining instance prediction through `C` latent categories reduce
   the kNN-minus-linear-probe gap?
2. Does the gap reappear as bottleneck capacity grows (`C`)?
3. Are latent categories used broadly, or does the model collapse to a small
   subset?

## Model

Implemented in `models/topk_categorical_bottleneck_pic_net.py`.

Flow:

```text
images [B,3,32,32]
  -> backbone
features [B,D]
  -> latent_assigner_matrix [D,C]
latent_logits [B,C]
  -> softmax
assignments [B,C], rows sum to 1
  -> optional one-shot column normalization over batch columns
decoder_input [B,C]
  -> linear or shallow MLP decoder
instance_logits [B,N]
  -> CE against image-index pseudo label
```

## Loss

```text
L = CE(instance_logits, instance_id)
  + lambda_balance * KL(mean q(c|x) || Uniform(C))
  + lambda_entropy * (H(q(c|x)) - target_entropy)^2
```

Defaults:

- `lambda_balance = 1.0`
- `lambda_entropy = 0.0`
- `target_entropy = null` by default; set explicitly when entropy control is used
- `column_normalize = True`
- linear decoder (`decoder_hidden_dim: null`)

The column normalization is one-shot over `[B,C]` and not iterative
Sinkhorn-Knopp normalization. It is included as an ablation knob, not as the
only anti-collapse mechanism.

## Test-Driven Implementation

Tests are in `tests/test_topk_categorical_bottleneck_pic.py`.

They specify:

- Soft categorical assignment rows are finite probabilities that sum to 1.
- Column normalization preserves shape/total mass and handles empty columns.
- The model forward pass returns scalar total/instance/balance/entropy losses
  and latent-usage diagnostics.

Run in an environment with PyTorch:

```bash
PYTHONPATH="$PWD" pytest -q tests/test_topk_categorical_bottleneck_pic.py
```

## Experiment

Config:

- `configs/topk_categorical_bottleneck_pic_cifar_colab.yaml`

Notebook:

- `notebooks/topk-categorical-bottleneck-pic-cifar10-sweep.ipynb`

Initial 100-epoch sweep:

```text
C=100
C=1000
C=5000
C=10000
```

For each run:

1. train for 100 epochs,
2. record kNN monitor accuracy,
3. run linear evaluation from the checkpoint,
4. report `kNN - linear` gap.

## Interpretation

If the kNN-linear gap grows with bottleneck capacity, that supports the
hypothesis that high-capacity instance prediction produces locally useful but
linearly fragmented features. If the gap remains unchanged, the root cause may
not be capacity alone; semantic target quality or a different geometry
diagnostic is needed.
