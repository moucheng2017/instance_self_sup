# Top-k Categorical Bottleneck PIC

Date: 2026-06-12

## Summary

- Added `topk_categorical_bottleneck_pic_net`, a PIC-style instance classifier
  with a sampled top-k categorical bottleneck between backbone features and
  instance-ID logits.
- Added focused tests in `tests/test_topk_categorical_bottleneck_pic.py` for
  the top-k sampler, column normalization, and model forward outputs.
- Registered the model in `models/__init__.py` and the pseudo-supervised
  source-pool loader path in `main.py`.
- Added `configs/topk_categorical_bottleneck_pic_cifar_colab.yaml`.
- Added `notebooks/topk-categorical-bottleneck-pic-cifar10-sweep.ipynb`,
  which trains each capacity setting for 100 epochs and then runs linear eval.
- Added design and workflow notes in `.agents/designs/` and
  `.agents/workflows/`.

## Research Rationale

This experiment is a diagnostic bridge between PIC and SwAV-like categorical
assignment. It keeps the instance-ID objective but forces prediction through a
capacity-controlled categorical bottleneck, allowing a sweep over latent
capacity `C` and active set size `k` to test whether high-capacity instance
classification drives the kNN-minus-linear-probe gap.
