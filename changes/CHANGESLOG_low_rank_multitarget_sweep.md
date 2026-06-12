# Low-Rank Multi-Target Instance Sweep

Date: 2026-06-12

## Summary

- Added `low_rank_multitarget_pseudo_supervised_net`, a diagnostic image-index
  supervision model that maps each instance to a fixed balanced set of latent
  labels.
- Added `configs/low_rank_multitarget_cifar_colab.yaml` for CIFAR-10 Colab
  runs.
- Added `notebooks/low-rank-multitarget-cifar10-sweep.ipynb`, which sweeps
  target rank `K`, trains each experiment for 100 epochs, then runs linear
  evaluation from each checkpoint.
- Registered the model in `models/__init__.py` and the pseudo-supervised
  source-pool loader path in `main.py`.

## Research Rationale

This tests whether reducing per-instance target independence reduces the
kNN-minus-linear-probe gap. `K=N,m=1` approximates ordinary instance
supervision; smaller `K` with `m=5` forces instances to share overclustered
latent targets.
