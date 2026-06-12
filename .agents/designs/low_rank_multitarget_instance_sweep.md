# Design: Low-Rank Multi-Target Instance Supervision Sweep

Status: implemented draft (2026-06-12)
Related: `.agents/designs/research_plan_instance_discrimination_limitations.md`

## Motivation

The current root-cause hypothesis is that independent image-index targets
fragment class geometry: kNN can remain high because local neighborhoods are
useful, while linear probing fails because each semantic class is scattered
across many unrelated instance neighborhoods.

This experiment directly varies target independence. Instead of assigning every
image one unique label, each image index receives a fixed code over `K` latent
labels:

```text
instance i -> membership set S_i, |S_i| = m, S_i subset {1, ..., K}
image x_i -> backbone -> K-way latent logits
loss = CE over the labels in S_i
```

`K` is the target rank / granularity. `m` is the minimum/shared membership size.
When `K=N` and `m=1`, this approximates ordinary instance supervision. When
`K<<N` and `m=5`, many instances must share latent targets.

## Constraint Choice

For the user's proposed "instance must belong to at least 5 classes" constraint,
the implemented default is `uniform_multi_ce`: the model is trained to put
probability mass on all `m` assigned latent labels. A weaker `set_ce` option is
also implemented; it only requires probability mass on at least one label in the
set.

The membership matrix is fixed and balanced, not learned. That keeps this as a
controlled diagnostic of target sharing. A later version can replace the fixed
matrix with learned/balanced assignments if this sweep shows that rank matters.

## Expected Diagnostic

If target independence is the root cause, the kNN-minus-linear gap should be
largest near `K=N, m=1` and shrink as `K` decreases or as `m` increases. If the
gap does not change, target rank alone is probably not enough; semantic target
quality or feature-driven clustering is required.

## Added Files

- `models/low_rank_multitarget_pseudo_supervised_net.py`
- `configs/low_rank_multitarget_cifar_colab.yaml`
- `notebooks/low-rank-multitarget-cifar10-sweep.ipynb`
