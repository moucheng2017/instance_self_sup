# Hierarchical Balanced vMF Self-Labeling Experiments

This repository began as a PyTorch SimSiam implementation, but the active work is now CIFAR/STL self-labeling on normalized embeddings:

```text
image -> backbone -> optional projector -> L2-normalized embedding
      -> batch-local or full-pool OT self-labels
      -> hierarchical/flat pseudo-label prediction loss
      -> optional sigmoid image-index regularization
```

The main experiment is `hierarchical_balanced_vmf_self_labeling_net` with `configs/hierarchical_balanced_vmf_cifar_colab.yaml`.

## Setup

```bash
conda create -n ssl_local python=3.11
conda activate ssl_local
pip install -r requirements.txt
```

Set paths explicitly on the command line, or export them once:

```bash
export DATA="/path/to/datasets"
export LOG="/path/to/logs"
export CHECKPOINT="/path/to/checkpoints"
```

## Main Training Run

```bash
python main.py \
  --data_dir "$DATA" \
  --log_dir "$LOG" \
  --ckpt_dir "$CHECKPOINT" \
  --download \
  --hide_progress \
  -c configs/hierarchical_balanced_vmf_cifar_colab.yaml
```

The default config trains on CIFAR-10 with:

- `batch_self_labeling: True`
- hierarchical vMF OT assignment
- prototype EMA
- sigmoid image-index regularization
- kNN monitoring enabled

Use `--debug` for a tiny smoke run:

```bash
python main.py \
  --data_dir "$DATA" \
  --log_dir "$LOG" \
  --ckpt_dir "$CHECKPOINT" \
  --download \
  --debug \
  --hide_progress \
  -c configs/hierarchical_balanced_vmf_cifar_colab.yaml
```

## Notebooks

Current Colab-oriented notebooks live in `notebooks/`:

- `notebooks/hierarchical-balanced-vmf-cifar10-ssl.ipynb`
- `notebooks/hierarchical-balanced-vmf-cifar10-ssl-no-sigmoid.ipynb`
- `notebooks/sigmoid-pseudo-supervised-cifar10-ssl.ipynb`

Notebook helpers are available in `colab_utils.py` and `jupyter_utils.py`.

## Other Remaining Configs

- `configs/sigmoid_pseudo_supervised_cifar_colab.yaml`: pairwise sigmoid image-index baseline.
- `configs/simsiam_cifar.yaml`, `configs/simsiam_cifar_colab.yaml`, `configs/simsiam_cifar_jupyter.yaml`: legacy SimSiam baselines.
- `configs/simclr_cifar.yaml`: legacy SimCLR baseline.
- `configs/byol_cifar.yaml`: legacy placeholder; BYOL is not wired for training yet.

## Linear Evaluation

For configs that include an `eval` section, run linear evaluation from a checkpoint with the same config family. For example:

```bash
python linear_eval.py \
  --data_dir "$DATA" \
  --log_dir "$LOG" \
  --ckpt_dir "$CHECKPOINT" \
  --hide_progress \
  --eval_from /path/to/checkpoint.pth \
  -c configs/simsiam_cifar.yaml
```

## Tests

```bash
PYTHONPATH="$PWD" conda run -n ssl_local pytest -q
```

The focused test suite currently covers the hierarchical balanced vMF self-labeling model and its OT helpers.

## Orientation Notes

Agent-facing design notes are in `.agents/`, especially:

- `.agents/codebase_structure.md`
- `.agents/hierarchical_balanced_vmf_self_labeling.txt`
- `.agents/OT_design_justifications.md`
