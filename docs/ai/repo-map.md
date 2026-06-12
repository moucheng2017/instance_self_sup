# Repo Map

Last updated: 2026-05-31

## Purpose

This repository implements self-supervised representation learning experiments in PyTorch. The main path trains an SSL model on image datasets such as CIFAR-10, CIFAR-100, STL-10, MNIST, ImageNet, or a synthetic random dataset, then optionally evaluates the learned backbone with a linear classifier.

Primary implemented methods include:

- SimSiam
- BYOL
- SimCLR
- SuperimposeNet
- PseudoSupervisedNet
- AugmentationBinaryClassifier / KeyQueryBinaryClassifier

## Read Order For New Tasks

For most tasks, read in this order:

1. `AGENTS.md`
2. The relevant YAML config under `configs/`
3. `arguments.py`
4. `main.py` for training changes, or `linear_eval.py` for evaluation changes
5. The relevant registry: `models/__init__.py`, `augmentations/__init__.py`, or `datasets/__init__.py`
6. The specific model, augmentation, dataset, optimizer, or tool file involved

## Entry Points

### `main.py`

Owns SSL training.

Important functions:

- `build_train_loader(args)`: builds the training loader, including special handling for `superimpose_net` and `pseudo_supervised_net`.
- `build_eval_loader(args, train)`: builds train/test evaluation loaders for kNN or linear evaluation.
- `prepare_batch(batch, device)` and `forward_batch(model, batch, device)`: normalize batch/device handling before model forward calls.
- `build_optimizer_and_scheduler(model, train_loader, args)`: creates optimizer and cosine/warmup scheduler.
- `train_model(args, device=None, finalize_logs=False)`: main reusable training loop.
- `save_checkpoint(model, epoch, args)`: writes checkpoint and `checkpoint_path.txt`.

The script uses `get_args()` from `arguments.py` when run from the CLI.

### `linear_eval.py`

Trains a linear classifier on a frozen backbone.

Important functions:

- `infer_num_classes(dataset)`: infers output classes from torchvision-style datasets and subsets.
- `load_backbone_weights(model, checkpoint_path)`: supports checkpoints that either include `backbone.` prefixes or direct backbone weights.
- `main(args)`: builds loaders, loads the backbone, trains the classifier, and reports final accuracy.

### `jupyter_utils.py` and `colab_utils.py`

Provide notebook-friendly wrappers around `arguments.build_args()`, `main.train_model()`, and `linear_eval.main()`.

- `jupyter_utils.py` expects explicit local paths.
- `colab_utils.py` can build default Drive or `/content` paths and optionally mount Google Drive.
- Both default to `num_workers: 0`, disable TensorBoard by default, and avoid `DataParallel` for notebook stability.

## Configuration

Configuration is YAML-driven. `arguments.py` loads config files, applies optional overrides, creates runtime directories, sets deterministic seeds, and derives:

- `args.aug_kwargs`
- `args.dataset_kwargs`
- `args.dataloader_kwargs`

Current config groups:

- `configs/python_runs/`: local Python runs for SimSiam, BYOL, and SimCLR.
- `configs/baselines_ssl/`: Colab-oriented baseline SSL configs.
- `configs/linear_evals/`: standalone linear evaluation configs.
- `configs/pseudo_sup/`: pseudo-supervised STL-10 config.

Common required CLI arguments:

```bash
python main.py \
  --data_dir ../Data \
  --log_dir ../logs \
  --ckpt_dir ~/.cache \
  -c configs/python_runs/simsiam_cifar.yaml \
  --hide_progress
```

Useful flags:

- `--debug`: forces tiny batch/epoch settings through `_apply_debug_overrides()`.
- `--debug_subset_size`: controls the dataset subset size used in debug mode.
- `--download`: allows torchvision datasets to download.
- `--eval_from`: checkpoint path for standalone linear evaluation.
- `--device`: defaults to CUDA when available.

Environment variable defaults:

- `DATA`
- `LOG`
- `CHECKPOINT`

## Module Layout

### Models

Registry: `models/__init__.py`

Key files:

- `models/simsiam.py`
- `models/byol.py`
- `models/simclr.py`
- `models/aug_binary.py`
- `models/superimpose_net.py`
- `models/pseudo_supervised_net.py`
- `models/swav.py`
- `models/backbones/cifar_resnet_1.py`
- `models/backbones/cifar_resnet_2.py`

`get_backbone(backbone, castrate=True)` constructs a backbone by name, records `output_dim`, and replaces `fc` with `torch.nn.Identity()` when castrating.

`get_model(model_cfg, num_classes=None)` dispatches by `model_cfg.name`.

### Augmentations

Registry: `augmentations/__init__.py`

Key files:

- `augmentations/simsiam_aug.py`
- `augmentations/byol_aug.py`
- `augmentations/simclr_aug.py`
- `augmentations/eval_aug.py`
- `augmentations/superimpose_aug.py`
- `augmentations/swav_aug.py`
- `augmentations/gaussian_blur.py`

Training augmentations are selected by model name. Evaluation uses `Transform_single(image_size, train=train_classifier)`.

### Datasets

Registry: `datasets/__init__.py`

Supported dataset names:

- `mnist`
- `stl10`
- `cifar10`
- `cifar100`
- `imagenet`
- `random`

Special dataset wrappers:

- `datasets/superimpose.py`
- `datasets/pseudo_supervised.py`
- `datasets/random_dataset.py`

`stl10` uses the labeled `train` split for train-time memory/evaluation compatibility.

### Optimizers

Registry: `optimizers/__init__.py`

Supported optimizer names:

- `sgd`
- `lars`
- `lars_simclr`
- `larc`

`get_optimizer()` builds separate parameter groups for base parameters and predictor parameters. `LR_Scheduler` lives in `optimizers/lr_scheduler.py`.

### Tools

Important helpers:

- `tools/logger.py`: TensorBoard or tensorboardX scalar logging plus optional SVG plotting.
- `tools/plotter.py`: saves `plotter.svg`.
- `tools/knn_monitor.py`: weighted kNN evaluation on learned features.
- `tools/accuracy.py`: accuracy helper.
- `tools/average_meter.py`: metric accumulator.
- `tools/file_exist_fn.py`: file existence helper.

## Data Flow

The normal training path is:

```text
YAML config
  -> arguments.build_args()
  -> main.train_model()
  -> build_train_loader() / build_eval_loader()
  -> get_dataset() + get_aug()
  -> get_model()
  -> get_optimizer() + LR_Scheduler
  -> training loop
  -> optional knn_monitor()
  -> save_checkpoint()
  -> optional linear_eval.main()
```

For linear evaluation:

```text
YAML config + checkpoint path
  -> arguments.build_args()
  -> linear_eval.main()
  -> get_dataset() + eval transforms
  -> get_backbone()
  -> load_backbone_weights()
  -> train classifier
  -> test accuracy
```

## Outputs

Training creates a timestamped log directory under `--log_dir`:

```text
in-progress_<timestamp>_<run-name>/
```

Typical outputs:

- copied or resolved config YAML
- TensorBoard event files when enabled
- `plotter.svg` when matplotlib plotting is enabled
- `checkpoint_path.txt`
- `.pth` checkpoints under `--ckpt_dir`

`finalize_log_dir()` can rename in-progress logs to `completed_...` or `debug_...`.

## Known Caveats

- `README.md` and `CHANGES.md` mention some files that are not present in this checkout, such as `configs/aug_binary_cifar_jupyter.yaml`, `configs/aug_binary_cifar_colab.yaml`, `datasets/augmentation_binary.py`, and `augmentations/augmentation_binary_aug.py`. Check the actual tree with `rg --files` before assuming a documented file exists.
- `tests/` currently has no source test files in this checkout.
- `models/swav.py` may have a pre-existing issue according to `CHANGES.md`; inspect before relying on SwAV.
- Long configs are designed for real training and can be slow. Prefer `--debug`, smaller configs, or targeted unit-style smoke checks during development.
- Many data paths assume torchvision dataset layouts under `--data_dir`.

## Suggested Research Workflow For Agents

1. Read this file and the relevant config.
2. State the likely files involved before opening many files.
3. Use `rg` to confirm symbols and call sites.
4. Read source around the exact functions to be changed.
5. Make scoped edits.
6. Run the smallest meaningful validation command.
7. If durable architecture knowledge changed, update this map.

