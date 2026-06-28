# Changes

## 2026-04-25

This repository was updated for Jupyter Notebook use and extended with a new augmentation-binary self-supervised experiment.

### Added

- `jupyter_utils.py`
  - Notebook-friendly training and evaluation helpers.

- `colab_utils.py`
  - Google Colab-specific setup and training helpers.

- `Google_Colab_SSL_exps.ipynb`
  - Starter notebook for running the project in Google Colab.

- `configs/simsiam_cifar_jupyter.yaml`
  - SimSiam config with notebook-safe defaults.

- `configs/aug_binary_cifar_jupyter.yaml`
  - Config for the new augmentation-vs-clean binary objective.

- `configs/simsiam_cifar_colab.yaml`
  - SimSiam config tuned for Google Colab.

- `configs/aug_binary_cifar_colab.yaml`
  - Colab config for the augmentation-vs-clean binary objective.

- `requirements_colab.txt`
  - Lightweight dependency list for Google Colab installs.

- `augmentations/augmentation_binary_aug.py`
  - Clean and augmented transforms for the binary pseudo-label task.

- `datasets/augmentation_binary.py`
  - Batch collator that creates pseudo labels:
  - `1` for augmented images
  - `0` for clean images

- `models/aug_binary.py`
  - Backbone plus binary classifier head trained with `BCEWithLogitsLoss`.

### Changed

- `arguments.py`
  - Refactored config loading so runs can be launched from CLI or notebook code.

- `main.py`
  - Refactored training into reusable functions.
  - Added support for the new binary pseudo-label training flow.
  - Preserved kNN monitoring during training.

- `linear_eval.py`
  - Made evaluation compatible with notebook-built args and different checkpoint layouts.

- `models/__init__.py`
  - Registered `aug_binary`.

- `augmentations/__init__.py`
  - Registered the new training mode.

- `datasets/__init__.py`
  - Exported the new collator for training.

- `tools/knn_monitor.py`
  - Fixed device handling and removed hard-coded CUDA-only behavior.

- `README.md`
  - Added notebook and Google Colab usage examples.

- `tools/logger.py`
  - Added TensorBoard writer fallback that works better across Colab environments.

### Result

- You can now train from Jupyter Notebook on a remote GPU.
- You can now run the project directly on Google Colab with a provided notebook and helper utilities.
- You can run your new idea that predicts whether an image was augmented or not.
- The learned backbone is still checked with the existing kNN monitor during training.

### Notes

- A fuller record is available in `CODEBASE_CHANGE_TRACKS_2026-04-25.md`.
- A pre-existing unrelated indentation issue remains in `models/swav.py`.
