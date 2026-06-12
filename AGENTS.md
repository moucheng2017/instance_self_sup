# Agent Notes

This repository is a PyTorch self-supervised learning project centered on exploring so-called pseudo-supervised learning, with baselines including SimSiam, BYOL, SimCLR. All the experiments are launched via Google Colab notebooks in `notebooks/`, with user-defined arguments in the overrides, on top of the default yaml configs under `configs/`.

## Start Here

Before broad code exploration, read:

1. `docs/ai/repo-map.md`
2. `README.md`
3. The config file relevant to the task under `configs/`

Use `rg` for targeted searches. Avoid remapping the whole repository unless `docs/ai/repo-map.md` is stale or the task crosses multiple subsystems.

## Common Commands

Install dependencies for Google Colab runs:

```bash
pip install -r requirements_colab.txt
```

Quick syntax check:

```bash
python -m compileall .
```

## Project Conventions

- Runtime behavior is config-driven through YAML files loaded by `arguments.py`.
- `main.py` owns SSL training and optional post-training linear evaluation.
- `linear_eval.py` trains a linear classifier on a frozen backbone loaded from a checkpoint.
- Models are registered in `models/__init__.py`.
- Training augmentations are registered in `augmentations/__init__.py`.
- Datasets are registered in `datasets/__init__.py`.
- Optimizers and schedulers are under `optimizers/`.
- Logging, plotting, accuracy, and kNN monitoring helpers are under `tools/`.

## Practical Cautions

- Some documentation appears newer than this checkout and references files that are not currently present. Check `rg --files` before relying on a mentioned path.
- There are no source test files in `tests/` in this checkout, only cached bytecode. Prefer focused smoke checks or `python -m compileall .` unless adding tests.
- Long training commands may need real datasets and GPU access. Use `--debug` or small configs when validating code paths locally.
- Do not delete or overwrite user checkpoints, logs, datasets, or notebook outputs.

