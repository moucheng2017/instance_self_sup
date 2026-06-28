# Changes — Branch 63

## 1. `train.saving_frequency` — periodic checkpoint saving

**Files changed:** `configs/meta_exps/meta_random_config.yaml`, `main.py`,
`notebooks/random-meta-cifar10-ssl.ipynb`

A new `train.saving_frequency` parameter controls how often checkpoints are
written within each episode. When set to a positive integer N, a checkpoint is
saved after every N epochs. The end-of-episode checkpoint is skipped when the
final epoch was already captured by the frequency trigger, avoiding duplicate
files.

- `configs/meta_exps/meta_random_config.yaml`: added `saving_frequency: null`
  under `train` (null = save only at episode end, preserving the original
  behaviour). Also added an `eval` section required by `linear_eval.py`.
- `main.py`:
  - `save_checkpoint()`: checkpoint filename now includes the epoch number
    (`<name>_epoch<N>_<timestamp>.pth`). `checkpoint_path.txt` is opened in
    append mode (`"a"`) so all checkpoints — including intermediate ones — are
    recorded, one path per line.
  - `train_model()`: intermediate saves triggered inside the epoch loop when
    `saving_frequency` is set. Final save is skipped if the last epoch was
    already saved.
  - `meta_train_model()`: same logic applied to the per-episode epoch loop.
- `notebooks/random-meta-cifar10-ssl.ipynb` (cell 5): added `SAVING_FREQUENCY`
  variable (default `10`) and wired it into `overrides['train']['saving_frequency']`.

## 2. Local linear evaluation

**Files changed:** `configs/linear_eval.yaml` *(new)*, `local_scripts/run_linear_eval_local.sh` *(new)*, `linear_eval.py`

### `configs/linear_eval.yaml`

Dedicated config for `linear_eval.py` with `dataset`, `model`, and `eval`
sections. Also contains a `local:` block that stores runtime paths read by the
evaluation script:

- `eval_from`: path to the `.pth` checkpoint to evaluate.
- `data_dir`: parent directory of `cifar-10-batches-py`.
- `save_dir`: directory where the timestamped run folder is created.
- `device`: compute device (default `mps`).

### `local_scripts/run_linear_eval_local.sh`

Bash script to run `linear_eval.py` on a local MacBook Air using the
`simsiam-mps` conda env. Workflow:

1. Reads path defaults from the `local:` block of `configs/linear_eval.yaml`.
2. Any value is overridden by the corresponding env var (`EVAL_FROM`,
   `DATA_DIR`, `SAVE_DIR`, `DEVICE`).
3. `ckpt_dir` (required by `build_args` but unused by linear eval) is
   auto-generated as `/tmp/linear_eval_ckpt_<checkpoint_basename>` so the
   user never has to set it.

Usage:
```bash
# Preferred: set paths in configs/linear_eval.yaml, then just:
bash local_scripts/run_linear_eval_local.sh

# Or override at call time:
EVAL_FROM=/path/to/checkpoint.pth bash local_scripts/run_linear_eval_local.sh
```

### `linear_eval.py` bug fix

`args.l2_norm_backbone_features` was referenced directly but the config nests
this flag under `model.l2_norm_backbone_features`, causing an `AttributeError`
at runtime. Fixed to `getattr(args.model, "l2_norm_backbone_features", False)`.

## 3. Documentation updates

- `README.md`: updated Local Linear Evaluation section to describe the
  config-first workflow and new `local:` block; added `configs/linear_eval.yaml`
  and `local_scripts/` to the Code Map.
- `.agent/code_structure.md`: added entries for `linear_eval.py`,
  `configs/linear_eval.yaml`, and `local_scripts/run_linear_eval_local.sh` in
  the Module Map; removed the misplaced `local_scripts` entry from Known Gaps.
