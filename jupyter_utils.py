from arguments import build_args
from linear_eval import main as run_linear_eval
from main import train_model


def _merge_notebook_defaults(overrides):
    notebook_defaults = {
        "dataset": {"num_workers": 0},
        "logger": {"tensorboard": False, "matplotlib": True},
        "train": {"use_data_parallel": False},
    }
    if overrides is None:
        return notebook_defaults

    merged = {**notebook_defaults}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def build_notebook_args(
    config_file,
    data_dir,
    log_dir,
    ckpt_dir,
    overrides=None,
    device=None,
    debug=False,
    download=False,
    hide_progress=False,
):
    args = build_args(
        config_file=config_file,
        overrides=_merge_notebook_defaults(overrides),
        data_dir=data_dir,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        device=device,
        debug=debug,
        download=download,
        hide_progress=hide_progress,
        create_dirs=True,
    )
    if getattr(args, "eval", False) is not False:
        args.eval.use_data_parallel = False
    return args


def train_from_notebook(
    config_file,
    data_dir,
    log_dir,
    ckpt_dir,
    overrides=None,
    device=None,
    debug=False,
    download=False,
    hide_progress=False,
):
    args = build_notebook_args(
        config_file=config_file,
        data_dir=data_dir,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        overrides=overrides,
        device=device,
        debug=debug,
        download=download,
        hide_progress=hide_progress,
    )
    return train_model(args=args, device=args.device, finalize_logs=True)


def linear_eval_from_notebook(
    config_file,
    checkpoint_path,
    data_dir,
    log_dir,
    ckpt_dir,
    overrides=None,
    device=None,
    debug=False,
    download=False,
    hide_progress=False,
):
    args = build_notebook_args(
        config_file=config_file,
        data_dir=data_dir,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        overrides=overrides,
        device=device,
        debug=debug,
        download=download,
        hide_progress=hide_progress,
    )
    args.eval_from = checkpoint_path
    return run_linear_eval(args)
