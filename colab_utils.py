import os

from arguments import build_args
from linear_eval import main as run_linear_eval
from main import meta_train_model, train_model


def is_running_in_colab():
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def mount_google_drive(mount_point="/content/drive"):
    if not is_running_in_colab():
        raise EnvironmentError("mount_google_drive() is intended to be called from Google Colab.")

    from google.colab import drive

    drive.mount(mount_point)
    return mount_point


def default_colab_paths(project_name="SSL_exps", use_drive=True, drive_mount_point="/content/drive"):
    if use_drive:
        base_dir = os.path.join(drive_mount_point, "MyDrive", project_name)
    else:
        base_dir = os.path.join("/content", project_name)

    return {
        "base_dir": base_dir,
        "data_dir": os.path.join(base_dir, "data"),
        "log_dir": os.path.join(base_dir, "logs"),
        "ckpt_dir": os.path.join(base_dir, "checkpoints"),
    }


def ensure_colab_dirs(paths):
    for key in ("base_dir", "data_dir", "log_dir", "ckpt_dir"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def _merge_colab_defaults(overrides):
    colab_defaults = {
        # Notebook runtimes are more fragile with multiprocessing, especially
        # when multiple loaders stay alive across epochs for kNN evaluation.
        "dataset": {"num_workers": 0},
        "logger": {"tensorboard": False, "matplotlib": True},
        "train": {"use_data_parallel": False},
    }
    if overrides is None:
        return colab_defaults

    merged = {**colab_defaults}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def build_colab_args(
    config_file,
    project_name="SSL_exps",
    use_drive=True,
    drive_mount_point="/content/drive",
    overrides=None,
    device="cuda",
    debug=False,
    download=True,
    hide_progress=False,
):
    paths = ensure_colab_dirs(
        default_colab_paths(
            project_name=project_name,
            use_drive=use_drive,
            drive_mount_point=drive_mount_point,
        )
    )
    args = build_args(
        config_file=config_file,
        overrides=_merge_colab_defaults(overrides),
        data_dir=paths["data_dir"],
        log_dir=paths["log_dir"],
        ckpt_dir=paths["ckpt_dir"],
        device=device,
        debug=debug,
        download=download,
        hide_progress=hide_progress,
        create_dirs=True,
    )
    if getattr(args, "eval", False) is not False:
        args.eval.use_data_parallel = False
    return args, paths


def train_from_colab(
    config_file,
    project_name="SSL_exps",
    use_drive=True,
    drive_mount_point="/content/drive",
    overrides=None,
    device="cuda",
    debug=False,
    download=True,
    hide_progress=False,
):
    args, paths = build_colab_args(
        config_file=config_file,
        project_name=project_name,
        use_drive=use_drive,
        drive_mount_point=drive_mount_point,
        overrides=overrides,
        device=device,
        debug=debug,
        download=download,
        hide_progress=hide_progress,
    )
    result = train_model(args=args, device=args.device, finalize_logs=True)
    result["paths"] = paths
    return result


def linear_eval_from_colab(
    config_file,
    checkpoint_path,
    project_name="SSL_exps",
    use_drive=True,
    drive_mount_point="/content/drive",
    overrides=None,
    device="cuda",
    debug=False,
    download=True,
    hide_progress=False,
):
    args, paths = build_colab_args(
        config_file=config_file,
        project_name=project_name,
        use_drive=use_drive,
        drive_mount_point=drive_mount_point,
        overrides=overrides,
        device=device,
        debug=debug,
        download=download,
        hide_progress=hide_progress,
    )
    args.eval_from = checkpoint_path
    accuracy = run_linear_eval(args)
    return {"accuracy": accuracy, "paths": paths}


def meta_train_from_colab(
    config_file,
    project_name="SSL_exps",
    use_drive=True,
    drive_mount_point="/content/drive",
    overrides=None,
    device="cuda",
    debug=False,
    download=True,
    hide_progress=False,
    resume_from_episode=0,
    resume_checkpoint=None,
    resume_mother_log_dir=None,
    resume_mother_ckpt_dir=None,
):
    """Run the meta pseudo-supervised training loop from a Colab notebook.

    The config must contain a ``meta`` section with at least ``episode_size``.
    See ``configs/meta_pseudo_supervised_cifar_colab.yaml`` for an example.

    To resume an interrupted run pass:
        - ``resume_from_episode``: 0-based index of the episode to restart from.
        - ``resume_checkpoint``: full path to the checkpoint saved at the end of
          episode ``resume_from_episode - 1``.
        - ``resume_mother_log_dir`` / ``resume_mother_ckpt_dir``: the mother
          directory paths printed when the original run started.
    """
    args, paths = build_colab_args(
        config_file=config_file,
        project_name=project_name,
        use_drive=use_drive,
        drive_mount_point=drive_mount_point,
        overrides=overrides,
        device=device,
        debug=debug,
        download=download,
        hide_progress=hide_progress,
    )
    result = meta_train_model(
        args=args,
        device=args.device,
        resume_from_episode=resume_from_episode,
        resume_checkpoint=resume_checkpoint,
        resume_mother_log_dir=resume_mother_log_dir,
        resume_mother_ckpt_dir=resume_mother_ckpt_dir,
    )
    result["paths"] = paths
    return result
