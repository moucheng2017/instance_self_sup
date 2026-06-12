import argparse
import copy
import os
import random
import re
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            if not (isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)):
                raise ValueError(f"Config key {key!r} is not a valid Python identifier.")
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!"
        )


def set_deterministic(seed):
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_debug_overrides(args):
    if args.debug:
        if getattr(args, "train", None):
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
            # Cap source_pool_size so PseudoSupervisedDataset does not exceed
            # the debug-subset dataset length (debug_subset_size elements).
            if hasattr(args.train, "source_pool_size"):
                args.train.source_pool_size = min(
                    int(args.train.source_pool_size),
                    args.debug_subset_size,
                )
            # Cap samples_per_epoch so an epoch is a handful of batches, not
            # thousands (default 20480 / batch_size=2 = 10240 batches).
            if hasattr(args.train, "samples_per_epoch"):
                args.train.samples_per_epoch = args.train.batch_size * 2
        if getattr(args, "eval", None):
            args.eval.batch_size = 2
            args.eval.num_epochs = 1
        args.dataset.num_workers = 0


def _prepare_runtime_args(args, config_file, merged_config, create_dirs=True):
    _apply_debug_overrides(args)

    if None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]:
        raise ValueError(
            "log_dir, data_dir, ckpt_dir, and name must all be set. "
            "Pass them explicitly or set the DATA, LOG, and CHECKPOINT environment variables."
        )

    run_name = "in-progress_" + datetime.now().strftime("%m%d%H%M%S_") + args.name
    args.log_dir = os.path.join(args.log_dir, run_name)

    if create_dirs:
        os.makedirs(args.log_dir, exist_ok=False)
        print(f"creating file {args.log_dir}")
        os.makedirs(args.ckpt_dir, exist_ok=True)
        # Save the fully-merged config (base YAML + all overrides) so the log
        # directory is self-contained and exactly reproduces this run.
        config_save_path = os.path.join(args.log_dir, os.path.basename(config_file))
        with open(config_save_path, "w") as f:
            yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

    set_deterministic(getattr(args, "seed", None))

    vars(args)["aug_kwargs"] = {
        "name": args.model.name,
        "image_size": args.dataset.image_size,
    }
    vars(args)["dataset_kwargs"] = {
        "dataset": args.dataset.name,
        "data_dir": args.data_dir,
        "download": args.download,
        "debug_subset_size": args.debug_subset_size if args.debug else None,
    }
    vars(args)["dataloader_kwargs"] = {
        "drop_last": True,
        "pin_memory": str(args.device).startswith("cuda"),
        "num_workers": args.dataset.num_workers,
    }

    if args.dataset.num_workers > 0:
        args.dataloader_kwargs["persistent_workers"] = True

    return args


def build_args(
    config_file,
    overrides=None,
    debug=False,
    debug_subset_size=8,
    download=False,
    data_dir=None,
    log_dir=None,
    ckpt_dir=None,
    device=None,
    eval_from=None,
    checkpoint_resume=None,
    hide_progress=False,
    create_dirs=True,
):
    config = copy.deepcopy(load_config(config_file))
    if overrides:
        _deep_update(config, copy.deepcopy(overrides))

    args = argparse.Namespace(
        config_file=config_file,
        debug=debug,
        debug_subset_size=debug_subset_size,
        download=download,
        data_dir=data_dir if data_dir is not None else os.getenv("DATA"),
        log_dir=log_dir if log_dir is not None else os.getenv("LOG"),
        ckpt_dir=ckpt_dir if ckpt_dir is not None else os.getenv("CHECKPOINT"),
        device=device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
        eval_from=eval_from,
        checkpoint_resume=checkpoint_resume,
        hide_progress=hide_progress,
    )

    for key, value in Namespace(config).__dict__.items():
        vars(args)[key] = value

    return _prepare_runtime_args(args, config_file=config_file, merged_config=config, create_dirs=create_dirs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", required=True, type=str, help="xxx.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_subset_size", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="if can't find dataset, download from web")
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA"))
    parser.add_argument("--log_dir", type=str, default=os.getenv("LOG"))
    parser.add_argument("--ckpt_dir", type=str, default=os.getenv("CHECKPOINT"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_from", type=str, default=None)
    parser.add_argument("--checkpoint_resume", type=str, default=None)
    parser.add_argument("--hide_progress", action="store_true")
    parsed = parser.parse_args()

    return build_args(
        config_file=parsed.config_file,
        debug=parsed.debug,
        debug_subset_size=parsed.debug_subset_size,
        download=parsed.download,
        data_dir=parsed.data_dir,
        log_dir=parsed.log_dir,
        ckpt_dir=parsed.ckpt_dir,
        device=parsed.device,
        eval_from=parsed.eval_from,
        checkpoint_resume=parsed.checkpoint_resume,
        hide_progress=parsed.hide_progress,
        create_dirs=True,
    )
