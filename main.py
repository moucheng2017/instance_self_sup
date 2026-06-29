import copy
import math
import os
from datetime import datetime

import torch
from tqdm import tqdm

from arguments import get_args, save_effective_config
from augmentations import get_aug
from datasets import get_dataset
from datasets.pseudo_supervised import PseudoSupervisedDataset
from datasets.subset import maybe_select_subset, save_subset_indices, select_subset_indices
from linear_eval import main as linear_eval
from models.checkpoint_init import load_init_weights
from models import get_model
from optimizers import LR_Scheduler, get_optimizer
from tools import Logger, knn_monitor


def maybe_data_parallel(model, args):
    use_data_parallel = getattr(args.train, "use_data_parallel", False)
    if str(args.device).startswith("cuda") and torch.cuda.device_count() > 1 and use_data_parallel:
        return torch.nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_classifier_head(model):
    raw = unwrap_model(model)
    classifier = getattr(raw, "classifier", None)
    if classifier is not None:
        return classifier
    return getattr(raw, "pred", None)


def reset_classifier_head(model):
    head = get_classifier_head(model)
    if head is None:
        raise ValueError("The model does not expose a classifier head to reset.")
    for module in head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def _get_num_workers(args):
    return args.dataloader_kwargs.get("num_workers", 0)


def _subset_cfg(args):
    subset_n = getattr(args.train, "subset_n", None)
    subset_seed = getattr(args.train, "subset_seed", 42)
    return subset_n, subset_seed


def _save_selected_indices(args, indices):
    path = save_subset_indices(indices, getattr(args, "log_dir", None))
    vars(args)["selected_subset_indices"] = indices
    vars(args)["selected_subset_indices_path"] = path
    if path:
        print(f"Saved subset indices to {path}")
    return path


def _selected_indices_path_after_finalize(path, completed_log_dir):
    if path is None:
        return None
    return os.path.join(completed_log_dir, os.path.basename(path))


def _check_nonempty_train_loader(dataset, args):
    if args.dataloader_kwargs.get("drop_last", False) and args.train.batch_size > len(dataset):
        raise ValueError(
            "Training loader would be empty because batch_size is greater than the "
            f"dataset length ({args.train.batch_size} > {len(dataset)}) with drop_last=True."
        )


def build_train_loader(args):
    subset_n, subset_seed = _subset_cfg(args)

    if args.model.name == "pseudo_supervised_net":
        base_dataset = get_dataset(
            transform=None,
            train=True,
            **args.dataset_kwargs,
        )
        selected_indices = select_subset_indices(len(base_dataset), subset_n, subset_seed)
        dataset = PseudoSupervisedDataset(
            dataset=base_dataset,
            image_size=args.dataset.image_size,
            source_pool_size=None if selected_indices is not None else getattr(args.train, "source_pool_size", None),
            augment_probability=getattr(args.train, "augment_probability", 1.0),
            subset_seed=getattr(args.train, "source_subset_seed", 0),
            samples_per_epoch=getattr(args.train, "samples_per_epoch", None),
            batch_size=args.train.batch_size,
            negatives_ratio=getattr(args.train, "negatives_ratio", None),
            explicit_indices=selected_indices,
        )
        _save_selected_indices(args, selected_indices)
        _check_nonempty_train_loader(dataset, args)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=args.train.batch_size,
            drop_last=args.dataloader_kwargs["drop_last"],
            pin_memory=args.dataloader_kwargs["pin_memory"],
            num_workers=_get_num_workers(args),
            persistent_workers=args.dataloader_kwargs.get("persistent_workers", False),
        )

    base_dataset = get_dataset(
        transform=get_aug(train=True, **args.aug_kwargs),
        train=True,
        **args.dataset_kwargs,
    )
    dataset, selected_indices = maybe_select_subset(base_dataset, subset_n, subset_seed)
    _save_selected_indices(args, selected_indices)

    # Optional per-epoch sample budget. When set, oversample the (small) pool with
    # replacement so the loader yields exactly `samples_per_epoch` items per epoch. This
    # lets the two-view methods match the pseudo_supervised path's iteration count, so
    # every method runs the same number of iterations per epoch at a given pool size.
    # Default (None) preserves the original behaviour: one pass over the N-sample pool.
    samples_per_epoch = getattr(args.train, "samples_per_epoch", None)
    if samples_per_epoch is not None:
        samples_per_epoch = int(samples_per_epoch)
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive.")
        if args.dataloader_kwargs.get("drop_last", False) and samples_per_epoch < args.train.batch_size:
            raise ValueError(
                "Training loader would be empty because samples_per_epoch is smaller than "
                f"batch_size ({samples_per_epoch} < {args.train.batch_size}) with drop_last=True."
            )
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=samples_per_epoch
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=args.train.batch_size,
            **args.dataloader_kwargs,
        )

    _check_nonempty_train_loader(dataset, args)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs,
    )


def build_eval_loader(args, train):
    dataloader_kwargs = dict(args.dataloader_kwargs)
    dataloader_kwargs["drop_last"] = False
    return torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=train,
            **args.dataset_kwargs,
        ),
        shuffle=False,
        batch_size=args.train.batch_size,
        **dataloader_kwargs,
    )


def prepare_batch(batch, device):
    inputs, targets = batch
    if isinstance(inputs, (tuple, list)):
        return [item.to(device, non_blocking=True) for item in inputs], targets
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def forward_batch(model, batch, device):
    inputs, targets = prepare_batch(batch, device)
    if isinstance(inputs, list):
        return model.forward(*inputs)
    return model.forward(inputs, targets)


def build_optimizer_and_scheduler(model, train_loader, args):
    optimizer = get_optimizer(
        args.train.optimizer.name,
        model,
        lr=args.train.base_lr * args.train.batch_size / 256,
        momentum=getattr(args.train.optimizer, "momentum", 0.9),
        weight_decay=args.train.optimizer.weight_decay,
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs,
        args.train.warmup_lr * args.train.batch_size / 256,
        args.train.num_epochs,
        args.train.base_lr * args.train.batch_size / 256,
        args.train.final_lr * args.train.batch_size / 256,
        len(train_loader),
        constant_predictor_lr=True,
    )
    return optimizer, lr_scheduler


def _wandb_config(args):
    """Flatten the primitive config values into a dict for W&B run config."""
    config = {"name": getattr(args, "name", None)}
    for section in ("model", "train", "dataset"):
        namespace = getattr(args, section, None)
        if namespace is None:
            continue
        config[section] = {
            key: value
            for key, value in vars(namespace).items()
            if isinstance(value, (int, float, str, bool, type(None)))
        }
    return config


def save_checkpoint(model, epoch, args):
    model_path = os.path.join(
        args.ckpt_dir,
        f"{args.name}_epoch{epoch + 1}_{datetime.now().strftime('%m%d%H%M%S')}.pth",
    )
    torch.save(
        {
            "epoch": epoch + 1,
            "state_dict": unwrap_model(model).state_dict(),
        },
        model_path,
    )
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, "checkpoint_path.txt"), "a") as f:
        f.write(f"{model_path}\n")
    return model_path


def finalize_log_dir(args):
    completed_log_dir = args.log_dir.replace("in-progress", "debug" if args.debug else "completed")
    os.rename(args.log_dir, completed_log_dir)
    print(f"Log file has been saved to {completed_log_dir}")
    return completed_log_dir


def train_model(args, device=None, finalize_logs=False):
    device = device or args.device

    train_loader = build_train_loader(args)
    memory_loader = build_eval_loader(args, train=True)
    test_loader = build_eval_loader(args, train=False)

    raw_model = get_model(args.model, num_classes=getattr(train_loader.dataset, "num_pseudo_classes", None))
    init_checkpoint = getattr(args.model, "init_checkpoint", None)
    if init_checkpoint:
        load_init_weights(
            raw_model,
            init_checkpoint,
            load_backbone=getattr(args.model, "init_load_backbone", True),
            load_projector=getattr(args.model, "init_load_projector", False),
            load_predictor=getattr(args.model, "init_load_predictor", False),
        )
    model = maybe_data_parallel(raw_model.to(device), args)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(model, train_loader, args)
    logger = Logger(
        tensorboard=args.logger.tensorboard,
        matplotlib=args.logger.matplotlib,
        log_dir=args.log_dir,
        wandb=getattr(args.logger, "wandb", False),
        wandb_project=getattr(args.logger, "wandb_project", None),
        wandb_entity=getattr(args.logger, "wandb_entity", None),
        wandb_run_name=getattr(args, "name", None),
        wandb_config=_wandb_config(args),
    )

    accuracy = 0.0
    last_intermediate_ckpt = None
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc="Training")
    for epoch in global_progress:
        model.train()
        local_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.train.num_epochs}",
            disable=args.hide_progress,
        )
        for batch in local_progress:
            optimizer.zero_grad(set_to_none=True)
            data_dict = forward_batch(model, batch, device)
            loss = data_dict["loss"].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({"lr": lr_scheduler.get_lr()})

            local_progress.set_postfix({k: v.item() if torch.is_tensor(v) else v for k, v in data_dict.items()})
            logger.update_scalers(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            accuracy = knn_monitor(
                unwrap_model(model).backbone,
                memory_loader,
                test_loader,
                device=device,
                k=min(args.train.knn_k, len(memory_loader.dataset)),
                hide_progress=args.hide_progress,
            )

        saving_freq = getattr(args.train, "saving_frequency", None)
        if saving_freq and (epoch + 1) % saving_freq == 0:
            last_intermediate_ckpt = save_checkpoint(model, epoch, args)
        else:
            last_intermediate_ckpt = None

        epoch_dict = {"epoch": epoch, "accuracy": accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

    # Save final checkpoint unless the last epoch was already saved by saving_frequency.
    saving_freq = getattr(args.train, "saving_frequency", None)
    if saving_freq and args.train.stop_at_epoch % saving_freq == 0:
        model_path = last_intermediate_ckpt
    else:
        model_path = save_checkpoint(model, epoch, args)

    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)

    logger.finish()
    completed_log_dir = finalize_log_dir(args) if finalize_logs else args.log_dir
    selected_subset_indices_path = getattr(args, "selected_subset_indices_path", None)
    if finalize_logs:
        selected_subset_indices_path = _selected_indices_path_after_finalize(
            selected_subset_indices_path,
            completed_log_dir,
        )
        vars(args)["selected_subset_indices_path"] = selected_subset_indices_path
    return {
        "model_path": model_path,
        "accuracy": accuracy,
        "log_dir": completed_log_dir,
        "selected_subset_indices": getattr(args, "selected_subset_indices", None),
        "selected_subset_indices_path": selected_subset_indices_path,
    }


def main(device, args):
    return train_model(args=args, device=device, finalize_logs=False)


def build_train_loader_for_episode(args, episode_indices):
    """Build a train loader for a single meta-loop episode using explicit indices."""
    base_dataset = get_dataset(
        transform=None,
        train=True,
        **args.dataset_kwargs,
    )
    dataset = PseudoSupervisedDataset(
        dataset=base_dataset,
        image_size=args.dataset.image_size,
        source_pool_size=None,
        augment_probability=getattr(args.train, "augment_probability", 1.0),
        explicit_indices=episode_indices,
        samples_per_epoch=getattr(args.train, "samples_per_epoch", None),
        batch_size=args.train.batch_size,
        negatives_ratio=getattr(args.train, "negatives_ratio", None),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=args.train.batch_size,
        drop_last=args.dataloader_kwargs["drop_last"],
        pin_memory=args.dataloader_kwargs["pin_memory"],
        num_workers=_get_num_workers(args),
        persistent_workers=args.dataloader_kwargs.get("persistent_workers", False),
    )


def build_optimizer_and_scheduler_for_meta(model, train_loader, args, total_num_epochs, start_iter=0):
    """Build optimizer and LR scheduler with support for global meta-loop scheduling."""
    optimizer = get_optimizer(
        args.train.optimizer.name,
        model,
        lr=args.train.base_lr * args.train.batch_size / 256,
        momentum=getattr(args.train.optimizer, "momentum", 0.9),
        weight_decay=args.train.optimizer.weight_decay,
    )
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs,
        args.train.warmup_lr * args.train.batch_size / 256,
        total_num_epochs,
        args.train.base_lr * args.train.batch_size / 256,
        args.train.final_lr * args.train.batch_size / 256,
        len(train_loader),
        constant_predictor_lr=True,
        start_iter=start_iter,
    )
    return optimizer, lr_scheduler


def _split_backbone_head_optimizers(model, train_loader, args, total_num_epochs, start_iter=0):
    """Create separate backbone and classifier-head optimizers with independent LR schedules.

    Used when ``reset_classifier_head=True`` and ``global_lr_schedule=True``.
    The backbone optimizer follows the global schedule (spanning all episodes).
    The classifier-head optimizer uses a fresh per-episode warmup+cosine schedule
    so that the re-initialised head always starts training from a high LR rather
    than the low LR the global schedule may have reached by that episode.
    """
    head = get_classifier_head(model)
    if head is None:
        raise ValueError("The model does not expose a classifier head for split optimization.")
    head_ids = {id(p) for p in head.parameters()}

    lr = args.train.base_lr * args.train.batch_size / 256
    warmup_lr = args.train.warmup_lr * args.train.batch_size / 256
    final_lr = args.train.final_lr * args.train.batch_size / 256
    momentum = getattr(args.train.optimizer, "momentum", 0.9)
    wd = args.train.optimizer.weight_decay
    opt_name = args.train.optimizer.name
    iters_per_epoch = len(train_loader)

    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = list(head.parameters())

    def _make_opt(params):
        if opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        elif opt_name == "lars":
            from optimizers import LARS as _LARS
            return _LARS(params, lr=lr, momentum=momentum, weight_decay=wd)
        elif opt_name == "larc":
            from optimizers import LARC as _LARC
            return _LARC(
                torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd),
                trust_coefficient=0.001,
                clip=False,
            )
        else:
            raise NotImplementedError(
                f"Optimizer '{opt_name}' is not supported for split backbone/head scheduling."
            )

    backbone_opt = _make_opt(backbone_params)
    head_opt = _make_opt(head_params)

    backbone_scheduler = LR_Scheduler(
        backbone_opt,
        args.train.warmup_epochs,
        warmup_lr,
        total_num_epochs,
        lr,
        final_lr,
        iters_per_epoch,
        constant_predictor_lr=False,
        start_iter=start_iter,
    )
    # Fresh per-episode schedule for the classifier head.
    head_scheduler = LR_Scheduler(
        head_opt,
        args.train.warmup_epochs,
        warmup_lr,
        args.train.num_epochs,
        lr,
        final_lr,
        iters_per_epoch,
        constant_predictor_lr=False,
    )
    return backbone_opt, backbone_scheduler, head_opt, head_scheduler


def meta_train_model(
    args,
    device=None,
    resume_from_episode=0,
    resume_checkpoint=None,
    resume_mother_log_dir=None,
    resume_mother_ckpt_dir=None,
):
    """Train the pseudo-supervised model with a meta-loop over the full dataset.

    Each episode samples a fixed-size window from the globally-shuffled dataset,
    optionally overlapping with the previous episode.  The model checkpoint from
    one episode is loaded as the starting point for the next.

    To resume a previously interrupted run:
        - ``resume_from_episode``: 0-based index of the episode to restart from.
        - ``resume_checkpoint``: path to the ``.pth`` file saved at the end of
          episode ``resume_from_episode - 1``.
        - ``resume_mother_log_dir`` / ``resume_mother_ckpt_dir``: paths to the
          mother directories created during the original run so that new episode
          sub-folders are written into the same tree.
    """
    device = device or args.device

    meta_cfg = getattr(args, "meta", None)
    if meta_cfg is None:
        raise ValueError("'meta' section is required in the config for meta_train_model.")

    episode_size = int(meta_cfg.episode_size)
    overlap_ratio = float(getattr(meta_cfg, "overlap_ratio", 0.0))
    shuffle_seed = int(getattr(meta_cfg, "dataset_shuffle_seed", 42))
    reset_head = bool(getattr(meta_cfg, "reset_classifier_head", False))
    global_lr = bool(getattr(meta_cfg, "global_lr_schedule", True))

    # Determine total dataset size using a throw-away dataset object.
    base_dataset = get_dataset(transform=None, train=True, **args.dataset_kwargs)
    total_size = len(base_dataset)
    del base_dataset

    sampling_mode = getattr(meta_cfg, "sampling_mode", "sequential")
    if sampling_mode == "random":
        num_episodes = int(meta_cfg.num_episodes)
        stride = None
        perm = None
    else:
        stride = max(1, int(episode_size * (1.0 - overlap_ratio)))
        if total_size <= episode_size:
            num_episodes = 1
        else:
            num_episodes = math.ceil((total_size - episode_size) / stride) + 1
        # Globally shuffle the dataset indices for reproducibility.
        generator = torch.Generator().manual_seed(shuffle_seed)
        perm = torch.randperm(total_size, generator=generator).tolist()

    # Create or reuse the mother output directory.
    if resume_mother_log_dir is not None or resume_mother_ckpt_dir is not None:
        if resume_mother_log_dir is None or resume_mother_ckpt_dir is None:
            raise ValueError(
                "Both resume_mother_log_dir and resume_mother_ckpt_dir must be provided together."
            )
        mother_log_dir = resume_mother_log_dir
        mother_ckpt_dir = resume_mother_ckpt_dir
        os.makedirs(mother_log_dir, exist_ok=True)
        os.makedirs(mother_ckpt_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        mother_name = f"{timestamp}_meta_training_{args.dataset.name}"
        mother_ckpt_dir = os.path.join(args.ckpt_dir, mother_name)
        mother_log_dir = os.path.join(args.log_dir, mother_name)
        os.makedirs(mother_ckpt_dir, exist_ok=True)
        os.makedirs(mother_log_dir, exist_ok=True)

    # Save the merged runtime config into the mother log directory for reference.
    save_effective_config(args, mother_log_dir)

    resuming = resume_from_episode > 0 or resume_checkpoint is not None
    if sampling_mode == "random":
        mode_info = f"sampling=random | num_episodes={num_episodes}"
    else:
        mode_info = f"overlap={overlap_ratio:.0%} | stride={stride}"
    print(
        f"Meta-training: {num_episodes} episodes | episode_size={episode_size} | "
        f"{mode_info} | total_dataset={total_size}"
        + (f" | resuming from episode {resume_from_episode}" if resuming else "")
    )

    last_checkpoint = resume_checkpoint
    model = None
    global_iter_offset = 0
    global_total_epochs = num_episodes * args.train.num_epochs if global_lr else None

    # When resuming with global LR scheduling, fast-forward the iter offset to
    # match where the interrupted run would have been.
    if global_lr and resume_from_episode > 0:
        dummy_indices = list(range(episode_size)) if perm is None else perm[:episode_size]
        temp_loader = build_train_loader_for_episode(args, dummy_indices)
        global_iter_offset = resume_from_episode * len(temp_loader) * args.train.num_epochs
        del temp_loader

    results = []

    # A single logger that accumulates data across *all* episodes so that the
    # saved plot always shows the full training history up to the latest step.
    meta_logger = Logger(
        tensorboard=False,
        matplotlib=args.logger.matplotlib,
        log_dir=mother_log_dir,
    )

    for episode_idx in range(resume_from_episode, num_episodes):
        print(f"\n--- Episode {episode_idx + 1}/{num_episodes} ---")

        # Compute per-episode indices.
        if sampling_mode == "random":
            ep_gen = torch.Generator().manual_seed(shuffle_seed + episode_idx)
            episode_indices = torch.randperm(total_size, generator=ep_gen)[:episode_size].tolist()
        else:
            start = episode_idx * stride
            end = min(start + episode_size, total_size)
            episode_indices = perm[start:end]

        # Build loaders.
        train_loader = build_train_loader_for_episode(args, episode_indices)
        memory_loader = build_eval_loader(args, train=True)
        test_loader = build_eval_loader(args, train=False)

        # Derive per-episode directories.
        episode_dir = os.path.join(mother_log_dir, f"episode_{episode_idx}")
        episode_ckpt_dir = os.path.join(mother_ckpt_dir, f"episode_{episode_idx}")
        os.makedirs(episode_dir, exist_ok=True)
        os.makedirs(episode_ckpt_dir, exist_ok=True)

        num_pseudo_classes = train_loader.dataset.num_pseudo_classes

        if model is None:
            # Build model (fresh start or first episode after resuming).
            model = maybe_data_parallel(
                get_model(args.model, num_classes=num_pseudo_classes).to(device),
                args,
            )

        # Load checkpoint: either a resume checkpoint (first iteration of a
        # resumed run) or the checkpoint saved at the end of the previous episode.
        if last_checkpoint is not None:
            state = torch.load(last_checkpoint, map_location=device)
            unwrap_model(model).load_state_dict(state["state_dict"])
            if reset_head:
                reset_classifier_head(model)

        # Build optimizer + LR scheduler.
        # Split backbone and head optimizers when optimizer_separate=True (explicit)
        # or when global_lr + reset_head (existing automatic behavior).
        optimizer_separate = bool(getattr(args.train, "optimizer_separate", False))
        use_split_head = (
            (optimizer_separate or (global_lr and reset_head))
            and get_classifier_head(model) is not None
        )
        if use_split_head:
            total_ep = global_total_epochs if global_lr else args.train.num_epochs
            start_it = global_iter_offset if global_lr else 0
            optimizer, lr_scheduler, head_optimizer, head_lr_scheduler = (
                _split_backbone_head_optimizers(
                    model, train_loader, args,
                    total_num_epochs=total_ep,
                    start_iter=start_it,
                )
            )
        elif global_lr:
            optimizer, lr_scheduler = build_optimizer_and_scheduler_for_meta(
                model,
                train_loader,
                args,
                total_num_epochs=global_total_epochs,
                start_iter=global_iter_offset,
            )
            head_optimizer = head_lr_scheduler = None
        else:
            optimizer, lr_scheduler = build_optimizer_and_scheduler(model, train_loader, args)
            head_optimizer = head_lr_scheduler = None

        # Per-episode args proxy for save_checkpoint / Logger.
        episode_args = copy.copy(args)
        episode_args.log_dir = episode_dir
        episode_args.ckpt_dir = episode_ckpt_dir
        episode_args.name = f"episode_{episode_idx}"

        logger = Logger(
            tensorboard=args.logger.tensorboard,
            matplotlib=args.logger.matplotlib,
            log_dir=episode_dir,
        )

        accuracy = 0.0
        last_intermediate_ckpt = None
        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f"Episode {episode_idx}")
        for epoch in global_progress:
            model.train()
            local_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{args.train.num_epochs}",
                disable=args.hide_progress,
            )
            for batch in local_progress:
                optimizer.zero_grad(set_to_none=True)
                if head_optimizer is not None:
                    head_optimizer.zero_grad(set_to_none=True)
                data_dict = forward_batch(model, batch, device)
                loss = data_dict["loss"].mean()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if head_optimizer is not None:
                    head_optimizer.step()
                    head_lr_scheduler.step()
                if global_lr:
                    global_iter_offset += 1
                data_dict.update({"lr": lr_scheduler.get_lr()})
                local_progress.set_postfix(
                    {k: v.item() if torch.is_tensor(v) else v for k, v in data_dict.items()}
                )
                logger.update_scalers(data_dict)
                meta_logger.update_scalers(data_dict)

            if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
                accuracy = knn_monitor(
                    unwrap_model(model).backbone,
                    memory_loader,
                    test_loader,
                    device=device,
                    k=min(args.train.knn_k, len(memory_loader.dataset)),
                    hide_progress=args.hide_progress,
                )

            saving_freq = getattr(args.train, "saving_frequency", None)
            if saving_freq and (epoch + 1) % saving_freq == 0:
                last_intermediate_ckpt = save_checkpoint(model, epoch, episode_args)
            else:
                last_intermediate_ckpt = None

            epoch_dict = {"epoch": epoch, "accuracy": accuracy}
            global_progress.set_postfix(epoch_dict)
            logger.update_scalers(epoch_dict)
            meta_logger.update_scalers(epoch_dict)

        # Save end-of-episode checkpoint unless already saved by saving_frequency.
        saving_freq = getattr(args.train, "saving_frequency", None)
        if saving_freq and args.train.stop_at_epoch % saving_freq == 0:
            last_checkpoint = last_intermediate_ckpt
        else:
            last_checkpoint = save_checkpoint(model, epoch, episode_args)
        results.append(
            {
                "episode": episode_idx,
                "model_path": last_checkpoint,
                "accuracy": accuracy,
                "log_dir": episode_dir,
            }
        )

    return {
        "episodes": results,
        "final_model_path": last_checkpoint,
        "final_accuracy": results[-1]["accuracy"] if results else None,
        "mother_log_dir": mother_log_dir,
        "mother_ckpt_dir": mother_ckpt_dir,
        "meta_plot": os.path.join(mother_log_dir, "plotter.svg"),
    }


if __name__ == "__main__":
    args = get_args()
    train_model(args=args, device=args.device, finalize_logs=True)
