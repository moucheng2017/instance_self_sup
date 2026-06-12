import os
from datetime import datetime

import torch
from tqdm import tqdm

from arguments import get_args
from augmentations import get_aug
from datasets import get_dataset
from datasets.pseudo_supervised import PseudoSupervisedDataset
from linear_eval import load_backbone_weights, main as linear_eval
from models import get_model
from models.hierarchical_balanced_vmf_self_labeling_net import compute_active_depth, compute_unbalanced_tau
from optimizers import LR_Scheduler, get_optimizer
from tools import Logger, knn_monitor
from tools.monitor_plots import (
    append_history,
    save_training_monitor_svg,
    save_tree_health_monitor_svg,
    save_tree_structure_monitor_svg,
)
from tools.tree_metrics import prefix_label_metrics


class PseudoSourcePoolDataset(torch.utils.data.Dataset):
    def __init__(self, pseudo_dataset):
        self.dataset = pseudo_dataset.dataset
        self.source_indices = pseudo_dataset.source_indices
        self.transform = pseudo_dataset.transform

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, idx):
        image, _ = self.dataset[self.source_indices[idx]]
        image_tensor = self.transform.postprocess(self.transform.clean(image))
        return image_tensor, torch.tensor(idx, dtype=torch.long)


def maybe_data_parallel(model, args):
    use_data_parallel = getattr(args.train, "use_data_parallel", False)
    if str(args.device).startswith("cuda") and torch.cuda.device_count() > 1 and use_data_parallel:
        return torch.nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_num_workers(args):
    return args.dataloader_kwargs.get("num_workers", 0)


def build_train_loader(args):

    if args.model.name in (
        "pseudo_supervised_net",
        "hierarchical_pseudo_supervised_net",
        "sigmoid_pseudo_supervised_net",
        "low_rank_multitarget_pseudo_supervised_net",
        "topk_categorical_bottleneck_pic_net",
        "hierarchical_balanced_vmf_self_labeling_net",
        "hierarchical_kway_vmf_ot_self_labeling_net",
    ):
        base_dataset = get_dataset(
            transform=None,
            train=True,
            **args.dataset_kwargs,
        )
        dataset = PseudoSupervisedDataset(
            dataset=base_dataset,
            image_size=args.dataset.image_size,
            source_pool_size=getattr(args.train, "source_pool_size", None),
            augment_probability=getattr(args.train, "augment_probability", 1.0),
            subset_seed=getattr(args.train, "source_subset_seed", 0),
            samples_per_epoch=getattr(args.train, "samples_per_epoch", None),
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

    train_dataset = get_dataset(
        transform=get_aug(train=True, **args.aug_kwargs),
        train=True,
        **args.dataset_kwargs,
    )
    source_pool_size = getattr(args.train, "source_pool_size", None)
    if source_pool_size is not None:
        source_subset_seed = getattr(args.train, "source_subset_seed", 0)
        rng = torch.Generator()
        rng.manual_seed(source_subset_seed)
        indices = torch.randperm(len(train_dataset), generator=rng)[:int(source_pool_size)].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    samples_per_epoch = getattr(args.train, "samples_per_epoch", None)
    sampler = None
    shuffle = True
    if samples_per_epoch is not None:
        sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=int(samples_per_epoch)
        )
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs,
    )


def build_eval_loader(args, train):
    dataloader_kwargs = dict(args.dataloader_kwargs)
    dataloader_kwargs["drop_last"] = False
    # Allow a dedicated kNN eval dataset (e.g. the labeled split when training on the
    # unlabeled split). Falls back to the training dataset when not configured.
    knn_dataset_name = getattr(args.dataset, "knn_dataset", args.dataset.name)
    eval_dataset_kwargs = {**args.dataset_kwargs, "dataset": knn_dataset_name}
    return torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=train,
            **eval_dataset_kwargs,
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
    weight_decay = args.train.optimizer.weight_decay
    optimizer = get_optimizer(
        args.train.optimizer.name,
        model,
        lr=args.train.base_lr * args.train.batch_size / 256,
        momentum=getattr(args.train.optimizer, "momentum", 0.9),
        weight_decay=weight_decay,
        beta1=getattr(args.train.optimizer, "beta1", 0.9),
        beta2=getattr(args.train.optimizer, "beta2", 0.95),
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


def load_pretrained_backbone(model, args):
    """Load backbone weights from a checkpoint.

    Note: this restores backbone parameters only — optimizer state, LR schedule
    position, and prototype/sigmoid buffers are NOT restored.  This is weight
    initialisation, not a full training resume.
    """
    checkpoint_resume = getattr(args, "checkpoint_resume", None)
    if checkpoint_resume is None:
        return

    load_backbone_weights(unwrap_model(model).backbone, checkpoint_resume)
    print(f"Loaded backbone weights from {checkpoint_resume}")


def maybe_refresh_hierarchical_assignments(model, train_loader, epoch, args, device):
    module = unwrap_model(model)
    if not hasattr(module, "refresh_assignments"):
        return {}
    if getattr(module, "batch_self_labeling", False):
        return {}

    refresh_interval = int(getattr(args.train, "tree_refresh_interval", 1))
    if refresh_interval <= 0 or epoch % refresh_interval != 0:
        return {}

    pseudo_dataset = train_loader.dataset
    if not hasattr(pseudo_dataset, "source_indices"):
        raise ValueError("Hierarchical vMF self-labeling requires a pseudo-supervised source pool.")

    source_loader = torch.utils.data.DataLoader(
        dataset=PseudoSourcePoolDataset(pseudo_dataset),
        shuffle=False,
        batch_size=int(getattr(args.train, "tree_refresh_batch_size", args.train.batch_size)),
        drop_last=False,
        pin_memory=args.dataloader_kwargs["pin_memory"],
        num_workers=_get_num_workers(args),
        persistent_workers=args.dataloader_kwargs.get("persistent_workers", False),
    )

    was_training = model.training
    model.eval()
    embeddings = []
    with torch.no_grad():
        progress = tqdm(
            source_loader,
            desc=f"Refreshing hierarchical vMF tree {epoch}",
            disable=args.hide_progress,
        )
        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            embeddings.append(module.encode(images).detach().cpu())

    stats = module.refresh_assignments(torch.cat(embeddings, dim=0).to(device))
    stats_text = ", ".join(f"{key}={value}" for key, value in stats.items())
    print(f"Refreshed hierarchical vMF self-labels: {stats_text}")

    if was_training:
        model.train()
    return stats or {}


def maybe_update_depth_annealing(model, epoch, args):
    """Staircase depth annealing: activate one extra tree level every
    train.depth_annealing_epochs_per_level epochs, starting from
    train.depth_annealing_initial_depth (default 1). Levels beyond the active
    depth are built during refresh/batch self-labeling but receive no loss, so
    the backbone is never supervised on fine-scale splits of unconsolidated
    features. Disabled when the config key is absent or null."""
    module = unwrap_model(model)
    if not hasattr(module, "set_active_depth"):
        return

    epochs_per_level = getattr(args.train, "depth_annealing_epochs_per_level", None)
    if epochs_per_level is None:
        return

    active_depth = compute_active_depth(
        epoch=epoch,
        depth=module.depth,
        epochs_per_level=epochs_per_level,
        initial_depth=int(getattr(args.train, "depth_annealing_initial_depth", 1)),
    )
    if active_depth != module.active_depth:
        print(f"Depth annealing: epoch {epoch}, active_depth {module.active_depth} -> {active_depth}")
    module.set_active_depth(active_depth)


def maybe_update_unbalanced_tau(model, epoch, args):
    """Cosine-anneal the unbalanced-OT marginal penalty tau from
    train.unbalanced_tau_start to train.unbalanced_tau_final over
    train.unbalanced_tau_anneal_epochs (start nearly balanced, relax toward
    natural split ratios as features consolidate). Disabled when
    unbalanced_tau_final is absent or null — the model then keeps its
    constructor value: a constant tau, or None = exact balanced Sinkhorn."""
    module = unwrap_model(model)
    if not hasattr(module, "set_ot_unbalanced_tau"):
        return
    tau_final = getattr(args.train, "unbalanced_tau_final", None)
    if tau_final is None:
        return
    tau = compute_unbalanced_tau(
        epoch=epoch,
        tau_start=float(getattr(args.train, "unbalanced_tau_start", tau_final)),
        tau_final=float(tau_final),
        anneal_epochs=getattr(args.train, "unbalanced_tau_anneal_epochs", 0),
    )
    module.set_ot_unbalanced_tau(tau)


def maybe_finalize_tree_node_stats(model):
    """Fold per-node branch-accuracy accumulators into epoch-level health
    stats (per-level + overall node accuracy, low-acc streak candidates).
    Logging-only; returns {} for models without the hierarchical tree."""
    module = unwrap_model(model)
    if not hasattr(module, "finalize_node_stats"):
        return {}
    return module.finalize_node_stats() or {}


def maybe_compute_tree_structure_metrics(model, memory_loader, epoch, args, device):
    """Per-level purity/NMI between predicted tree-path prefixes and true labels.

    Diagnostic only (peeks at ground-truth labels; nothing feeds back into
    training). Runs every train.tree_metrics_interval epochs (0/null disables)
    on up to train.tree_metrics_max_samples clean memory-loader images, using
    the model's read-only predict_paths descent so no tree state mutates.
    Returns {} for models without predict_paths."""
    module = unwrap_model(model)
    if not hasattr(module, "predict_paths"):
        return {}
    interval = int(getattr(args.train, "tree_metrics_interval", 0) or 0)
    if interval <= 0 or epoch % interval != 0:
        return {}
    max_samples = int(getattr(args.train, "tree_metrics_max_samples", 10000))

    was_training = model.training
    model.eval()
    features, labels = [], []
    seen = 0
    with torch.no_grad():
        for images, targets in memory_loader:
            images = images.to(device, non_blocking=True)
            features.append(module.encode(images).cpu())
            labels.extend(int(t) for t in targets)
            seen += images.shape[0]
            if seen >= max_samples:
                break
    if was_training:
        model.train()
    if not features:
        return {}

    embeddings = torch.cat(features, dim=0)[:max_samples]
    paths = module.predict_paths(embeddings.to(device)).cpu().tolist()
    radices = getattr(module, "rank_schedule", None)
    return prefix_label_metrics(paths, labels[:max_samples], radices=radices)


def maybe_update_sigmoid_regularization_progress(model, epoch, batch_idx, train_loader, args):
    module = unwrap_model(model)
    if not hasattr(module, "set_sigmoid_regularization_progress"):
        return

    rampup_epochs = float(getattr(args.train, "sigmoid_regularization_rampup_epochs", 0))
    current_step = epoch * len(train_loader) + batch_idx + 1
    rampup_steps = int(rampup_epochs * len(train_loader))
    module.set_sigmoid_regularization_progress(current_step=current_step, rampup_steps=rampup_steps)


def save_checkpoint(model, epoch, args, extra_tags=None):
    tag = ""
    if extra_tags:
        tag = "_" + "_".join(str(item) for item in extra_tags)
    model_path = os.path.join(args.ckpt_dir, f"{args.name}{tag}_{datetime.now().strftime('%m%d%H%M%S')}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "state_dict": unwrap_model(model).state_dict(),
        },
        model_path,
    )
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, "checkpoint_path.txt"), "w+") as f:
        f.write(f"{model_path}")
    return model_path


def should_save_periodic_checkpoint(epoch, args):
    checkpoint_saving_frequency = getattr(args.train, "checkpoint_saving_frequency", None)
    if checkpoint_saving_frequency is None:
        return False
    checkpoint_saving_frequency = int(checkpoint_saving_frequency)
    if checkpoint_saving_frequency <= 0:
        return False
    completed_epoch = epoch + 1
    return completed_epoch % checkpoint_saving_frequency == 0


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

    model = maybe_data_parallel(
        get_model(args.model, num_classes=getattr(train_loader.dataset, "num_pseudo_classes", None)).to(device),
        args,
    )

    # Catch a silently broken configuration: if batch_self_labeling is disabled,
    # the model relies entirely on the stored assignment paths, which are only
    # updated by the full-pool refresh.  A tree_refresh_interval of 0 means no
    # refresh ever happens, so the model trains forever against the initial
    # round-robin pseudo-labels that have no relation to the actual features.
    _module = unwrap_model(model)
    if (
        hasattr(_module, "batch_self_labeling")
        and not _module.batch_self_labeling
        and hasattr(_module, "refresh_assignments")
        and int(getattr(args.train, "tree_refresh_interval", 0)) <= 0
    ):
        raise ValueError(
            "batch_self_labeling=False requires tree_refresh_interval > 0. "
            "With tree_refresh_interval=0 the assignment paths are never updated "
            "from their initial random state, so training is effectively random. "
            "Either set batch_self_labeling=True or set tree_refresh_interval to a "
            "positive integer (e.g. 1 to refresh every epoch)."
        )

    load_pretrained_backbone(model, args)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(model, train_loader, args)
    logger = Logger(
        tensorboard=args.logger.tensorboard,
        matplotlib=args.logger.matplotlib,
        log_dir=args.log_dir,
    )

    accuracy = 0.0
    epoch = -1  # guard against stop_at_epoch=0 leaving epoch unbound below
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc="Training")
    history = {}
    for epoch in global_progress:
        maybe_update_depth_annealing(model, epoch, args)
        maybe_update_unbalanced_tau(model, epoch, args)
        refresh_stats = maybe_refresh_hierarchical_assignments(model, train_loader, epoch, args, device)
        model.train()
        local_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.train.num_epochs}",
            disable=args.hide_progress,
        )
        epoch_sums, epoch_batches = {}, 0
        for batch_idx, batch in enumerate(local_progress):
            maybe_update_sigmoid_regularization_progress(model, epoch, batch_idx, train_loader, args)
            optimizer.zero_grad(set_to_none=True)
            data_dict = forward_batch(model, batch, device)
            loss = data_dict["loss"].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({"lr": lr_scheduler.get_lr()})

            epoch_batches += 1
            for key, value in data_dict.items():
                value = value.item() if torch.is_tensor(value) else float(value)
                epoch_sums[key] = epoch_sums.get(key, 0.0) + value

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

        tree_stats = maybe_finalize_tree_node_stats(model)
        structure_stats = maybe_compute_tree_structure_metrics(model, memory_loader, epoch, args, device)
        if structure_stats:
            tree_stats.update(structure_stats)
        if tree_stats and hasattr(unwrap_model(model), "select_reseed_nodes"):
            reseeded = unwrap_model(model).select_reseed_nodes()
            tree_stats["tree_reseeded_nodes"] = float(reseeded)
            if reseeded:
                print(f"Selective re-seeding: cold-restarting {reseeded} subtree root(s) at next refresh")
        current_tau = getattr(unwrap_model(model), "ot_unbalanced_tau", None)
        if current_tau is not None:
            tree_stats["ot_unbalanced_tau"] = float(current_tau)
        if tree_stats:
            summary = ", ".join(
                f"{key}={tree_stats[key]:.3f}"
                for key in ("tree_node_acc_overall", "tree_reseed_candidates")
                if key in tree_stats
            )
            if summary:
                print(f"Tree health (epoch {epoch}): {summary}")

        epoch_dict = {"epoch": epoch, "accuracy": accuracy, **tree_stats}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

        # Refresh the monitoring SVGs (training + tree health + tree structure)
        # every epoch.
        epoch_means = {key: total / max(epoch_batches, 1) for key, total in epoch_sums.items()}
        append_history(history, {**epoch_means, **refresh_stats, **tree_stats, "epoch": epoch, "accuracy": accuracy})
        try:
            save_training_monitor_svg(history, os.path.join(args.log_dir, "monitor_training.svg"))
            if any(key.startswith("tree_") or key == "active_depth" for key in history):
                save_tree_health_monitor_svg(history, os.path.join(args.log_dir, "monitor_tree_health.svg"))
            if any(key.startswith("tree_purity_level") for key in history):
                save_tree_structure_monitor_svg(history, os.path.join(args.log_dir, "monitor_tree_structure.svg"))
        except Exception as exc:
            print(f"Monitoring plot update failed (non-fatal): {exc}")

        if should_save_periodic_checkpoint(epoch, args):
            save_checkpoint(
                model,
                epoch,
                args,
                extra_tags=[
                    f"epoch_{epoch + 1}",
                    f"knn_acc{accuracy:.2f}",
                ],
            )

    model_path = save_checkpoint(model, epoch, args)

    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)

    completed_log_dir = finalize_log_dir(args) if finalize_logs else args.log_dir
    return {
        "model_path": model_path,
        "accuracy": accuracy,
        "log_dir": completed_log_dir,
    }


def main(device, args):
    return train_model(args=args, device=device, finalize_logs=False)


if __name__ == "__main__":
    args = get_args()
    train_model(args=args, device=args.device, finalize_logs=True)
