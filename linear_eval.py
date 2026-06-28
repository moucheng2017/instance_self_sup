import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from arguments import get_args
from augmentations import get_aug
from datasets import get_dataset
from models import get_backbone
from optimizers import LR_Scheduler, get_optimizer
from tools import AverageMeter


def maybe_data_parallel(model, device, use_data_parallel=False):
    if str(device).startswith("cuda") and torch.cuda.device_count() > 1 and use_data_parallel:
        return torch.nn.DataParallel(model)
    return model


def infer_num_classes(dataset):
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
        return len(dataset.dataset.classes)
    if hasattr(dataset, "targets"):
        return int(max(dataset.targets)) + 1
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        return int(max(dataset.dataset.targets)) + 1
    raise ValueError("Unable to infer the number of classes from the evaluation dataset.")


def load_backbone_weights(model, checkpoint_path):
    save_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = save_dict["state_dict"]
    if any(key.startswith("backbone.") for key in state_dict):
        backbone_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    else:
        backbone_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
    model.load_state_dict(backbone_state_dict, strict=True)


def main(args):
    dataloader_kwargs = dict(args.dataloader_kwargs)
    dataloader_kwargs["drop_last"] = False
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs,
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs,
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    model = get_backbone(args.model.backbone)
    classifier = nn.Linear(
        in_features=model.output_dim,
        out_features=infer_num_classes(train_loader.dataset),
        bias=True,
    ).to(args.device)

    assert args.eval_from is not None
    load_backbone_weights(model, args.eval_from)

    model = maybe_data_parallel(model.to(args.device), args.device, getattr(args.eval, "use_data_parallel", False))
    classifier = maybe_data_parallel(classifier, args.device, getattr(args.eval, "use_data_parallel", False))

    optimizer = get_optimizer(
        args.eval.optimizer.name,
        classifier,
        lr=args.eval.base_lr * args.eval.batch_size / 256,
        momentum=getattr(args.eval.optimizer, "momentum", 0.9),
        weight_decay=args.eval.optimizer.weight_decay,
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs,
        args.eval.warmup_lr * args.eval.batch_size / 256,
        args.eval.num_epochs,
        args.eval.base_lr * args.eval.batch_size / 256,
        args.eval.final_lr * args.eval.batch_size / 256,
        len(train_loader),
    )

    loss_meter = AverageMeter(name="Loss")
    acc_meter = AverageMeter(name="Accuracy")

    global_progress = tqdm(range(0, args.eval.num_epochs), desc="Evaluating")
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.eval.num_epochs}", disable=True)

        for images, labels in local_progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                feature = model(images.to(args.device, non_blocking=True))
            if getattr(args.model, "l2_norm_backbone_features", False):
                feature = F.normalize(feature, dim=1)
            preds = classifier(feature)
            loss = F.cross_entropy(preds, labels.to(args.device, non_blocking=True))
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({"lr": lr, "loss": loss_meter.val, "loss_avg": loss_meter.avg})

    classifier.eval()
    acc_meter.reset()
    for images, labels in test_loader:
        with torch.no_grad():
            feature = model(images.to(args.device, non_blocking=True))
            if getattr(args.model, "l2_norm_backbone_features", False):
                feature = F.normalize(feature, dim=1)
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device, non_blocking=True)).sum().item()
            acc_meter.update(correct / preds.shape[0])

    final_accuracy = acc_meter.avg * 100
    print(f"Accuracy = {final_accuracy:.2f}")
    return final_accuracy


if __name__ == "__main__":
    main(args=get_args())
