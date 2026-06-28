import warnings

import numpy as np
import torch
import torch.nn.functional as torch_F

from augmentations import get_aug
from datasets import get_dataset
from datasets.subset import subset_with_metadata


def _as_tensor(features):
    return features if torch.is_tensor(features) else torch.as_tensor(features)


def _singular_values(features):
    features = _as_tensor(features).detach().float()
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D].")
    return torch.linalg.svdvals(features).cpu().numpy()


def effective_rank(F):
    """Return exp(H(p)) using natural-log entropy over normalized singular values."""
    singular_values = _singular_values(F)
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    p = singular_values / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def spectral_diagnostics(F):
    singular_values = _singular_values(F)
    sq = singular_values ** 2
    total = sq.sum()
    if total > 0:
        explained_variance = np.cumsum(sq / total)
    else:
        explained_variance = np.zeros_like(singular_values)
    return singular_values, effective_rank(F), explained_variance


def knn_eval(F, labels, k=20, n_train=None):
    features = torch_F.normalize(_as_tensor(F).float(), dim=1)
    labels = _as_tensor(labels).long()
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same number of rows.")
    if features.shape[0] < 2:
        raise ValueError("at least two samples are required for KNN evaluation.")

    n_train = int(0.8 * len(features)) if n_train is None else int(n_train)
    if n_train < 1 or n_train >= len(features):
        raise ValueError("n_train must leave at least one train and one test sample.")
    k = min(int(k), n_train)

    train_features = features[:n_train]
    train_labels = labels[:n_train]
    test_features = features[n_train:]
    test_labels = labels[n_train:]

    sim = test_features @ train_features.t()
    nn_idx = sim.topk(k=k, dim=1).indices
    nn_labels = train_labels[nn_idx]
    preds = []
    for row in nn_labels:
        values, counts = row.unique(return_counts=True)
        preds.append(values[counts.argmax()])
    preds = torch.stack(preds)
    return float((preds == test_labels).float().mean().item())


def _sampler_is_sequential(sampler):
    """A non-shuffling DataLoader uses SequentialSampler (the default for shuffle=False).

    We accept that exact type only; any other sampler (RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler, or a third-party/custom sampler) is treated as potentially
    shuffling and therefore unsafe for the diagnostics loader (design.md §3.4 / §3.5).
    """
    return isinstance(sampler, torch.utils.data.SequentialSampler)


def _loader_is_shuffling(loader):
    """Return True unless we can positively confirm the loader iterates in fixed order.

    Structural (not name-based) check so a custom or subset random sampler cannot slip
    past the non-shuffling guard. A loader is considered non-shuffling only when its
    (batch_)sampler is a plain SequentialSampler. When a custom ``batch_sampler`` is
    supplied, ``loader.sampler`` is a SequentialSampler placeholder, so we inspect the
    batch_sampler's underlying sampler instead and require it to be sequential too.
    """
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None:
        inner = getattr(batch_sampler, "sampler", None)
        if inner is not None:
            # Default BatchSampler wraps loader.sampler; require it to be sequential.
            return not _sampler_is_sequential(inner)
        # Custom batch_sampler with no inspectable inner sampler -> cannot confirm order.
        if not isinstance(batch_sampler, torch.utils.data.BatchSampler):
            return True

    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        return False
    return not _sampler_is_sequential(sampler)


def extract_features(backbone, loader, device, n_samples=1000):
    if _loader_is_shuffling(loader):
        raise ValueError(
            "extract_features requires a non-shuffling loader "
            "(expected a SequentialSampler / shuffle=False)."
        )

    backbone.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            if isinstance(data, (tuple, list)):
                raise ValueError("diagnostics loader must yield single-view inputs.")
            feature = backbone(data.to(device, non_blocking=True))
            features.append(feature.detach().cpu())
            labels.append(target.detach().cpu())
            if sum(batch.shape[0] for batch in features) >= n_samples:
                break

    F_out = torch.cat(features, dim=0)[:n_samples]
    labels_out = torch.cat(labels, dim=0)[:n_samples]
    if F_out.shape[0] < n_samples:
        # Defense-in-depth for the same-data invariant (design.md §3.4 / §3.5 item 2):
        # the diagnostics loader is expected to be the shared N-pool, so a short matrix
        # means the caller passed a mismatched loader/n_samples and the effective-rank /
        # KNN split would be computed over the wrong set of images.
        warnings.warn(
            f"extract_features requested n_samples={n_samples} but the loader yielded "
            f"only {F_out.shape[0]} samples; the feature matrix is under-filled. "
            "Ensure the diagnostics loader covers the full N-pool.",
            stacklevel=2,
        )
    return F_out, labels_out


def build_diagnostics_loader(args, indices, batch_size=None):
    dataset = get_dataset(
        transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
        train=True,
        **args.dataset_kwargs,
    )
    if indices is not None:
        dataset = subset_with_metadata(dataset, list(indices))
    dataloader_kwargs = dict(args.dataloader_kwargs)
    dataloader_kwargs["drop_last"] = False
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size or args.train.batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )
