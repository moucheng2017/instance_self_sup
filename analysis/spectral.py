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


def _loader_is_shuffling(loader):
    sampler = getattr(loader, "sampler", None)
    return sampler is not None and sampler.__class__.__name__ == "RandomSampler"


def extract_features(backbone, loader, device, n_samples=1000):
    if _loader_is_shuffling(loader):
        raise ValueError("extract_features requires a non-shuffling loader.")

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
