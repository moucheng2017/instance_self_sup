import json
import os

import numpy as np
import torch


def select_subset_indices(dataset_size, subset_n=None, subset_seed=42):
    """Select a sorted, deterministic subset using only subset_n and subset_seed."""
    if subset_n is None:
        return None
    subset_n = int(subset_n)
    if subset_n < 1:
        raise ValueError("subset_n must be at least 1.")
    if subset_n > int(dataset_size):
        raise ValueError("subset_n cannot exceed the dataset size.")

    rng = np.random.RandomState(int(subset_seed))
    return sorted(rng.choice(int(dataset_size), size=subset_n, replace=False).tolist())


def subset_with_metadata(dataset, indices):
    """Build a Subset that preserves class/label metadata in subset-local order."""
    subset = torch.utils.data.Subset(dataset, list(indices))
    if hasattr(dataset, "classes"):
        subset.classes = dataset.classes
    if hasattr(dataset, "targets"):
        subset.targets = [dataset.targets[i] for i in indices]
    if hasattr(dataset, "labels"):
        labels = dataset.labels
        try:
            subset.labels = labels[list(indices)]
        except TypeError:
            subset.labels = [labels[i] for i in indices]
    return subset


def maybe_select_subset(dataset, subset_n=None, subset_seed=42):
    indices = select_subset_indices(len(dataset), subset_n=subset_n, subset_seed=subset_seed)
    if indices is None:
        return dataset, None
    return subset_with_metadata(dataset, indices), indices


def save_subset_indices(indices, log_dir, filename="subset_indices.json"):
    if indices is None or log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w") as f:
        json.dump(list(indices), f)
    return path


def load_subset_indices(path):
    with open(path, "r") as f:
        return json.load(f)
