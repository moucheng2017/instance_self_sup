import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main
from datasets.pseudo_supervised import PseudoSupervisedDataset
from datasets.subset import maybe_select_subset, select_subset_indices, subset_with_metadata


class TinyDataset(torch.utils.data.Dataset):
    classes = ["a", "b"]

    def __init__(self, n=32):
        self.targets = [i % 2 for i in range(n)]
        self.labels = np.array(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor([idx]), self.targets[idx]


def _args(model_name="simclr", subset_n=8, batch_size=4):
    return SimpleNamespace(
        model=SimpleNamespace(name=model_name),
        dataset=SimpleNamespace(image_size=32),
        train=SimpleNamespace(
            subset_n=subset_n,
            subset_seed=123,
            batch_size=batch_size,
            source_pool_size=None,
            augment_probability=1.0,
            samples_per_epoch=None,
            negatives_ratio=None,
        ),
        dataset_kwargs={"dataset": "cifar10", "data_dir": "", "download": False, "debug_subset_size": None},
        aug_kwargs={"name": model_name, "image_size": 32},
        dataloader_kwargs={"drop_last": True, "pin_memory": False, "num_workers": 0},
        log_dir=None,
    )


def test_select_subset_indices_seed_only_invariant():
    first = select_subset_indices(100, subset_n=10, subset_seed=7)
    torch.rand(100)
    np.random.rand(100)
    random.random()
    second = select_subset_indices(100, subset_n=10, subset_seed=7)
    _ = select_subset_indices(100, subset_n=10, subset_seed=8)
    third = select_subset_indices(100, subset_n=10, subset_seed=7)
    assert first == second == third
    assert first == sorted(first)
    assert first != select_subset_indices(100, subset_n=10, subset_seed=9)


def test_subset_with_metadata_preserves_subset_local_labels():
    dataset, indices = maybe_select_subset(TinyDataset(32), subset_n=8, subset_seed=3)
    assert len(dataset) == 8
    assert dataset.classes == ["a", "b"]
    assert dataset.targets == [TinyDataset(32).targets[i] for i in indices]
    assert dataset.labels.tolist() == dataset.targets


def test_pseudo_supervised_dataset_uses_explicit_selected_pool():
    indices = select_subset_indices(32, subset_n=10, subset_seed=5)
    dataset = PseudoSupervisedDataset(
        TinyDataset(32),
        image_size=32,
        source_pool_size=None,
        explicit_indices=indices,
        batch_size=5,
    )
    assert dataset.source_indices == indices
    assert dataset.num_pseudo_classes == 10


def test_build_train_loader_regular_subset(monkeypatch):
    monkeypatch.setattr(main, "get_dataset", lambda **kwargs: TinyDataset(32))
    monkeypatch.setattr(main, "get_aug", lambda **kwargs: None)
    args = _args("simclr", subset_n=8, batch_size=4)
    loader = main.build_train_loader(args)
    assert len(loader.dataset) == 8
    assert args.selected_subset_indices == select_subset_indices(32, 8, 123)
    assert len(loader) == 2


def test_build_train_loader_pseudo_supervised_explicit_indices(monkeypatch):
    monkeypatch.setattr(main, "get_dataset", lambda **kwargs: TinyDataset(32))
    args = _args("pseudo_supervised_net", subset_n=8, batch_size=4)
    loader = main.build_train_loader(args)
    assert loader.dataset.source_indices == select_subset_indices(32, 8, 123)
    assert loader.dataset.num_pseudo_classes == 8


def test_build_train_loader_raises_for_empty_drop_last(monkeypatch):
    monkeypatch.setattr(main, "get_dataset", lambda **kwargs: TinyDataset(32))
    monkeypatch.setattr(main, "get_aug", lambda **kwargs: None)
    with pytest.raises(ValueError, match="would be empty"):
        main.build_train_loader(_args("simclr", subset_n=4, batch_size=8))


def test_selected_indices_path_tracks_finalized_log_dir():
    path = "/tmp/in-progress_0101_run/subset_indices.json"
    completed = "/tmp/completed_0101_run"

    assert main._selected_indices_path_after_finalize(path, completed) == (
        "/tmp/completed_0101_run/subset_indices.json"
    )
    assert main._selected_indices_path_after_finalize(None, completed) is None
