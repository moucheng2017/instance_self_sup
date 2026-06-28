from types import SimpleNamespace

import pytest
import torch

from analysis import spectral


class TinyDataset(torch.utils.data.Dataset):
    classes = [0, 1]

    def __init__(self, n=6):
        self.targets = [i % 2 for i in range(n)]
        self.data = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx].repeat(3, 4, 4), torch.tensor(self.targets[idx])


class TinyBackbone(torch.nn.Module):
    output_dim = 512

    def __init__(self, scale=1.0):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(512)
        self.scale = scale

    def forward(self, x):
        base = x.flatten(1).mean(dim=1, keepdim=True) * self.scale
        return self.bn(base.repeat(1, 512))


def test_effective_rank_extremes():
    assert spectral.effective_rank(torch.eye(4)) == pytest.approx(4.0)
    rank_one = torch.ones(5, 3)
    assert spectral.effective_rank(rank_one) == pytest.approx(1.0)


def test_spectral_diagnostics_sorted_and_explained_variance():
    singular_values, erank, explained = spectral.spectral_diagnostics(torch.diag(torch.tensor([3.0, 2.0, 1.0])))
    assert singular_values.tolist() == sorted(singular_values.tolist(), reverse=True)
    assert erank > 1.0
    assert all(explained[i] <= explained[i + 1] for i in range(len(explained) - 1))
    assert explained[-1] == pytest.approx(1.0)


def test_knn_eval_separable_clusters_and_default_split():
    features = torch.tensor(
        [[1.0, 0.0], [1.1, 0.0], [-1.0, 0.0], [-1.1, 0.0], [0.9, 0.0], [-0.9, 0.0]]
    )
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    assert spectral.knn_eval(features, labels, k=1, n_train=4) == 1.0
    assert 0.0 <= spectral.knn_eval(features, labels, k=20, n_train=4) <= 1.0


def test_extract_features_is_eval_deterministic_raw_and_label_aligned():
    loader = torch.utils.data.DataLoader(TinyDataset(6), batch_size=2, shuffle=False, drop_last=False)
    backbone = TinyBackbone()
    backbone.train()
    f1, labels1 = spectral.extract_features(backbone, loader, device="cpu", n_samples=5)
    f2, labels2 = spectral.extract_features(backbone, loader, device="cpu", n_samples=5)
    assert f1.shape == (5, 512)
    assert torch.equal(labels1, labels2)
    assert torch.equal(f1, f2)
    assert not torch.allclose(f1.norm(dim=1), torch.ones(5))

    other_backbone = TinyBackbone(scale=2.0)
    _, labels3 = spectral.extract_features(other_backbone, loader, device="cpu", n_samples=5)
    assert torch.equal(labels1, labels3)


def test_extract_features_rejects_shuffling_loader():
    loader = torch.utils.data.DataLoader(TinyDataset(6), batch_size=2, shuffle=True)
    with pytest.raises(ValueError, match="non-shuffling"):
        spectral.extract_features(TinyBackbone(), loader, device="cpu", n_samples=5)


def test_build_diagnostics_loader_reuses_selected_indices(monkeypatch):
    monkeypatch.setattr(spectral, "get_dataset", lambda **kwargs: TinyDataset(6))
    monkeypatch.setattr(spectral, "get_aug", lambda **kwargs: None)
    args = SimpleNamespace(
        aug_kwargs={"name": "simclr", "image_size": 32},
        dataset_kwargs={"dataset": "cifar10", "data_dir": "", "download": False, "debug_subset_size": None},
        dataloader_kwargs={"drop_last": True, "pin_memory": False, "num_workers": 0},
        train=SimpleNamespace(batch_size=2),
    )
    loader = spectral.build_diagnostics_loader(args, indices=[4, 1, 3], batch_size=2)
    assert loader.drop_last is False
    assert loader.dataset.indices == [4, 1, 3]
    assert loader.dataset.targets == [0, 1, 1]
