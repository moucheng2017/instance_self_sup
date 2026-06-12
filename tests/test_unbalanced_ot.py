import json
import math
import os

import torch
import torch.nn.functional as F
import yaml

from models.hierarchical_balanced_vmf_self_labeling_net import (
    HierarchicalBalancedVMFSelfLabelingNet,
    compute_unbalanced_tau,
)


class TinyBackbone(torch.nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 4 * 4, self.output_dim),
        )

    def forward(self, images):
        return self.net(images)


def build_model(**kwargs):
    defaults = {
        "num_classes": 32,
        "backbone": TinyBackbone(),
        "embedding_dim": 8,
        "depth": 3,
        "kappa": 5.0,
        "ot_epsilon": 0.1,
        "sinkhorn_iters": 30,
        "em_iters": 3,
        "batch_self_labeling": True,
        "sigmoid_init_temperature": 1.0,
        "sigmoid_init_bias": -3.0,
    }
    defaults.update(kwargs)
    return HierarchicalBalancedVMFSelfLabelingNet(**defaults)


def _two_cluster_embeddings(n_a=12, n_b=4, noise=0.03, seed=0):
    torch.manual_seed(seed)
    a = F.normalize(torch.tensor([[1.0] + [0.0] * 7]) + noise * torch.randn(n_a, 8), dim=1)
    b = F.normalize(torch.tensor([[-1.0] + [0.0] * 7]) + noise * torch.randn(n_b, 8), dim=1)
    return torch.cat([a, b], dim=0)


# ---------------------------------------------------------------- schedule


def test_compute_unbalanced_tau_cosine_anneal():
    # Endpoints.
    assert abs(compute_unbalanced_tau(0, tau_start=8.0, tau_final=0.1, anneal_epochs=100) - 8.0) < 1e-9
    assert abs(compute_unbalanced_tau(100, tau_start=8.0, tau_final=0.1, anneal_epochs=100) - 0.1) < 1e-9
    assert abs(compute_unbalanced_tau(500, tau_start=8.0, tau_final=0.1, anneal_epochs=100) - 0.1) < 1e-9
    # Midpoint of a cosine sits at the arithmetic mean; monotone decreasing.
    mid = compute_unbalanced_tau(50, tau_start=8.0, tau_final=0.1, anneal_epochs=100)
    assert abs(mid - (8.0 + 0.1) / 2) < 1e-9
    earlier = compute_unbalanced_tau(25, tau_start=8.0, tau_final=0.1, anneal_epochs=100)
    later = compute_unbalanced_tau(75, tau_start=8.0, tau_final=0.1, anneal_epochs=100)
    assert earlier > mid > later
    # Disabled schedule returns tau_final immediately.
    assert compute_unbalanced_tau(0, tau_start=8.0, tau_final=0.5, anneal_epochs=0) == 0.5
    assert compute_unbalanced_tau(0, tau_start=8.0, tau_final=0.5, anneal_epochs=None) == 0.5

    for bad in (
        {"tau_start": 0.0, "tau_final": 0.1, "anneal_epochs": 10},
        {"tau_start": 8.0, "tau_final": -1.0, "anneal_epochs": 10},
    ):
        try:
            compute_unbalanced_tau(0, **bad)
            assert False, f"Expected ValueError for {bad}"
        except ValueError:
            pass


# ------------------------------------------------- unbalanced sinkhorn core


def test_unbalanced_sinkhorn_recovers_balanced_at_large_tau():
    model = build_model()
    z = _two_cluster_embeddings()
    mu = F.normalize(torch.tensor([[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7]), dim=1)
    scores = model._ot_scores(z, mu)

    q_balanced = model._sinkhorn(scores, num_iters=50)
    q_unbalanced = model._unbalanced_sinkhorn(scores, tau=1e6, num_iters=50)

    assert torch.allclose(q_unbalanced, q_balanced, atol=1e-4)
    # Component masses forced to ~50/50 despite the 12/4 data.
    assert torch.allclose(q_unbalanced.sum(dim=0), torch.full((2,), 8.0), atol=1e-3)


def test_unbalanced_sinkhorn_finds_natural_ratio_at_small_tau():
    model = build_model()
    z = _two_cluster_embeddings(n_a=12, n_b=4)
    mu = F.normalize(torch.tensor([[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7]), dim=1)
    scores = model._ot_scores(z, mu)

    q = model._unbalanced_sinkhorn(scores, tau=0.01, num_iters=50)

    # Rows stay distributions over components.
    assert torch.allclose(q.sum(dim=1), torch.ones(16), atol=1e-4)
    # Argmax follows the data's 12/4 structure, not 8/8.
    assert torch.bincount(q.argmax(dim=1), minlength=2).tolist() == [12, 4]
    # Component masses approach the natural ratio.
    masses = q.sum(dim=0)
    assert masses[0] > 10.0 and masses[1] < 6.0


def test_unbalanced_sinkhorn_rows_sum_to_one_across_taus():
    model = build_model()
    torch.manual_seed(3)
    scores = torch.randn(32, 2)
    for tau in (1e6, 1.0, 0.1, 0.01):
        q = model._unbalanced_sinkhorn(scores, tau=tau, num_iters=40)
        assert torch.allclose(q.sum(dim=1), torch.ones(32), atol=1e-4), tau
        assert torch.isfinite(q).all()


# ----------------------------------------------------- fit_binary_ot wiring


def test_fit_binary_ot_unbalanced_split_follows_natural_ratio():
    model = build_model(ot_unbalanced_tau=0.01, tree_warm_start=False)
    z = _two_cluster_embeddings(n_a=12, n_b=4)
    init = torch.tensor([[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7])

    _, hard = model._fit_binary_ot(z, init_mu=init)

    assert sorted(torch.bincount(hard, minlength=2).tolist()) == [4, 12]


def test_fit_binary_ot_balanced_mode_unchanged():
    model = build_model(ot_unbalanced_tau=None)
    z = _two_cluster_embeddings(n_a=12, n_b=4)

    _, hard = model._fit_binary_ot(z)

    # Legacy behavior: exact median split regardless of natural ratio.
    assert torch.bincount(hard, minlength=2).tolist() == [8, 8]


def test_fit_binary_ot_min_split_fraction_guard():
    # A 15-vs-1 outlier configuration: the EM locks one prototype onto the
    # lone point and argmax yields a [1, 15] split, starving a child below
    # floor(0.2 * 16) = 3. The guard must fall back to the balanced median
    # split. (Note: a *single* tight blob does NOT trigger the guard — the EM
    # re-fits both prototypes inside the blob and finds a legitimate near-even
    # internal split.)
    model = build_model(ot_unbalanced_tau=0.01, ot_unbalanced_min_split_fraction=0.2)
    z = _two_cluster_embeddings(n_a=15, n_b=1)

    _, hard = model._fit_binary_ot(z)

    assert torch.bincount(hard, minlength=2).tolist() == [8, 8]


def test_tree_stats_report_split_fractions():
    model = build_model(num_classes=16, depth=2, ot_unbalanced_tau=0.01)
    z = _two_cluster_embeddings(n_a=12, n_b=4)

    _, _, _, _, stats = model._build_tree_from_embeddings(
        z, fallback_prototypes=model._prepared_node_prototypes()
    )

    assert 0.0 < stats["tree_min_split_fraction"] <= 0.5
    assert stats["tree_min_split_fraction"] <= stats["tree_mean_split_fraction"] <= 0.5


# --------------------------------------------------------------- validation


def test_set_ot_unbalanced_tau_validation():
    model = build_model()
    assert model.ot_unbalanced_tau is None  # default = legacy balanced

    model.set_ot_unbalanced_tau(0.5)
    assert model.ot_unbalanced_tau == 0.5
    model.set_ot_unbalanced_tau(None)
    assert model.ot_unbalanced_tau is None

    for bad in (0.0, -1.0):
        try:
            model.set_ot_unbalanced_tau(bad)
            assert False, f"Expected ValueError for {bad}"
        except ValueError:
            pass


def test_constructor_validates_min_split_fraction():
    for bad in (-0.1, 0.5, 0.9):
        try:
            build_model(ot_unbalanced_min_split_fraction=bad)
            assert False, f"Expected ValueError for {bad}"
        except ValueError:
            pass
    model = build_model(ot_unbalanced_tau=1.0, ot_unbalanced_min_split_fraction=0.1)
    assert model.ot_unbalanced_min_split_fraction == 0.1


def test_forward_backward_with_unbalanced_tau():
    model = build_model(
        num_classes=16, depth=2,
        ot_unbalanced_tau=0.1,
        prototype_ema_momentum=0.9,
        tree_warm_start=True,
    )
    images = torch.randn(16, 3, 4, 4)
    labels = torch.arange(16)

    out = model(images, labels)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert model.backbone.net[1].weight.grad.abs().sum().item() > 0


# ----------------------------------------------------- argument passing


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_get_model_passes_unbalanced_args():
    from arguments import Namespace
    from models import get_model

    cfg = Namespace({
        "name": "hierarchical_balanced_vmf_self_labeling_net",
        "backbone": "resnet18",
        "embedding_dim": 16,
        "depth": 2,
        "ot_unbalanced_tau": 0.7,
        "ot_unbalanced_min_split_fraction": 0.08,
    })
    model = get_model(cfg, num_classes=8)

    assert model.ot_unbalanced_tau == 0.7
    assert model.ot_unbalanced_min_split_fraction == 0.08

    cfg_default = Namespace({
        "name": "hierarchical_balanced_vmf_self_labeling_net",
        "backbone": "resnet18",
        "embedding_dim": 16,
        "depth": 2,
    })
    model_default = get_model(cfg_default, num_classes=8)
    assert model_default.ot_unbalanced_tau is None  # backward compatible


def test_unbalanced_config_keys_parse():
    path = os.path.join(_repo_root(), "configs", "hierarchical_unbalanced_vmf_cifar_colab.yaml")
    cfg = yaml.safe_load(open(path))

    assert cfg["model"]["name"] == "hierarchical_balanced_vmf_self_labeling_net"
    assert cfg["model"]["ot_unbalanced_tau"] > 0
    assert 0.0 <= cfg["model"]["ot_unbalanced_min_split_fraction"] < 0.5
    assert cfg["train"]["unbalanced_tau_start"] >= cfg["train"]["unbalanced_tau_final"] > 0
    assert cfg["train"]["unbalanced_tau_anneal_epochs"] > 0
    # The experiment needs warm start + annealing context.
    assert cfg["model"]["tree_warm_start"] is True


def test_balanced_config_defaults_keep_legacy_behavior():
    path = os.path.join(_repo_root(), "configs", "hierarchical_balanced_vmf_cifar_colab.yaml")
    cfg = yaml.safe_load(open(path))
    assert cfg["model"]["ot_unbalanced_tau"] is None
    assert cfg["train"]["unbalanced_tau_final"] is None


def test_unbalanced_notebook_overrides_present():
    path = os.path.join(_repo_root(), "notebooks", "hierarchical-unbalanced-vmf-cifar10-ssl.ipynb")
    nb = json.load(open(path))
    source = "".join("".join(cell["source"]) for cell in nb["cells"])

    assert "59-unbalanced-ot" in source
    assert "hierarchical_unbalanced_vmf_cifar_colab.yaml" in source
    for token in (
        "ot_unbalanced_tau",
        "ot_unbalanced_min_split_fraction",
        "unbalanced_tau_start",
        "unbalanced_tau_final",
        "unbalanced_tau_anneal_epochs",
    ):
        assert f"'{token}'" in source or f"{token} =" in source, token
