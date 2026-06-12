import math

import pytest
import torch

from models.hierarchical_kway_vmf_ot_self_labeling_net import (
    HierarchicalKWayVMFOTSelfLabelingNet,
    resolve_rank_schedule,
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
        "rank_schedule": [2, 2, 2],
        "kappa": 5.0,
        "ot_epsilon": 0.1,
        "sinkhorn_iters": 20,
        "em_iters": 5,
        "batch_self_labeling": True,
    }
    defaults.update(kwargs)
    return HierarchicalKWayVMFOTSelfLabelingNet(**defaults)


def _images(n):
    return torch.randn(n, 3, 4, 4)


def test_resolve_rank_schedule_validates_explicit_list():
    assert resolve_rank_schedule(rank_schedule=[3, 4]) == [3, 4]
    with pytest.raises(ValueError):
        resolve_rank_schedule(rank_schedule=[1, 4])  # rank < 2


def test_dp_derived_schedule_when_no_explicit_list():
    model = build_model(rank_schedule=None, num_leaf_clusters=64, rank_schedule_max_rank=8)
    assert math.prod(model.rank_schedule) == 64
    assert all(2 <= r <= 8 for r in model.rank_schedule)


def test_depth_and_supervised_depth_wiring():
    model = build_model(rank_schedule=[2, 3, 4], supervised_depth=2)
    assert model.depth == 3
    assert model.supervised_depth == 2
    assert model.active_depth == 2
    # set_active_depth is clamped to supervised_depth.
    model.set_active_depth(3)
    assert model.active_depth == 2
    model.set_active_depth(1)
    assert model.active_depth == 1


def test_node_levels_buffer_matches_schedule():
    model = build_model(rank_schedule=[2, 3])
    # level 0: 1 node, level 1: 2 nodes.
    assert model.n_internal == 1 + 2
    assert model.node_levels.tolist() == [0, 1, 1]


def test_balanced_kway_build_splits_into_uniform_leaves():
    # 27 points, schedule [3,3,3] -> 27 leaves, balanced => 1 per leaf.
    model = build_model(num_classes=27, rank_schedule=[3, 3, 3], sinkhorn_iters=50)
    embeddings = torch.randn(27, 8)
    _, paths, _, masks, stats = model._build_kway_tree(
        embeddings, fallback_prototypes=model._prepared_prototypes()
    )
    assert masks.all()
    leaf_ids = torch.zeros(27, dtype=torch.long)
    for t in range(model.depth):
        leaf_ids = leaf_ids * model.rank_schedule[t] + paths[:, t]
    counts = torch.bincount(leaf_ids, minlength=27)
    # Balanced OT should fill every leaf (allow a little slack for argmax ties).
    assert stats["tree_nonempty_leaves"] >= 24
    assert counts.max().item() <= 3


def test_build_masks_inactive_levels_when_node_too_small():
    # 4 points but a node would need >= 4 children at deeper level.
    model = build_model(num_classes=8, rank_schedule=[2, 4], sinkhorn_iters=30)
    embeddings = torch.randn(4, 8)
    _, _, _, masks, _ = model._build_kway_tree(
        embeddings, fallback_prototypes=model._prepared_prototypes()
    )
    # Level 0 (r=2) splits 4 points into 2 children of ~2 each; each child has
    # < 4 points so the level-1 r=4 split cannot run -> masked.
    assert masks[:, 0].all()
    assert not masks[:, 1].any()


def test_forward_returns_finite_loss_and_backprops():
    model = build_model(rank_schedule=[2, 2, 2])
    images = _images(32)
    pseudo_labels = torch.arange(32)
    out = model(images, pseudo_labels)
    assert torch.isfinite(out["loss"])
    assert 0.0 <= out["acc"].item() <= 1.0
    out["loss"].backward()
    grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_supervised_depth_caps_active_levels_in_loss():
    model = build_model(rank_schedule=[2, 2, 2], supervised_depth=1)
    images = _images(32)
    out = model(images, torch.arange(32))
    assert out["active_depth"] == 1.0


def test_unbalanced_mode_runs_and_respects_collapse_guard():
    model = build_model(
        rank_schedule=[2, 2],
        ot_unbalanced_tau=0.01,
        ot_unbalanced_min_split_fraction=0.2,
        sinkhorn_iters=20,
    )
    embeddings = torch.randn(16, 8)
    _, paths, _, masks, stats = model._build_kway_tree(
        embeddings, fallback_prototypes=model._prepared_prototypes()
    )
    # Collapse guard ensures no built level collapses below the floor fraction.
    assert stats["tree_min_split_fraction"] >= 0.2 - 1e-6 or stats["tree_min_split_fraction"] == 0.0


def test_non_batch_local_uses_stored_assignments():
    model = build_model(rank_schedule=[2, 2], batch_self_labeling=False, supervised_depth=2)
    images = _images(8)
    pseudo_labels = torch.arange(8)
    out = model(images, pseudo_labels)
    assert torch.isfinite(out["loss"])


def test_predict_paths_is_read_only_and_mixed_radix():
    model = build_model(rank_schedule=[3, 2])
    fitted_before = model.node_fitted.clone()
    prototypes_before = model.node_prototypes.clone()
    embeddings = torch.randn(12, 8)

    paths = model.predict_paths(embeddings)

    assert paths.shape == (12, 2)
    assert paths[:, 0].max().item() < 3
    assert paths[:, 1].max().item() < 2
    # Read-only: no tree state mutates (unlike the builders).
    assert torch.equal(model.node_fitted, fitted_before)
    assert torch.equal(model.node_prototypes, prototypes_before)
    # Deterministic descent.
    assert torch.equal(paths, model.predict_paths(embeddings))


def test_predict_paths_follows_prototype_geometry():
    model = build_model(rank_schedule=[2, 2])
    # Root prototypes at +/- e0; both level-1 nodes split on +/- e1.
    prototypes = torch.zeros_like(model.node_prototypes)
    prototypes[0, 0, 0] = 1.0
    prototypes[0, 1, 0] = -1.0
    for node in (1, 2):
        prototypes[node, 0, 1] = 1.0
        prototypes[node, 1, 1] = -1.0
    model.node_prototypes.copy_(prototypes)

    z = torch.tensor(
        [
            [1.0, 1.0, 0, 0, 0, 0, 0, 0],
            [1.0, -1.0, 0, 0, 0, 0, 0, 0],
            [-1.0, 1.0, 0, 0, 0, 0, 0, 0],
            [-1.0, -1.0, 0, 0, 0, 0, 0, 0],
        ]
    )
    paths = model.predict_paths(z)
    assert paths.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_finalize_node_stats_reports_levels_and_resets():
    model = build_model(rank_schedule=[2, 2])
    model(_images(32), torch.arange(32))
    stats = model.finalize_node_stats()
    assert "tree_nodes_visited" in stats
    assert model.node_count.sum().item() == 0  # reset
