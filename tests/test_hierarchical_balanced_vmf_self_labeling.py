import itertools

import torch
import torch.nn.functional as F

from models.hierarchical_balanced_vmf_self_labeling_net import (
    HierarchicalBalancedVMFSelfLabelingNet,
    _path_node_ids,
    _subtree_node_ids,
    compute_active_depth,
    select_reseed_indices,
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
        "sinkhorn_iters": 5,
        "em_iters": 3,
        "batch_self_labeling": True,
        "sigmoid_init_temperature": 1.0,
        "sigmoid_init_bias": -3.0,
    }
    defaults.update(kwargs)
    return HierarchicalBalancedVMFSelfLabelingNet(**defaults)


def test_sinkhorn_balances_transport_marginals():
    model = build_model()
    scores = torch.randn(32, 8)

    q = model._sinkhorn(scores, num_iters=model._FLAT_SINKHORN_MIN_ITERS)

    assert torch.allclose(q.sum(dim=1), torch.ones(32), atol=1e-4)
    assert torch.allclose(q.sum(dim=0), torch.full((8,), 4.0), atol=1e-4)


def test_recursive_tree_splits_to_balanced_leaves_for_power_of_two_batch():
    for mode in ("hierarchical_vmf_ot", "recursive_ot"):
        model = build_model(
            num_classes=16,
            depth=4,
            ot_assignment_mode=mode,
            batch_self_labeling=True,
        )
        embeddings = torch.randn(16, 8)

        _, paths, _, masks, stats = model._build_tree_from_embeddings(
            embeddings,
            fallback_prototypes=model._prepared_node_prototypes(),
        )
        leaf_ids = torch.zeros(16, dtype=torch.long)
        for level in range(model.depth):
            leaf_ids = leaf_ids * 2 + paths[:, level].cpu()

        assert masks.all()
        assert stats["tree_nonempty_leaves"] == 16
        assert torch.bincount(leaf_ids, minlength=16).tolist() == [1] * 16


def test_recursive_tree_masks_inactive_levels_when_batch_is_too_small():
    model = build_model(num_classes=8, depth=4, ot_assignment_mode="hierarchical_vmf_ot")
    embeddings = torch.randn(4, 8)

    _, _, _, masks, stats = model._build_tree_from_embeddings(
        embeddings,
        fallback_prototypes=model._prepared_node_prototypes(),
    )

    assert stats["tree_nonempty_leaves"] == 4
    assert masks[:, :2].all()
    assert not masks[:, 2:].any()


def test_hierarchical_vmf_assigns_branches_without_equal_size_constraint():
    model = build_model(num_classes=10, depth=1, ot_assignment_mode="hierarchical_vmf")
    embeddings = torch.randn(10, 8)
    logits = torch.tensor([[10.0, 0.0]] * 8 + [[0.0, 10.0]] * 2)
    model._prediction_logits = lambda z, prototypes: logits.to(z.device)

    _, hard = model._fit_binary_ot(
        embeddings,
        fallback_mu=model._prepared_node_prototypes()[0],
    )

    assert torch.bincount(hard.cpu(), minlength=2).tolist() == [8, 2]


def test_flat_ot_uses_balanced_soft_transport_targets():
    model = build_model(
        num_classes=32,
        depth=3,
        ot_assignment_mode="flat_ot",
        sinkhorn_iters=200,
    )
    embeddings = torch.randn(32, 8)

    _, target_probs, labels, stats = model._build_flat_from_embeddings(
        embeddings,
        fallback_prototypes=model._prepared_flat_prototypes(),
    )

    assert target_probs.shape == (32, 8)
    assert labels.shape == (32,)
    assert torch.allclose(target_probs.sum(dim=1), torch.ones(32), atol=1e-4)
    assert torch.allclose(target_probs.sum(dim=0), torch.full((8,), 4.0), atol=5e-2)
    assert stats["flat_nonempty_prototypes"] == 8
    assert abs(stats["flat_min_transport_mass"] - 4.0) < 5e-2
    assert abs(stats["flat_max_transport_mass"] - 4.0) < 5e-2


def test_flat_vmf_ot_uses_requested_balanced_spherical_components():
    model = build_model(
        num_classes=30,
        depth=3,
        ot_assignment_mode="flat_vmf_ot",
        flat_vmf_num_components=5,
        sinkhorn_iters=200,
    )
    embeddings = torch.randn(30, 8)

    prototypes, target_probs, labels, stats = model._build_flat_from_embeddings(
        embeddings,
        fallback_prototypes=model._prepared_flat_prototypes(),
    )

    assert model.num_flat_prototypes == 5
    assert model.flat_prototypes.shape == (5, 8)
    assert prototypes.shape == (5, 8)
    assert target_probs.shape == (30, 5)
    assert labels.shape == (30,)
    assert torch.allclose(prototypes.norm(dim=1), torch.ones(5), atol=1e-5)
    assert torch.allclose(target_probs.sum(dim=1), torch.ones(30), atol=1e-4)
    assert torch.allclose(target_probs.sum(dim=0), torch.full((5,), 6.0), atol=5e-2)
    assert stats["flat_nonempty_prototypes"] == 5
    assert abs(stats["flat_min_transport_mass"] - 6.0) < 5e-2
    assert abs(stats["flat_max_transport_mass"] - 6.0) < 5e-2


def test_vmf_uses_unconstrained_hard_assignments():
    model = build_model(
        num_classes=8,
        depth=1,
        ot_assignment_mode="vmf",
        kappa=20.0,
        em_iters=1,
    )
    embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 7 + [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    fallback_prototypes = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    _, target_probs, labels, stats = model._build_vmf_from_embeddings(
        embeddings,
        fallback_prototypes=fallback_prototypes,
    )

    assert target_probs.shape == (8, 2)
    assert labels.shape == (8,)
    assert torch.allclose(target_probs.sum(dim=1), torch.ones(8))
    assert torch.bincount(labels, minlength=2).tolist() == [7, 1]
    assert target_probs.sum(dim=0).tolist() == [7.0, 1.0]
    assert stats["flat_min_assignment_count"] == 1
    assert stats["flat_max_assignment_count"] == 7


def test_hierachical_vmf_typo_is_accepted_as_alias():
    model = build_model(ot_assignment_mode="hierachical_vmf")

    assert model.ot_assignment_mode == "hierarchical_vmf"


def test_learnable_prototypes_and_ema_momentum_raises():
    try:
        build_model(learnable_vmf_prototypes=True, prototype_ema_momentum=0.9)
        assert False, "Expected ValueError for learnable_vmf_prototypes + prototype_ema_momentum"
    except ValueError:
        pass


def test_forward_backward_across_ot_modes_and_prototype_options():
    for mode, learnable, ema, sigmoid_weight in itertools.product(
        ("hierarchical_vmf_ot", "hierarchical_vmf", "recursive_ot", "flat_ot", "flat_vmf_ot", "vmf"),
        (False, True),
        (None, 0.9),
        (0.0, 0.2),
    ):
        if learnable and ema is not None:
            continue  # now a ValueError; tested separately above
        model = build_model(
            ot_assignment_mode=mode,
            learnable_vmf_prototypes=learnable,
            prototype_ema_momentum=ema,
            sigmoid_regularization_weight=sigmoid_weight,
            flat_vmf_num_components=4,
        )
        model.set_sigmoid_regularization_progress(current_step=5, rampup_steps=10)
        prototype_tensor = model.flat_prototypes if mode in ("flat_ot", "flat_vmf_ot", "vmf") else model.node_prototypes
        before = prototype_tensor.detach().clone()

        images = torch.randn(8, 3, 4, 4)
        labels = torch.arange(8)
        out = model(images, labels)
        out["loss"].backward()

        assert torch.isfinite(out["loss"])
        assert model.backbone.net[1].weight.grad.abs().sum().item() > 0
        if sigmoid_weight > 0:
            assert "loss_sigmoid_regularization" in out
            assert model.sigmoid_label_embeddings.weight.grad.abs().sum().item() > 0
        else:
            assert "loss_sigmoid_regularization" not in out

        if learnable:
            assert prototype_tensor.grad is not None
            assert prototype_tensor.grad.abs().sum().item() > 0
        else:
            assert getattr(prototype_tensor, "grad", None) is None

        if ema is None:
            assert torch.allclose(before, prototype_tensor.detach())
        else:
            assert not torch.allclose(before, prototype_tensor.detach())


def _two_cluster_embeddings():
    # Two tight clusters around +e1 and -e1 on the sphere.
    torch.manual_seed(1)
    a = F.normalize(torch.tensor([[1.0] + [0.0] * 7]) + 0.05 * torch.randn(8, 8), dim=1)
    b = F.normalize(torch.tensor([[-1.0] + [0.0] * 7]) + 0.05 * torch.randn(8, 8), dim=1)
    return torch.cat([a, b], dim=0)


def test_fit_binary_ot_honors_init_mu_polarity():
    model = build_model(num_classes=16, depth=1, em_iters=2)
    z = _two_cluster_embeddings()
    e1 = torch.tensor([[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7])

    mu_fwd, _ = model._fit_binary_ot(z, init_mu=e1)
    mu_rev, _ = model._fit_binary_ot(z, init_mu=e1.flip(0))

    # Polarity follows the init: child 0 tracks whichever cluster init_mu[0] pointed at.
    assert mu_fwd[0, 0] > 0 and mu_fwd[1, 0] < 0
    assert mu_rev[0, 0] < 0 and mu_rev[1, 0] > 0


def test_fit_binary_ot_falls_back_to_cold_seed_for_degenerate_init():
    model = build_model(num_classes=16, depth=1, em_iters=2)
    z = _two_cluster_embeddings()
    degenerate = F.normalize(torch.ones(2, 8), dim=1)  # identical pair

    mu, hard = model._fit_binary_ot(z, init_mu=degenerate)

    assert not model._is_degenerate_pair(mu)
    assert torch.bincount(hard, minlength=2).tolist() == [8, 8]


def test_tree_warm_start_marks_nodes_and_uses_stored_prototypes():
    model = build_model(num_classes=16, depth=2, tree_warm_start=True, prototype_ema_momentum=0.9)
    z = F.normalize(torch.randn(16, 8), dim=1)

    assert not model.node_fitted.any()  # cold before any build
    prototypes, _, _, masks, _ = model._build_tree_from_embeddings(
        z, fallback_prototypes=model._prepared_node_prototypes()
    )
    assert model.node_fitted.all()

    # Second build warm-starts from the produced prototypes: with identical
    # embeddings the fit must be a fixed point (identical paths and polarity).
    _, paths_a, _, _, _ = model._build_tree_from_embeddings(z, fallback_prototypes=prototypes)
    _, paths_b, _, _, _ = model._build_tree_from_embeddings(z, fallback_prototypes=prototypes)
    assert torch.equal(paths_a, paths_b)


def test_node_stats_accumulate_and_finalize_per_level():
    model = build_model(num_classes=16, depth=2, reseed_min_node_samples=1)
    z = F.normalize(torch.randn(16, 8), dim=1)
    paths = torch.randint(0, 2, (16, 2))
    node_ids = _path_node_ids(paths)
    masks = torch.ones(16, 2, dtype=torch.bool)

    model._forward_embeddings(
        z, paths=paths, node_ids=node_ids, masks=masks,
        prototypes=model._prepared_node_prototypes(),
    )
    # Level 0 = root only: all 16 samples; level 1 = nodes 1 and 2 share 16.
    assert model.node_count[0].item() == 16.0
    assert model.node_count[1].item() + model.node_count[2].item() == 16.0

    stats = model.finalize_node_stats()
    assert "tree_node_acc_overall" in stats
    assert "tree_node_acc_level0" in stats
    assert "tree_node_acc_level1" in stats
    assert stats["tree_nodes_visited"] >= 2.0
    # Accumulators reset after finalize.
    assert model.node_count.sum().item() == 0.0
    assert model.node_correct_sum.sum().item() == 0.0


def test_low_acc_streaks_increment_reset_and_respect_min_samples():
    model = build_model(
        num_classes=16, depth=1,
        reseed_acc_threshold=0.9, reseed_patience=2, reseed_min_node_samples=10,
    )

    # Below min samples: ignored, no streak.
    model.node_correct_sum[0] = 1.0
    model.node_count[0] = 5.0
    model.finalize_node_stats()
    assert model.node_low_acc_streak[0].item() == 0

    # Low accuracy with enough samples: streak increments each finalize.
    for expected in (1, 2):
        model.node_correct_sum[0] = 10.0
        model.node_count[0] = 20.0  # acc 0.5 < 0.9
        stats = model.finalize_node_stats()
        assert model.node_low_acc_streak[0].item() == expected
    assert stats["tree_reseed_candidates"] == 1.0
    assert model.node_last_acc[0].item() == 0.5

    # High accuracy resets the streak.
    model.node_correct_sum[0] = 19.0
    model.node_count[0] = 20.0  # acc 0.95 >= 0.9
    stats = model.finalize_node_stats()
    assert model.node_low_acc_streak[0].item() == 0
    assert stats["tree_reseed_candidates"] == 0.0


def test_select_reseed_indices_patience_order_and_budget():
    streaks = [0, 3, 5, 2, 4]
    last_accs = [0.9, 0.55, 0.50, 0.45, 0.52]
    # patience 3 -> candidates {1, 2, 4}; sorted by acc -> [2, 4, 1]; budget 2.
    assert select_reseed_indices(streaks, last_accs, patience=3, budget=2) == [2, 4]
    assert select_reseed_indices(streaks, last_accs, patience=3, budget=10) == [2, 4, 1]
    assert select_reseed_indices(streaks, last_accs, patience=6, budget=2) == []
    assert select_reseed_indices(streaks, last_accs, patience=3, budget=0) == []


def test_subtree_node_ids_covers_internal_descendants_only():
    # Depth-3 tree: 7 internal nodes (0..6).
    assert sorted(_subtree_node_ids(1, 7)) == [1, 3, 4]
    assert sorted(_subtree_node_ids(0, 7)) == [0, 1, 2, 3, 4, 5, 6]
    assert _subtree_node_ids(3, 7) == [3]  # children 7, 8 are leaves
    # Depth-2 tree: 3 internal nodes.
    assert _subtree_node_ids(1, 3) == [1]


def test_reseed_enabled_requires_threshold_and_warm_start():
    for kwargs in (
        {"reseed_enabled": True, "tree_warm_start": True},  # missing threshold
        {"reseed_enabled": True, "reseed_acc_threshold": 0.65},  # missing warm start
    ):
        try:
            build_model(**kwargs)
            assert False, f"Expected ValueError for {kwargs}"
        except ValueError:
            pass


def test_batch_local_warm_start_requires_persisted_prototypes():
    try:
        build_model(
            batch_self_labeling=True,
            tree_warm_start=True,
            learnable_vmf_prototypes=False,
            prototype_ema_momentum=None,
        )
        assert False, "Expected ValueError for batch-local warm start without persisted prototypes"
    except ValueError:
        pass

    model = build_model(
        batch_self_labeling=False,
        tree_warm_start=True,
        learnable_vmf_prototypes=False,
        prototype_ema_momentum=None,
    )
    assert model.tree_warm_start


def test_select_reseed_nodes_cold_restarts_subtree_and_resets_stats():
    model = build_model(
        num_classes=16, depth=3,
        batch_self_labeling=False, tree_warm_start=True,
        reseed_acc_threshold=0.9, reseed_patience=1, reseed_min_node_samples=1,
        reseed_enabled=True, reseed_budget_fraction=1.0,
    )
    model.node_fitted.fill_(True)

    # Node 1 is persistently unlearnable; node 2 is healthy.
    model.node_correct_sum[1] = 5.0
    model.node_count[1] = 10.0  # acc 0.5
    model.node_correct_sum[2] = 10.0
    model.node_count[2] = 10.0  # acc 1.0
    model.finalize_node_stats()

    reseeded = model.select_reseed_nodes()

    assert reseeded == 1
    # Subtree of node 1 = {1, 3, 4}: cold-restarted, stats reset.
    for nid in (1, 3, 4):
        assert not model.node_fitted[nid]
        assert model.node_low_acc_streak[nid].item() == 0
        assert model.node_last_acc[nid].item() == -1.0
    # Healthy nodes untouched.
    for nid in (0, 2, 5, 6):
        assert model.node_fitted[nid]
    assert model.node_last_acc[2].item() == 1.0


def test_compute_active_depth_staircase():
    # Disabled schedule keeps full depth.
    assert compute_active_depth(epoch=0, depth=6, epochs_per_level=None) == 6
    assert compute_active_depth(epoch=500, depth=6, epochs_per_level=None) == 6
    # Staircase: +1 level every 50 epochs starting from 1, capped at depth.
    assert compute_active_depth(epoch=0, depth=6, epochs_per_level=50) == 1
    assert compute_active_depth(epoch=49, depth=6, epochs_per_level=50) == 1
    assert compute_active_depth(epoch=50, depth=6, epochs_per_level=50) == 2
    assert compute_active_depth(epoch=249, depth=6, epochs_per_level=50) == 5
    assert compute_active_depth(epoch=250, depth=6, epochs_per_level=50) == 6
    assert compute_active_depth(epoch=10000, depth=6, epochs_per_level=50) == 6
    # initial_depth offsets the staircase.
    assert compute_active_depth(epoch=0, depth=6, epochs_per_level=50, initial_depth=3) == 3
    assert compute_active_depth(epoch=50, depth=6, epochs_per_level=50, initial_depth=3) == 4

    for bad_kwargs in (
        {"epochs_per_level": 0},
        {"epochs_per_level": -5},
        {"epochs_per_level": 50, "initial_depth": 0},
        {"epochs_per_level": 50, "initial_depth": 7},
    ):
        try:
            compute_active_depth(epoch=0, depth=6, **bad_kwargs)
            assert False, f"Expected ValueError for {bad_kwargs}"
        except ValueError:
            pass


def test_set_active_depth_validates_range_and_defaults_to_full_depth():
    model = build_model(depth=3)
    assert model.active_depth == 3  # backward compatible default

    model.set_active_depth(1)
    assert model.active_depth == 1
    model.set_active_depth(3)
    assert model.active_depth == 3

    for bad in (0, 4, -1):
        try:
            model.set_active_depth(bad)
            assert False, f"Expected ValueError for active_depth={bad}"
        except ValueError:
            pass


def test_active_depth_clamps_supervision_to_active_levels():
    # Clamping the loss loop to active_depth must be exactly equivalent to
    # masking out the inactive levels.
    torch.manual_seed(0)
    model = build_model(num_classes=16, depth=3)
    z = F.normalize(torch.randn(16, 8), dim=1)
    prototypes = model._prepared_node_prototypes()
    paths = torch.randint(0, 2, (16, 3))
    node_ids = _path_node_ids(paths)
    masks = torch.ones(16, 3, dtype=torch.bool)

    model.set_active_depth(1)
    clamped = model._forward_embeddings(
        z, paths=paths, node_ids=node_ids, masks=masks, prototypes=prototypes
    )

    masked = masks.clone()
    masked[:, 1:] = False
    model.set_active_depth(3)
    reference = model._forward_embeddings(
        z, paths=paths, node_ids=node_ids, masks=masked, prototypes=prototypes
    )

    assert torch.allclose(clamped["loss"], reference["loss"])
    assert torch.allclose(clamped["acc"], reference["acc"])
    assert torch.allclose(clamped["acc_branch"], reference["acc_branch"])
    assert clamped["active_depth"] == 1.0

    # Full active depth reproduces the unrestricted loss.
    model.set_active_depth(3)
    full = model._forward_embeddings(
        z, paths=paths, node_ids=node_ids, masks=masks, prototypes=prototypes
    )
    assert not torch.allclose(full["loss"], clamped["loss"])  # deeper levels add loss terms


def test_forward_backward_respects_active_depth_in_batch_self_labeling():
    model = build_model(num_classes=16, depth=3, batch_self_labeling=True)
    model.set_active_depth(1)
    images = torch.randn(16, 3, 4, 4)
    labels = torch.arange(16)

    out = model(images, labels)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    assert out["active_depth"] == 1.0
    assert model.backbone.net[1].weight.grad.abs().sum().item() > 0


def test_full_refresh_paths_work_for_tree_and_flat_modes():
    for mode in ("hierarchical_vmf_ot", "hierarchical_vmf", "recursive_ot", "flat_ot", "flat_vmf_ot", "vmf"):
        model = build_model(
            num_classes=16,
            depth=3,
            ot_assignment_mode=mode,
            flat_vmf_num_components=4,
            batch_self_labeling=False,
            learnable_vmf_prototypes=True,
            prototype_ema_momentum=None,
            sigmoid_regularization_weight=0.0,
        )
        stats = model.refresh_assignments(torch.randn(16, 8))
        images = torch.randn(4, 3, 4, 4)
        labels = torch.arange(4)
        out = model(images, labels)
        out["loss"].backward()

        assert stats
        assert torch.isfinite(out["loss"])
        assert model.backbone.net[1].weight.grad.abs().sum().item() > 0
