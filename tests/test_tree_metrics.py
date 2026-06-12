import math

from tools.tree_metrics import (
    nmi,
    prefix_cluster_ids,
    prefix_label_metrics,
    purity,
    source_pool_true_labels,
)


def test_prefix_cluster_ids_levels():
    paths = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert prefix_cluster_ids(paths, 0) == [0, 0, 1, 1]
    assert prefix_cluster_ids(paths, 1) == [0, 1, 2, 3]


def test_prefix_cluster_ids_mixed_radix():
    # radices [3, 2]: prefix id at level 1 is digit0 * 2 + digit1.
    paths = [[0, 0], [0, 1], [2, 1], [1, 0]]
    assert prefix_cluster_ids(paths, 0, radices=[3, 2]) == [0, 0, 2, 1]
    assert prefix_cluster_ids(paths, 1, radices=[3, 2]) == [0, 1, 5, 2]
    # radices [2, 2] matches the legacy binary encoding exactly.
    binary_paths = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert prefix_cluster_ids(binary_paths, 1, radices=[2, 2]) == prefix_cluster_ids(binary_paths, 1)


def test_prefix_label_metrics_mixed_radix():
    # 3-way root split separates 3 labels perfectly.
    paths = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    labels = ["a", "a", "b", "b", "c", "c"]
    metrics = prefix_label_metrics(paths, labels, radices=[3, 2])
    assert metrics["tree_purity_level0"] == 1.0
    assert metrics["tree_nmi_level0"] == 1.0
    assert metrics["tree_nmi_level1"] < 1.0
    # Mismatched radices length -> unusable -> {}
    assert prefix_label_metrics(paths, labels, radices=[3]) == {}


def test_purity_perfect_and_mixed():
    assert purity([0, 0, 1, 1], ["a", "a", "b", "b"]) == 1.0
    # cluster 0: 3 a + 1 b -> majority 3; cluster 1: 2 b -> majority 2; (3+2)/6
    assert abs(purity([0, 0, 0, 0, 1, 1], ["a", "a", "a", "b", "b", "b"]) - 5 / 6) < 1e-9
    assert purity([], []) == 0.0


def test_nmi_perfect_independent_and_constant():
    assert abs(nmi([0, 0, 1, 1], ["a", "a", "b", "b"]) - 1.0) < 1e-9
    # Independent: every (cluster, label) cell equally filled -> MI = 0.
    clusters = [0, 0, 1, 1]
    labels = ["a", "b", "a", "b"]
    assert abs(nmi(clusters, labels)) < 1e-9
    # Constant clustering carries no information.
    assert nmi([0, 0, 0, 0], ["a", "a", "b", "b"]) == 0.0
    assert nmi([], []) == 0.0


def test_nmi_hand_computed_value():
    # clusters: [0,0,0,1], labels: [a,a,b,b]
    # joint: (0,a)=2/4, (0,b)=1/4, (1,b)=1/4
    clusters = [0, 0, 0, 1]
    labels = ["a", "a", "b", "b"]
    p = 1 / 4
    mi = (
        2 * p * math.log((2 * p) / ((3 / 4) * (2 / 4)))
        + p * math.log(p / ((3 / 4) * (2 / 4)))
        + p * math.log(p / ((1 / 4) * (2 / 4)))
    )
    h_c = -(3 / 4) * math.log(3 / 4) - (1 / 4) * math.log(1 / 4)
    h_l = math.log(2)
    expected = mi / ((h_c + h_l) / 2)
    assert abs(nmi(clusters, labels) - expected) < 1e-9


def test_prefix_label_metrics_per_level_keys_and_alignment():
    # Level 0 separates labels perfectly; level 1 over-splits (purity stays 1,
    # NMI drops below 1 because clusters carry extra information).
    paths = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = ["a", "a", "b", "b"]
    metrics = prefix_label_metrics(paths, labels)
    assert metrics["tree_purity_level0"] == 1.0
    assert metrics["tree_nmi_level0"] == 1.0
    assert metrics["tree_purity_level1"] == 1.0
    assert metrics["tree_nmi_level1"] < 1.0
    assert set(metrics) == {
        "tree_purity_level0", "tree_nmi_level0",
        "tree_purity_level1", "tree_nmi_level1",
    }
    # Unusable inputs -> {}
    assert prefix_label_metrics([], labels) == {}
    assert prefix_label_metrics(paths, None) == {}
    assert prefix_label_metrics(paths, ["a"]) == {}


class _FakeCifar:
    def __init__(self):
        self.targets = [10, 11, 12, 13, 14]


class _FakeSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices


def test_source_pool_true_labels_direct_and_subset():
    base = _FakeCifar()
    assert source_pool_true_labels(base, [0, 2, 4]) == [10, 12, 14]
    # Subset wrapper: pool position -> subset index -> base index.
    subset = _FakeSubset(base, [4, 3, 2, 1, 0])
    assert source_pool_true_labels(subset, [0, 1]) == [14, 13]
    # No labels anywhere -> None.
    assert source_pool_true_labels(object(), [0]) is None
