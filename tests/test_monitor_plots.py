import os
import tempfile

from tools.monitor_plots import (
    append_history,
    save_training_monitor_svg,
    save_tree_health_monitor_svg,
    save_tree_structure_monitor_svg,
)


def _fake_history(num_epochs=5):
    history = {}
    for epoch in range(num_epochs):
        scalars = {
            "epoch": epoch,
            "lr": 0.06 * (1 - epoch / num_epochs),
            "loss": 1.0 / (epoch + 1),
            "loss_hierarchical": 0.8 / (epoch + 1),
            "accuracy": 30.0 + 5 * epoch,
            "acc": 0.5 + 0.05 * epoch,
            "acc_branch": 0.6 + 0.05 * epoch,
            "active_depth": min(1 + epoch // 2, 3),
            "tree_node_acc_overall": 0.55 + 0.05 * epoch,
            "tree_node_acc_level0": 0.7 + 0.04 * epoch,
            "tree_node_acc_level1": 0.5 + 0.05 * epoch,
            "tree_nodes_visited": 3.0,
            "tree_reseed_candidates": max(0, 2 - epoch),
            "tree_reseeded_nodes": 1.0 if epoch == 3 else 0.0,
            "tree_nonempty_leaves": 8.0,
            "tree_min_leaf_count": 5.0,
            "tree_max_leaf_count": 9.0,
            "tree_purity_level0": 0.55 + 0.05 * epoch,
            "tree_purity_level1": 0.6 + 0.05 * epoch,
            "tree_nmi_level0": 0.2 + 0.05 * epoch,
            "tree_nmi_level1": 0.25 + 0.05 * epoch,
            "ot_unbalanced_tau": 5.0 * (1 - epoch / num_epochs) + 0.02,
            "tree_min_split_fraction": 0.5 - 0.05 * epoch,
            "tree_mean_split_fraction": 0.5 - 0.02 * epoch,
        }
        if epoch == 2:
            scalars.pop("tree_node_acc_level1")  # simulate a missing series
        append_history(history, scalars)
    return history


def test_append_history_aligns_missing_keys():
    history = {}
    append_history(history, {"epoch": 0, "a": 1.0})
    append_history(history, {"epoch": 1, "b": 2.0})

    assert history["a"] == [1.0, None]
    assert history["b"] == [None, 2.0]
    assert history["epoch"] == [0, 1]


def _assert_svg(path):
    assert os.path.exists(path)
    with open(path) as f:
        head = f.read(512)
    assert "<svg" in head


def test_monitor_svgs_are_written():
    history = _fake_history()
    with tempfile.TemporaryDirectory() as tmp:
        training_path = os.path.join(tmp, "monitor_training.svg")
        tree_path = os.path.join(tmp, "monitor_tree_health.svg")
        structure_path = os.path.join(tmp, "monitor_tree_structure.svg")
        save_training_monitor_svg(history, training_path)
        save_tree_health_monitor_svg(history, tree_path)
        save_tree_structure_monitor_svg(history, structure_path)
        _assert_svg(training_path)
        _assert_svg(tree_path)
        _assert_svg(structure_path)


def test_monitor_svgs_tolerate_empty_history():
    with tempfile.TemporaryDirectory() as tmp:
        training_path = os.path.join(tmp, "monitor_training.svg")
        tree_path = os.path.join(tmp, "monitor_tree_health.svg")
        structure_path = os.path.join(tmp, "monitor_tree_structure.svg")
        save_training_monitor_svg({}, training_path)
        save_tree_health_monitor_svg({}, tree_path)
        save_tree_structure_monitor_svg({}, structure_path)
        _assert_svg(training_path)
        _assert_svg(tree_path)
        _assert_svg(structure_path)
