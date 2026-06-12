"""Epoch-level monitoring plots saved as SVG (torch-free).

Three figures, refreshed once per epoch by the trainer:

- ``monitor_training.svg``: learning rate, loss components, kNN accuracy.
- ``monitor_tree_health.svg``: pseudo-label-tree health — per-level and
  overall node branch accuracy (against the 0.5 chance line), sample-level
  path/branch accuracy, depth-annealing progress, re-seeding activity, and
  leaf balance from the refresh stats. One panel per concern so a sick level
  is visible at a glance.
- ``monitor_tree_structure.svg``: per-level purity and NMI of predicted tree
  path prefixes against ground-truth labels (only written when the
  diagnostics produce ``tree_purity_level*`` keys). Standalone so trees with
  many levels stay readable.
"""

import matplotlib

try:
    matplotlib.use("Agg")
except Exception:  # pragma: no cover - backend already fixed by the host app
    pass

from matplotlib import pyplot as plt


def append_history(history, scalars):
    """Append one epoch of scalars to history, aligning all series.

    ``history`` maps key -> list with one entry per epoch (None where a key
    was absent that epoch), so series of different lifetimes stay aligned.
    """
    length = len(history.get("epoch", []))
    for key in set(history) | set(scalars):
        series = history.setdefault(key, [None] * length)
        series.append(scalars.get(key))


def _series(history, key):
    xs, ys = [], []
    for x, y in zip(history.get("epoch", []), history.get(key, [])):
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _plot_keys(ax, history, keys, title, ylabel=None):
    plotted = False
    for key in keys:
        xs, ys = _series(history, key)
        if xs:
            ax.plot(xs, ys, label=key, linewidth=1.2)
            plotted = True
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if plotted:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes, alpha=0.5)
    return plotted


def save_training_monitor_svg(history, path):
    """Learning rate, losses, and kNN accuracy vs epoch."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    _plot_keys(axes[0], history, ["lr"], "learning rate")
    loss_keys = sorted(key for key in history if key == "loss" or key.startswith("loss_"))
    _plot_keys(axes[1], history, loss_keys, "losses")
    _plot_keys(axes[2], history, ["accuracy"], "kNN accuracy")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _level_keys(history, prefix):
    keys = [key for key in history if key.startswith(prefix)]
    return sorted(keys, key=lambda key: int(key.rsplit("level", 1)[1]))


def _level_acc_keys(history):
    return _level_keys(history, "tree_node_acc_level")


def save_tree_health_monitor_svg(history, path):
    """Pseudo-label-tree health: per-level + overall, one panel per concern."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Per-level node branch accuracy. Healthy levels rise away from the 0.5
    # chance line; an augmentation-unstable level stays pinned to it.
    top_left = axes[0][0]
    plotted = _plot_keys(
        top_left, history,
        _level_acc_keys(history) + ["tree_node_acc_overall"],
        "node branch accuracy per level (mean over nodes)",
    )
    if plotted:
        top_left.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="chance (0.5)")
        top_left.legend(fontsize=7)

    # Sample-level training accuracies plus node split fractions (all in
    # [0, 1]; split fraction 0.5 = balanced, lower = natural ratios emerging
    # under unbalanced OT).
    _plot_keys(
        axes[0][1], history,
        ["acc", "acc_branch", "tree_min_split_fraction", "tree_mean_split_fraction"],
        "sample accuracy + node split fractions",
    )

    # Depth annealing progress, re-seeding activity, unbalanced-OT tau.
    _plot_keys(
        axes[1][0], history,
        ["active_depth", "tree_reseed_candidates", "tree_reseeded_nodes", "ot_unbalanced_tau"],
        "depth annealing + re-seeding + unbalanced-OT tau",
    )

    # Leaf balance from the tree refresh stats.
    _plot_keys(
        axes[1][1], history,
        ["tree_nonempty_leaves", "tree_min_leaf_count", "tree_max_leaf_count", "tree_nodes_visited"],
        "tree occupancy / leaf balance",
    )

    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_tree_structure_monitor_svg(history, path):
    """Tree-vs-ground-truth structure: per-level purity and NMI of predicted
    path prefixes against true labels (diagnostic only; see
    tools/tree_metrics.py and main.maybe_compute_tree_structure_metrics).

    Standalone SVG with one full-width panel per metric so trees with many
    levels stay readable. Purity rises mechanically with cluster count, so
    compare a level mostly against its own history; NMI is the fairer
    cross-level number.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    _plot_keys(
        axes[0], history,
        _level_keys(history, "tree_purity_level"),
        "cluster purity per level (vs true labels)",
    )
    _plot_keys(
        axes[1], history,
        _level_keys(history, "tree_nmi_level"),
        "cluster NMI per level (vs true labels)",
    )

    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
