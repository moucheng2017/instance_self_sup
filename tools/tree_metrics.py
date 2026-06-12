"""Tree-vs-ground-truth structure diagnostics (torch-free).

Measures whether the discovered pseudo-label tree aligns with real semantic
structure: per-level purity and NMI between tree path *prefixes* and
ground-truth class labels. Level numbering matches the node-accuracy stats:
level 0 = the root decision (2 clusters), level l = 2^(l+1) prefix clusters.

These metrics are diagnostic only (they peek at labels); nothing here feeds
back into training.
"""

import math
from collections import Counter, defaultdict


def prefix_cluster_ids(paths, level, radices=None):
    """Integer cluster id from the first ``level + 1`` branch decisions.

    ``radices`` gives the per-level branching factors (a K-way rank schedule);
    None keeps the legacy binary encoding (radix 2 at every level).
    """
    ids = []
    for path in paths:
        value = 0
        for t, branch in enumerate(path[: level + 1]):
            radix = 2 if radices is None else int(radices[t])
            value = value * radix + int(branch)
        ids.append(value)
    return ids


def purity(clusters, labels):
    """Fraction of samples whose cluster's majority label is their label."""
    if len(labels) == 0:
        return 0.0
    per_cluster = defaultdict(Counter)
    for cluster, label in zip(clusters, labels):
        per_cluster[cluster][label] += 1
    return sum(max(counter.values()) for counter in per_cluster.values()) / len(labels)


def _entropy(counts, total):
    h = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            h -= p * math.log(p)
    return h


def nmi(clusters, labels):
    """Normalized mutual information (arithmetic-mean normalization,
    matching sklearn's default). Returns 0.0 when either side is constant."""
    total = len(labels)
    if total == 0:
        return 0.0
    joint = Counter(zip(clusters, labels))
    cluster_counts = Counter(clusters)
    label_counts = Counter(labels)

    mi = 0.0
    for (cluster, label), count in joint.items():
        p_joint = count / total
        p_cluster = cluster_counts[cluster] / total
        p_label = label_counts[label] / total
        mi += p_joint * math.log(p_joint / (p_cluster * p_label))

    h_cluster = _entropy(cluster_counts.values(), total)
    h_label = _entropy(label_counts.values(), total)
    if h_cluster <= 0.0 or h_label <= 0.0:
        return 0.0
    return max(0.0, mi / ((h_cluster + h_label) / 2.0))


def prefix_label_metrics(paths, labels, radices=None):
    """Per-level purity and NMI between path prefixes and true labels.

    paths: sequence of equal-length branch sequences (one per sample). Binary
    by default; pass ``radices`` (per-level branching factors, e.g. a K-way
    rank schedule) for mixed-radix paths.
    labels: sequence of hashable ground-truth labels, same length as paths.
    Returns {} when inputs are unusable.
    """
    if not paths or labels is None or len(labels) != len(paths):
        return {}
    depth = len(paths[0])
    if radices is not None and len(radices) != depth:
        return {}
    metrics = {}
    for level in range(depth):
        ids = prefix_cluster_ids(paths, level, radices=radices)
        metrics[f"tree_purity_level{level}"] = round(purity(ids, labels), 4)
        metrics[f"tree_nmi_level{level}"] = round(nmi(ids, labels), 4)
    return metrics


def source_pool_true_labels(dataset, source_indices):
    """Ground-truth labels for each source-pool position, or None.

    Unwraps Subset-like wrappers (objects with .dataset / .indices) until a
    dataset exposing .targets or .labels is found. Diagnostic helper: returns
    None rather than raising when labels are unavailable.
    """
    try:
        indices = [int(i) for i in source_indices]
    except (TypeError, ValueError):
        return None
    while not (hasattr(dataset, "targets") or hasattr(dataset, "labels")):
        inner_indices = getattr(dataset, "indices", None)
        inner = getattr(dataset, "dataset", None)
        if inner is None:
            return None
        if inner_indices is not None:
            try:
                indices = [int(inner_indices[i]) for i in indices]
            except (TypeError, ValueError, IndexError):
                return None
        dataset = inner
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        return None
    try:
        return [int(targets[i]) for i in indices]
    except (TypeError, ValueError, IndexError):
        return None
