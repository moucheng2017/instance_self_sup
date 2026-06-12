"""HiRef-style rank-annealing schedule and mixed-radix tree indexing.

Ports the optimal rank-annealing schedule of Halmos et al., "Hierarchical
Refinement: Optimal Transport to Infinity and Beyond" (ICML 2025), Section 3.3
and Appendix E.1, plus the mixed-radix index arithmetic used to address a tree
whose branching factor varies per level.

A rank schedule ``(r_1, ..., r_kappa)`` factorizes a target leaf count
``n = prod(r_t)``. Level ``t`` (0-indexed) holds ``rho_t = prod(r_s for s < t)``
nodes (``rho_0 = 1``) and refines each node into ``r_t`` children. The DP picks
the feasible schedule minimizing the sum of partial products
``sum_t prod_{s<=t} r_s`` (proportional to the number of low-rank OT calls),
subject to ``r_t <= max_rank``.

This module is pure-Python (no torch) so the schedule logic is unit-testable in
isolation from the model.
"""

from __future__ import annotations

from math import inf, prod
from typing import List, Sequence


def partial_product_sum(schedule: Sequence[int]) -> int:
    """Sum of partial products ``sum_j prod_{i<=j} r_i`` (the HiRef objective)."""
    total = 0
    running = 1
    for r in schedule:
        running *= int(r)
        total += running
    return total


def optimal_rank_schedule(
    n: int,
    depth: int,
    max_rank: int,
    base_rank: int = 1,
) -> List[int]:
    """Return the optimal rank-annealing schedule of length ``depth``.

    Minimizes ``partial_product_sum`` over integer schedules ``(r_1, ..., r_depth)``
    with each ``2 <= r_t <= max_rank`` and ``prod(r_t) == n / base_rank``.

    ``base_rank`` (HiRef's ``Q``) is the maximum block size tolerated at the leaf
    level; when ``Q != 1`` the annealed ranks only need to factor ``n / Q``
    (HiRef E.1). Raises ``ValueError`` if no feasible schedule exists.
    """
    n = int(n)
    depth = int(depth)
    max_rank = int(max_rank)
    base_rank = int(base_rank)
    if depth < 1:
        raise ValueError("depth must be at least 1.")
    if max_rank < 2:
        raise ValueError("max_rank must be at least 2.")
    if base_rank < 1:
        raise ValueError("base_rank must be at least 1.")
    if n % base_rank != 0:
        raise ValueError(f"base_rank={base_rank} must divide n={n}.")
    target = n // base_rank
    if target < 2:
        raise ValueError("n / base_rank must be at least 2 to build a tree.")

    # dp[k] maps an achievable product m (using exactly k factors, each in
    # [2, max_rank]) to (min_partial_product_sum, schedule_list).
    dp = [{} for _ in range(depth + 1)]
    dp[0][1] = (0, [])
    for k in range(1, depth + 1):
        layer = dp[k]
        for m_prev, (cost_prev, sched_prev) in dp[k - 1].items():
            for r in range(2, max_rank + 1):
                m = m_prev * r
                if m > target or target % m != 0:
                    continue
                cost = cost_prev + m  # m is the new partial product rho_k
                best = layer.get(m)
                if best is None or cost < best[0]:
                    layer[m] = (cost, sched_prev + [r])

    if target not in dp[depth]:
        raise ValueError(
            f"No rank schedule of depth {depth} with ranks in [2, {max_rank}] "
            f"factorizes n/base_rank = {target} (n={n}, base_rank={base_rank})."
        )
    return list(dp[depth][target][1])


def level_sizes(schedule: Sequence[int]) -> List[int]:
    """Number of nodes at each level: ``rho_t = prod(r_s for s < t)``."""
    sizes = []
    running = 1
    for r in schedule:
        sizes.append(running)
        running *= int(r)
    return sizes


def level_offsets(schedule: Sequence[int]) -> List[int]:
    """Global-id offset of each level (cumulative node count before the level)."""
    offsets = []
    running = 0
    for size in level_sizes(schedule):
        offsets.append(running)
        running += size
    return offsets


def num_internal_nodes(schedule: Sequence[int]) -> int:
    """Total internal nodes across levels ``0 .. depth-1``."""
    return sum(level_sizes(schedule))


def child_global_id(
    schedule: Sequence[int], level: int, local_index: int, child: int
) -> int:
    """Global id of the ``child``-th child of node ``local_index`` at ``level``.

    Children live at ``level + 1``; the child's local index is the mixed-radix
    digit append ``local_index * r_{level} + child``.
    """
    r = int(schedule[level])
    if not 0 <= child < r:
        raise ValueError(f"child must be in [0, {r}) at level {level}.")
    child_local = local_index * r + child
    return level_offsets(schedule)[level + 1] + child_local


def leaf_index(schedule: Sequence[int], path: Sequence[int]) -> int:
    """Mixed-radix encode a root-to-leaf ``path`` into a flat leaf id."""
    if len(path) != len(schedule):
        raise ValueError("path length must equal schedule length (tree depth).")
    idx = 0
    for r, digit in zip(schedule, path):
        idx = idx * int(r) + int(digit)
    return idx
