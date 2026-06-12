import math
from itertools import product

import pytest

from tools.rank_annealing import (
    child_global_id,
    leaf_index,
    level_offsets,
    level_sizes,
    num_internal_nodes,
    optimal_rank_schedule,
    partial_product_sum,
)


def _brute_force_min_schedule(n, depth, max_rank):
    """Reference: smallest sum-of-partial-products factorization of n."""
    best = None
    best_sum = math.inf
    for combo in product(range(2, max_rank + 1), repeat=depth):
        prod = 1
        for r in combo:
            prod *= r
        if prod != n:
            continue
        s = partial_product_sum(combo)
        if s < best_sum:
            best_sum = s
            best = combo
    return list(best) if best is not None else None


def test_schedule_factorizes_n_and_respects_max_rank():
    schedule = optimal_rank_schedule(n=256, depth=4, max_rank=16)
    assert math.prod(schedule) == 256
    assert all(2 <= r <= 16 for r in schedule)
    assert len(schedule) == 4


def test_schedule_minimizes_partial_product_sum():
    for n, depth, max_rank in [(256, 4, 16), (64, 3, 8), (216, 3, 16), (128, 4, 8)]:
        schedule = optimal_rank_schedule(n=n, depth=depth, max_rank=max_rank)
        brute = _brute_force_min_schedule(n, depth, max_rank)
        assert brute is not None
        assert partial_product_sum(schedule) == partial_product_sum(brute)
        assert math.prod(schedule) == n


def test_schedule_raises_when_infeasible():
    # 250 = 2 * 5^3; cannot be split into 2 factors each <= 16.
    with pytest.raises(ValueError):
        optimal_rank_schedule(n=250, depth=2, max_rank=16)


def test_base_rank_reduces_problem():
    # With base rank Q the leaf block is allowed to hold up to Q points, so the
    # product of the annealed ranks only needs to reach n / Q.
    schedule = optimal_rank_schedule(n=512, depth=2, max_rank=64, base_rank=8)
    assert math.prod(schedule) == 64  # 512 / 8


def test_level_sizes_and_offsets_are_partial_products():
    schedule = [2, 3, 4]
    assert level_sizes(schedule) == [1, 2, 6]  # rho_0..rho_{k-1}
    assert level_offsets(schedule) == [0, 1, 3]
    assert num_internal_nodes(schedule) == 1 + 2 + 6


def test_child_global_id_matches_binary_heap():
    schedule = [2, 2, 2]
    # Binary heap: children of node j at level t are 2j+1, 2j+2.
    assert child_global_id(schedule, level=0, local_index=0, child=0) == 1
    assert child_global_id(schedule, level=0, local_index=0, child=1) == 2
    assert child_global_id(schedule, level=1, local_index=0, child=0) == 3
    assert child_global_id(schedule, level=1, local_index=1, child=1) == 6


def test_child_global_id_kway():
    schedule = [3, 2]
    # Level 0: one node (global 0), splits into 3 -> locals 0,1,2 at level 1.
    offsets = level_offsets(schedule)
    assert offsets == [0, 1]
    for child in range(3):
        assert child_global_id(schedule, 0, 0, child) == 1 + child


def test_leaf_index_mixed_radix():
    schedule = [2, 3, 4]
    # path (1, 2, 3) -> ((1*3)+2)*4 + 3 = 23
    assert leaf_index(schedule, [1, 2, 3]) == 23
    assert leaf_index(schedule, [0, 0, 0]) == 0
    assert leaf_index(schedule, [1, 2, 3]) < math.prod(schedule)
