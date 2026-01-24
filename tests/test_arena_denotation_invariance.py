import random

import pytest

from tests import harness
import prism_vm as pv

pytestmark = pytest.mark.m3


def _rand_expr(rng, depth):
    if depth <= 0:
        return "zero"
    roll = rng.random()
    if roll < 0.35:
        return "zero"
    if roll < 0.7:
        return f"(suc {_rand_expr(rng, depth - 1)})"
    op = "add" if rng.random() < 0.7 else "mul"
    return f"({op} {_rand_expr(rng, depth - 1)} {_rand_expr(rng, depth - 1)})"


def test_arena_denotation_invariance_random_suite():
    rng = random.Random(1)
    for _ in range(10):
        expr = _rand_expr(rng, 4)
        no_sort = harness.run_arena(expr, steps=4, do_sort=False, use_morton=False)
        rank_sort = harness.run_arena(expr, steps=4, do_sort=True, use_morton=False)
        morton_sort = harness.run_arena(expr, steps=4, do_sort=True, use_morton=True)
        assert no_sort == rank_sort
        assert no_sort == morton_sort


def test_arena_denotation_invariance_blocked_small_suite():
    size = int(pv.init_arena().rank.shape[0])
    block_size = 3
    assert size % block_size == 0
    cases = [
        "zero",
        "(suc zero)",
        "(add (suc zero) (suc zero))",
        "(mul (suc (suc zero)) (suc zero))",
    ]
    for expr in cases:
        no_sort = harness.run_arena(expr, steps=4, do_sort=False, use_morton=False)
        rank_sort = harness.run_arena(expr, steps=4, do_sort=True, use_morton=False)
        morton_sort = harness.run_arena(expr, steps=4, do_sort=True, use_morton=True)
        blocked = harness.run_arena(expr, steps=4, do_sort=True, block_size=block_size)
        assert no_sort == rank_sort
        assert no_sort == morton_sort
        assert no_sort == blocked


def test_arena_decode_hides_ids_by_default():
    vm = pv.PrismVM_BSP_Legacy()
    root_ptr = vm.parse(harness.tokenize("(mul zero zero)"))
    no_ids = vm.decode(root_ptr)
    with_ids = vm.decode(root_ptr, show_ids=True)
    assert no_ids == "<mul>"
    assert with_ids.startswith("<mul:")
    assert with_ids.endswith(">")
