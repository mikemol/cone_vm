import random

import pytest

from tests import harness
import prism_vm as pv

pytestmark = pytest.mark.m4


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


def _run_arena(expr, steps, do_sort, use_morton):
    vm = pv.PrismVM_BSP_Legacy()
    root_ptr = vm.parse(harness.tokenize(expr))
    arena = vm.arena
    for _ in range(steps):
        arena, root_ptr = pv.cycle(
            arena, root_ptr, do_sort=do_sort, use_morton=use_morton
        )
    vm.arena = arena
    return vm.decode(int(root_ptr))


def test_arena_denotation_invariance_random_suite():
    rng = random.Random(1)
    for _ in range(10):
        expr = _rand_expr(rng, 4)
        no_sort = _run_arena(expr, 4, do_sort=False, use_morton=False)
        rank_sort = _run_arena(expr, 4, do_sort=True, use_morton=False)
        morton_sort = _run_arena(expr, 4, do_sort=True, use_morton=True)
        assert no_sort == rank_sort
        assert no_sort == morton_sort


def test_arena_decode_hides_ids_by_default():
    vm = pv.PrismVM_BSP_Legacy()
    root_ptr = vm.parse(harness.tokenize("(mul zero zero)"))
    no_ids = vm.decode(int(root_ptr))
    with_ids = vm.decode(int(root_ptr), show_ids=True)
    assert no_ids == "<mul>"
    assert with_ids.startswith("<mul:")
    assert with_ids.endswith(">")
