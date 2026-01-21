import re

import jax.numpy as jnp

import prism_vm as pv


def _eval_baseline(expr):
    vm = pv.PrismVM()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    ptr = vm.parse(tokens)
    res_ptr = vm.eval(ptr)
    return vm.decode(res_ptr)


def _eval_bsp_candidates(expr, max_steps=32):
    vm = pv.PrismVM_BSP()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    root_ptr = vm.parse(tokens)
    frontier = jnp.array([root_ptr], dtype=jnp.int32)
    last = None
    for _ in range(max_steps):
        vm.ledger, next_frontier = pv.cycle_candidates(vm.ledger, frontier)
        if int(next_frontier.shape[0]) == 0:
            return vm.decode(int(frontier[0]))
        curr = int(next_frontier[0])
        if curr == last:
            return vm.decode(curr)
        last = curr
        frontier = next_frontier
    raise AssertionError("Candidate BSP evaluation did not converge within max_steps")


def test_bsp_candidates_matches_baseline_add_zero():
    expr = "(add zero (suc zero))"
    assert _eval_bsp_candidates(expr) == _eval_baseline(expr)


def test_bsp_candidates_matches_baseline_mul_zero():
    expr = "(mul zero (suc zero))"
    assert _eval_bsp_candidates(expr) == _eval_baseline(expr)
