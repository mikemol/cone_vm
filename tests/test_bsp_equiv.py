import re

import jax.numpy as jnp
import pytest

import prism_vm as pv


def _eval_baseline(expr):
    vm = pv.PrismVM()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    ptr = vm.parse(tokens)
    res_ptr = vm.eval(ptr)
    return vm.decode(res_ptr)


def _eval_bsp(expr, max_steps=64):
    vm = pv.PrismVM_BSP()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    root_ptr = vm.parse(tokens)
    frontier = jnp.array([root_ptr], dtype=jnp.int32)
    last = None
    for _ in range(max_steps):
        vm.ledger, frontier = pv.cycle_intrinsic(vm.ledger, frontier)
        curr = int(frontier[0])
        if curr == last:
            return vm.decode(curr)
        last = curr
    raise AssertionError("BSP evaluation did not converge within max_steps")


@pytest.mark.xfail(
    reason="BSP ledger lacks continuation/strata semantics; add+suc not yet equivalent",
    strict=False,
)
def test_bsp_matches_baseline_add():
    expr = "(add (suc zero) (suc zero))"
    assert _eval_bsp(expr) == _eval_baseline(expr)


def test_bsp_matches_baseline_add_zero():
    expr = "(add zero (suc (suc zero)))"
    assert _eval_bsp(expr) == _eval_baseline(expr)
