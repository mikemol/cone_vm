import re

import jax.numpy as jnp
import prism_vm as pv


def _peano_expr(n):
    expr = "zero"
    for _ in range(n):
        expr = f"(suc {expr})"
    return expr


def _count_suc(expr):
    return expr.count("suc")


def _eval_baseline(expr):
    vm = pv.PrismVM()
    tokens = re.findall(r"\(|\)|[a-z]+", expr)
    ptr = vm.parse(tokens)
    res_ptr = vm.eval(ptr)
    return vm.decode(res_ptr)


def _eval_bsp(expr, max_steps=128):
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


def test_small_add_mul_baseline_vs_bsp():
    for a in range(4):
        for b in range(4):
            add_expr = f"(add {_peano_expr(a)} {_peano_expr(b)})"
            mul_expr = f"(mul {_peano_expr(a)} {_peano_expr(b)})"

            add_base = _eval_baseline(add_expr)
            add_bsp = _eval_bsp(add_expr)
            assert _count_suc(add_base) == a + b
            assert _count_suc(add_bsp) == a + b
            assert add_base == add_bsp

            mul_base = _eval_baseline(mul_expr)
            mul_bsp = _eval_bsp(mul_expr)
            assert _count_suc(mul_base) == a * b
            assert _count_suc(mul_bsp) == a * b
            assert mul_base == mul_bsp
