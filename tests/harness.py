import re

import jax.numpy as jnp

import prism_vm as pv

TOKEN_RE = re.compile(r"\(|\)|[a-z]+")


def tokenize(expr):
    return TOKEN_RE.findall(expr)


def parse_expr(vm, expr):
    tokens = tokenize(expr)
    return vm.parse(tokens)


def denote_baseline(expr, vm=None):
    vm = vm or pv.PrismVM()
    ptr = parse_expr(vm, expr)
    res_ptr = vm.eval(ptr)
    return vm, int(res_ptr)


def pretty_baseline(expr, vm=None):
    vm, ptr = denote_baseline(expr, vm=vm)
    return vm.decode(ptr)


def _run_intrinsic(ledger, frontier, max_steps):
    last = None
    for _ in range(max_steps):
        ledger, frontier = pv.cycle_intrinsic(ledger, frontier)
        curr = int(frontier[0])
        if curr == last:
            return ledger, frontier
        last = curr
    raise AssertionError("BSP intrinsic evaluation did not converge within max_steps")


def _run_candidates(ledger, frontier, max_steps, validate_stratum=False):
    last = None
    for _ in range(max_steps):
        ledger, next_frontier, _ = pv.cycle_candidates(
            ledger, frontier, validate_stratum=validate_stratum
        )
        if int(next_frontier.shape[0]) == 0:
            return ledger, frontier
        curr = int(next_frontier[0])
        if curr == last:
            return ledger, next_frontier
        last = curr
        frontier = next_frontier
    raise AssertionError("BSP candidate evaluation did not converge within max_steps")


def denote_bsp_intrinsic(expr, max_steps=64, vm=None):
    vm = vm or pv.PrismVM_BSP()
    root_ptr = parse_expr(vm, expr)
    frontier = jnp.array([root_ptr], dtype=jnp.int32)
    vm.ledger, frontier = _run_intrinsic(vm.ledger, frontier, max_steps)
    return vm, int(frontier[0])


def pretty_bsp_intrinsic(expr, max_steps=64, vm=None):
    vm, ptr = denote_bsp_intrinsic(expr, max_steps=max_steps, vm=vm)
    return vm.decode(ptr)


def denote_bsp_candidates(expr, max_steps=64, vm=None, validate_stratum=False):
    vm = vm or pv.PrismVM_BSP()
    root_ptr = parse_expr(vm, expr)
    frontier = jnp.array([root_ptr], dtype=jnp.int32)
    vm.ledger, frontier = _run_candidates(
        vm.ledger, frontier, max_steps, validate_stratum=validate_stratum
    )
    return vm, int(frontier[0])


def pretty_bsp_candidates(expr, max_steps=64, vm=None, validate_stratum=False):
    vm, ptr = denote_bsp_candidates(
        expr, max_steps=max_steps, vm=vm, validate_stratum=validate_stratum
    )
    return vm.decode(ptr)
