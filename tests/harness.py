import re

import jax.numpy as jnp

import prism_vm as pv

TOKEN_RE = re.compile(r"\(|\)|[a-z]+")
STATUS_CONVERGED = "converged"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_ERROR = "error"


def tokenize(expr):
    return TOKEN_RE.findall(expr)


def parse_expr(vm, expr):
    tokens = tokenize(expr)
    return vm.parse(tokens)


def normalize_baseline(expr, max_steps=64, vm=None):
    vm = vm or pv.PrismVM()
    ptr = parse_expr(vm, expr)
    last = None
    for _ in range(max_steps):
        ptr = vm.eval(ptr)
        curr = int(ptr)
        if curr == last:
            return STATUS_CONVERGED, vm, ptr
        last = curr
    return STATUS_BUDGET_EXHAUSTED, vm, ptr


def run_baseline(expr, max_steps=64, vm=None):
    try:
        status, vm, ptr = normalize_baseline(expr, max_steps=max_steps, vm=vm)
    except Exception:
        return STATUS_ERROR, None
    return status, vm.decode(ptr)


def denote_baseline(expr, vm=None, max_steps=64):
    status, vm, ptr = normalize_baseline(expr, max_steps=max_steps, vm=vm)
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"baseline evaluation did not converge within max_steps={max_steps}"
        )
    return vm, ptr


def pretty_baseline(expr, vm=None, max_steps=64):
    status, pretty = run_baseline(expr, max_steps=max_steps, vm=vm)
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"baseline evaluation did not converge within max_steps={max_steps}"
        )
    return pretty


def denote_pretty_baseline(expr, vm=None, max_steps=64):
    vm, ptr = denote_baseline(expr, vm=vm, max_steps=max_steps)
    return vm.decode(ptr)


def normalize_bsp_intrinsic(expr, max_steps=64, vm=None):
    vm = vm or pv.PrismVM_BSP()
    root_ptr = parse_expr(vm, expr)
    frontier = jnp.array([int(root_ptr)], dtype=jnp.int32)
    last = None
    for _ in range(max_steps):
        vm.ledger, frontier = pv.cycle_intrinsic(vm.ledger, frontier)
        vm.ledger.count.block_until_ready()
        if pv.ledger_has_corrupt(vm.ledger):
            raise RuntimeError(
                "CORRUPT: key encoding alias risk (id width exceeded)"
            )
        curr = pv._host_int_value(frontier[0])
        if curr == last:
            return STATUS_CONVERGED, vm, pv._ledger_id(curr)
        last = curr
    return STATUS_BUDGET_EXHAUSTED, vm, pv._ledger_id(pv._host_int_value(frontier[0]))


def run_bsp_intrinsic(expr, max_steps=64, vm=None):
    try:
        status, vm, ptr = normalize_bsp_intrinsic(
            expr, max_steps=max_steps, vm=vm
        )
    except Exception:
        return STATUS_ERROR, None
    return status, vm.decode(ptr)


def denote_bsp_intrinsic(expr, max_steps=64, vm=None):
    status, vm, ptr = normalize_bsp_intrinsic(expr, max_steps=max_steps, vm=vm)
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP intrinsic evaluation did not converge within max_steps={max_steps}"
        )
    return vm, ptr


def pretty_bsp_intrinsic(expr, max_steps=64, vm=None):
    status, pretty = run_bsp_intrinsic(expr, max_steps=max_steps, vm=vm)
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP intrinsic evaluation did not converge within max_steps={max_steps}"
        )
    return pretty


def denote_pretty_bsp_intrinsic(expr, max_steps=64, vm=None):
    vm, ptr = denote_bsp_intrinsic(expr, max_steps=max_steps, vm=vm)
    return vm.decode(ptr)


def normalize_bsp_candidates(expr, max_steps=64, vm=None, validate_stratum=False):
    vm = vm or pv.PrismVM_BSP()
    root_ptr = parse_expr(vm, expr)
    frontier = pv._committed_ids(
        jnp.array([int(root_ptr)], dtype=jnp.int32)
    )
    last = None
    for _ in range(max_steps):
        vm.ledger, next_frontier_prov, _, q_map = pv.cycle_candidates(
            vm.ledger, frontier, validate_stratum=validate_stratum
        )
        next_frontier = pv.apply_q(q_map, next_frontier_prov)
        vm.ledger.count.block_until_ready()
        if pv.ledger_has_corrupt(vm.ledger):
            raise RuntimeError(
                "CORRUPT: key encoding alias risk (id width exceeded)"
            )
        if next_frontier.a.shape[0] == 0:
            return STATUS_CONVERGED, vm, pv._ledger_id(pv._host_int_value(frontier.a[0]))
        curr = pv._host_int_value(next_frontier.a[0])
        if curr == last:
            return STATUS_CONVERGED, vm, pv._ledger_id(curr)
        last = curr
        frontier = next_frontier
    return STATUS_BUDGET_EXHAUSTED, vm, pv._ledger_id(pv._host_int_value(frontier.a[0]))


def run_bsp_candidates(expr, max_steps=64, vm=None, validate_stratum=False):
    try:
        status, vm, ptr = normalize_bsp_candidates(
            expr,
            max_steps=max_steps,
            vm=vm,
            validate_stratum=validate_stratum,
        )
    except Exception:
        return STATUS_ERROR, None
    return status, vm.decode(ptr)


def denote_bsp_candidates(expr, max_steps=64, vm=None, validate_stratum=False):
    status, vm, ptr = normalize_bsp_candidates(
        expr, max_steps=max_steps, vm=vm, validate_stratum=validate_stratum
    )
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP candidates evaluation did not converge within max_steps={max_steps}"
        )
    return vm, ptr


def pretty_bsp_candidates(expr, max_steps=64, vm=None, validate_stratum=False):
    status, pretty = run_bsp_candidates(
        expr, max_steps=max_steps, vm=vm, validate_stratum=validate_stratum
    )
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP candidates evaluation did not converge within max_steps={max_steps}"
        )
    return pretty


def denote_pretty_bsp_candidates(expr, max_steps=64, vm=None, validate_stratum=False):
    vm, ptr = denote_bsp_candidates(
        expr, max_steps=max_steps, vm=vm, validate_stratum=validate_stratum
    )
    return vm.decode(ptr)


def _assert_converged(status, label, max_steps):
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"{label} evaluation did not converge within max_steps={max_steps}"
        )


def assert_baseline_equals_bsp_intrinsic(expr, max_steps=64):
    pretty_base = denote_pretty_baseline(expr, max_steps=max_steps)
    pretty_bsp = denote_pretty_bsp_intrinsic(expr, max_steps=max_steps)
    assert pretty_base == pretty_bsp


def assert_baseline_equals_bsp_candidates(expr, max_steps=64, validate_stratum=False):
    pretty_base = denote_pretty_baseline(expr, max_steps=max_steps)
    pretty_bsp = denote_pretty_bsp_candidates(
        expr, max_steps=max_steps, validate_stratum=validate_stratum
    )
    assert pretty_base == pretty_bsp


def run_arena(
    expr,
    steps=4,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
):
    vm = pv.PrismVM_BSP_Legacy()
    root_ptr = vm.parse(tokenize(expr))
    arena = vm.arena
    for _ in range(steps):
        arena, root_ptr = pv.cycle(
            arena,
            root_ptr,
            do_sort=do_sort,
            use_morton=use_morton,
            block_size=block_size,
            l2_block_size=l2_block_size,
            l1_block_size=l1_block_size,
            do_global=do_global,
        )
        root_ptr = pv._arena_ptr(pv._host_int_value(root_ptr))
    vm.arena = arena
    return vm.decode(root_ptr)


def assert_arena_schedule_invariance(expr, steps=4):
    no_sort = run_arena(expr, steps=steps, do_sort=False, use_morton=False)
    rank_sort = run_arena(expr, steps=steps, do_sort=True, use_morton=False)
    morton_sort = run_arena(expr, steps=steps, do_sort=True, use_morton=True)
    assert no_sort == rank_sort
    assert no_sort == morton_sort
