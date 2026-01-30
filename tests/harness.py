import re

import jax.numpy as jnp

import prism_vm as pv
import ic_vm as ic

TOKEN_RE = re.compile(r"\(|\)|[a-z]+")
STATUS_CONVERGED = "converged"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_ERROR = "error"


def i32(values):
    """Return int32 JAX array from a list/tuple/array of values."""
    return jnp.asarray(values, dtype=jnp.int32)


def i32v(*values):
    """Return int32 JAX 1D array from varargs."""
    return jnp.array(values, dtype=jnp.int32)


def intern_nodes(ledger, ops, a1s, a2s, **kwargs):
    """Intern nodes using Python lists/arrays; returns (ids, ledger)."""
    return pv.intern_nodes(ledger, i32(ops), i32(a1s), i32(a2s), **kwargs)


def intern1(ledger, op, a1, a2):
    """Intern a single node; returns (id, ledger)."""
    ids, ledger = intern_nodes(ledger, [op], [a1], [a2])
    return ids[0], ledger


def committed_ids(*values):
    """Build CommittedIds from values or a single iterable."""
    if len(values) == 1 and isinstance(values[0], (list, tuple, jnp.ndarray)):
        arr = i32(values[0])
    else:
        arr = i32v(*values)
    return pv._committed_ids(arr)


def make_cycle_intrinsic_jit(**kwargs):
    """Build a jitted intrinsic cycle entrypoint with fixed DI."""
    return pv.cycle_intrinsic_jit(**kwargs)


def make_cycle_intrinsic_jit_cfg(**kwargs):
    """Build a jitted intrinsic cycle entrypoint from a config."""
    return pv.cycle_intrinsic_jit_cfg(**kwargs)


def make_cycle_candidates_jit(**kwargs):
    """Build a jitted CNF-2 cycle entrypoint with fixed DI."""
    return pv.cycle_candidates_jit(**kwargs)


def make_cycle_candidates_static_jit(**kwargs):
    """Build a jitted CNF-2 cycle entrypoint (static policy)."""
    return pv.cycle_candidates_static_jit(**kwargs)


def make_cycle_candidates_value_jit(**kwargs):
    """Build a jitted CNF-2 cycle entrypoint (policy as JAX value)."""
    return pv.cycle_candidates_value_jit(**kwargs)


def make_emit_candidates_jit_cfg(**kwargs):
    """Build a jitted emit_candidates entrypoint from a config."""
    return pv.emit_candidates_jit_cfg(**kwargs)


def make_compact_candidates_jit_cfg(**kwargs):
    """Build a jitted compact_candidates entrypoint from a config."""
    return pv.compact_candidates_jit_cfg(**kwargs)


def make_compact_candidates_result_jit_cfg(**kwargs):
    """Build a jitted compact_candidates_result entrypoint from a config."""
    return pv.compact_candidates_result_jit_cfg(**kwargs)


def make_compact_candidates_with_index_jit_cfg(**kwargs):
    """Build a jitted compact_candidates_with_index entrypoint from a config."""
    return pv.compact_candidates_with_index_jit_cfg(**kwargs)


def make_compact_candidates_with_index_result_jit_cfg(**kwargs):
    """Build a jitted compact_candidates_with_index_result entrypoint from a config."""
    return pv.compact_candidates_with_index_result_jit_cfg(**kwargs)


def make_intern_candidates_jit_cfg(**kwargs):
    """Build a jitted intern_candidates entrypoint from a config."""
    return pv.intern_candidates_jit_cfg(**kwargs)


def make_cycle_jit(**kwargs):
    """Build a jitted BSP arena cycle entrypoint with fixed DI."""
    return pv.cycle_jit(**kwargs)


def make_cycle_jit_cfg(**kwargs):
    """Build a jitted BSP arena cycle entrypoint from a config."""
    return pv.cycle_jit_cfg(**kwargs)


def make_op_interact_jit_cfg(**kwargs):
    """Build a jitted op_interact entrypoint from a config."""
    return pv.op_interact_jit_cfg(**kwargs)


def make_ic_apply_active_pairs_jit(**kwargs):
    """Build a jitted IC apply_active_pairs entrypoint with fixed DI."""
    return ic.apply_active_pairs_jit(**kwargs)


def make_ic_apply_active_pairs_jit_cfg(**kwargs):
    """Build a jitted IC apply_active_pairs entrypoint from a config."""
    return ic.apply_active_pairs_jit_cfg(**kwargs)


def make_ic_reduce_jit(**kwargs):
    """Build a jitted IC reduce entrypoint with fixed DI."""
    return ic.reduce_jit(**kwargs)


def make_ic_reduce_jit_cfg(**kwargs):
    """Build a jitted IC reduce entrypoint from a config."""
    return ic.reduce_jit_cfg(**kwargs)


def make_ic_find_active_pairs_jit(**kwargs):
    """Build a jitted IC active-pair finder entrypoint with fixed DI."""
    return ic.find_active_pairs_jit(**kwargs)


def make_ic_find_active_pairs_jit_cfg(**kwargs):
    """Build a jitted IC active-pair finder entrypoint from a config."""
    return ic.find_active_pairs_jit_cfg(**kwargs)


def make_ic_compact_active_pairs_jit(**kwargs):
    """Build a jitted IC compact active-pairs entrypoint with fixed DI."""
    return ic.compact_active_pairs_jit(**kwargs)


def make_ic_compact_active_pairs_jit_cfg(**kwargs):
    """Build a jitted IC compact active-pairs entrypoint from a config."""
    return ic.compact_active_pairs_jit_cfg(**kwargs)


def make_ic_compact_active_pairs_result_jit(**kwargs):
    """Build a jitted IC compact-result active-pairs entrypoint with fixed DI."""
    return ic.compact_active_pairs_result_jit(**kwargs)


def make_ic_compact_active_pairs_result_jit_cfg(**kwargs):
    """Build a jitted IC compact-result active-pairs entrypoint from a config."""
    return ic.compact_active_pairs_result_jit_cfg(**kwargs)


def make_ic_wire_jax_jit(**kwargs):
    """Build a jitted IC wire entrypoint with fixed DI."""
    return ic.wire_jax_jit(**kwargs)


def make_ic_wire_jax_jit_cfg(**kwargs):
    """Build a jitted IC wire entrypoint from a config."""
    return ic.wire_jax_jit_cfg(**kwargs)


def make_ic_wire_jax_safe_jit(**kwargs):
    """Build a jitted IC NULL-safe wire entrypoint with fixed DI."""
    return ic.wire_jax_safe_jit(**kwargs)


def make_ic_wire_jax_safe_jit_cfg(**kwargs):
    """Build a jitted IC NULL-safe wire entrypoint from a config."""
    return ic.wire_jax_safe_jit_cfg(**kwargs)


def make_ic_wire_ptrs_jit(**kwargs):
    """Build a jitted IC ptr wire entrypoint with fixed DI."""
    return ic.wire_ptrs_jit(**kwargs)


def make_ic_wire_ptrs_jit_cfg(**kwargs):
    """Build a jitted IC ptr wire entrypoint from a config."""
    return ic.wire_ptrs_jit_cfg(**kwargs)


def make_ic_wire_pairs_jit(**kwargs):
    """Build a jitted IC wire-pairs entrypoint with fixed DI."""
    return ic.wire_pairs_jit(**kwargs)


def make_ic_wire_pairs_jit_cfg(**kwargs):
    """Build a jitted IC wire-pairs entrypoint from a config."""
    return ic.wire_pairs_jit_cfg(**kwargs)


def make_ic_wire_ptr_pairs_jit(**kwargs):
    """Build a jitted IC ptr wire-pairs entrypoint with fixed DI."""
    return ic.wire_ptr_pairs_jit(**kwargs)


def make_ic_wire_ptr_pairs_jit_cfg(**kwargs):
    """Build a jitted IC ptr wire-pairs entrypoint from a config."""
    return ic.wire_ptr_pairs_jit_cfg(**kwargs)


def make_ic_wire_star_jit(**kwargs):
    """Build a jitted IC wire-star entrypoint with fixed DI."""
    return ic.wire_star_jit(**kwargs)


def make_ic_wire_star_jit_cfg(**kwargs):
    """Build a jitted IC wire-star entrypoint from a config."""
    return ic.wire_star_jit_cfg(**kwargs)


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


def normalize_bsp_intrinsic(
    expr,
    max_steps=64,
    vm=None,
    *,
    cycle_intrinsic_fn=None,
    cycle_intrinsic_kwargs=None,
):
    vm = vm or pv.PrismVM_BSP()
    root_ptr = parse_expr(vm, expr)
    frontier = jnp.array([int(root_ptr)], dtype=jnp.int32)
    if cycle_intrinsic_fn is None:
        cycle_intrinsic_fn = pv.cycle_intrinsic
    if cycle_intrinsic_kwargs is None:
        cycle_intrinsic_kwargs = {}
    last = None
    for _ in range(max_steps):
        vm.ledger, frontier = cycle_intrinsic_fn(
            vm.ledger, frontier, **cycle_intrinsic_kwargs
        )
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


def run_bsp_intrinsic(expr, max_steps=64, vm=None, **kwargs):
    try:
        status, vm, ptr = normalize_bsp_intrinsic(
            expr, max_steps=max_steps, vm=vm, **kwargs
        )
    except Exception:
        return STATUS_ERROR, None
    return status, vm.decode(ptr)


def denote_bsp_intrinsic(expr, max_steps=64, vm=None, **kwargs):
    status, vm, ptr = normalize_bsp_intrinsic(
        expr, max_steps=max_steps, vm=vm, **kwargs
    )
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP intrinsic evaluation did not converge within max_steps={max_steps}"
        )
    return vm, ptr


def pretty_bsp_intrinsic(expr, max_steps=64, vm=None, **kwargs):
    status, pretty = run_bsp_intrinsic(
        expr, max_steps=max_steps, vm=vm, **kwargs
    )
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP intrinsic evaluation did not converge within max_steps={max_steps}"
        )
    return pretty


def denote_pretty_bsp_intrinsic(expr, max_steps=64, vm=None, **kwargs):
    vm, ptr = denote_bsp_intrinsic(expr, max_steps=max_steps, vm=vm, **kwargs)
    return vm.decode(ptr)


def normalize_bsp_candidates(
    expr,
    max_steps=64,
    vm=None,
    validate_mode=pv.ValidateMode.NONE,
    *,
    cycle_candidates_fn=None,
    cycle_candidates_kwargs=None,
):
    vm = vm or pv.PrismVM_BSP()
    root_ptr = parse_expr(vm, expr)
    frontier = pv._committed_ids(
        jnp.array([int(root_ptr)], dtype=jnp.int32)
    )
    if cycle_candidates_fn is None:
        cycle_candidates_fn = pv.cycle_candidates
        cycle_candidates_kwargs = {"validate_mode": validate_mode}
    if cycle_candidates_kwargs is None:
        cycle_candidates_kwargs = {}
    last = None
    for _ in range(max_steps):
        vm.ledger, next_frontier_prov, _, q_map = cycle_candidates_fn(
            vm.ledger, frontier, **cycle_candidates_kwargs
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


def run_bsp_candidates(expr, max_steps=64, vm=None, validate_mode=pv.ValidateMode.NONE, **kwargs):
    try:
        status, vm, ptr = normalize_bsp_candidates(
            expr,
            max_steps=max_steps,
            vm=vm,
            validate_mode=validate_mode,
            **kwargs,
        )
    except Exception:
        return STATUS_ERROR, None
    return status, vm.decode(ptr)


def denote_bsp_candidates(expr, max_steps=64, vm=None, validate_mode=pv.ValidateMode.NONE, **kwargs):
    status, vm, ptr = normalize_bsp_candidates(
        expr,
        max_steps=max_steps,
        vm=vm,
        validate_mode=validate_mode,
        **kwargs,
    )
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP candidates evaluation did not converge within max_steps={max_steps}"
        )
    return vm, ptr


def pretty_bsp_candidates(expr, max_steps=64, vm=None, validate_mode=pv.ValidateMode.NONE, **kwargs):
    status, pretty = run_bsp_candidates(
        expr,
        max_steps=max_steps,
        vm=vm,
        validate_mode=validate_mode,
        **kwargs,
    )
    if status != STATUS_CONVERGED:
        raise AssertionError(
            f"BSP candidates evaluation did not converge within max_steps={max_steps}"
        )
    return pretty


def denote_pretty_bsp_candidates(
    expr, max_steps=64, vm=None, validate_mode=pv.ValidateMode.NONE, **kwargs
):
    vm, ptr = denote_bsp_candidates(
        expr,
        max_steps=max_steps,
        vm=vm,
        validate_mode=validate_mode,
        **kwargs,
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


def assert_baseline_equals_bsp_candidates(expr, max_steps=64, validate_mode=pv.ValidateMode.NONE):
    pretty_base = denote_pretty_baseline(expr, max_steps=max_steps)
    pretty_bsp = denote_pretty_bsp_candidates(
        expr, max_steps=max_steps, validate_mode=validate_mode
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
    *,
    cycle_fn=None,
    cycle_kwargs=None,
):
    vm = pv.PrismVM_BSP_Legacy()
    root_ptr = vm.parse(tokenize(expr))
    arena = vm.arena
    if cycle_fn is None:
        cycle_fn = pv.cycle
        cycle_kwargs = {
            "do_sort": do_sort,
            "use_morton": use_morton,
            "block_size": block_size,
            "l2_block_size": l2_block_size,
            "l1_block_size": l1_block_size,
            "do_global": do_global,
        }
    if cycle_kwargs is None:
        cycle_kwargs = {}
    for _ in range(steps):
        arena, root_ptr = cycle_fn(arena, root_ptr, **cycle_kwargs)
        root_ptr = pv._arena_ptr(pv._host_int_value(root_ptr))
    vm.arena = arena
    return vm.decode(root_ptr)


def denote_pretty_arena(
    expr,
    steps=4,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
    *,
    cycle_fn=None,
    cycle_kwargs=None,
):
    return run_arena(
        expr,
        steps=steps,
        do_sort=do_sort,
        use_morton=use_morton,
        block_size=block_size,
        l2_block_size=l2_block_size,
        l1_block_size=l1_block_size,
        do_global=do_global,
        cycle_fn=cycle_fn,
        cycle_kwargs=cycle_kwargs,
    )


def assert_arena_schedule_invariance(expr, steps=4):
    no_sort = denote_pretty_arena(expr, steps=steps, do_sort=False, use_morton=False)
    rank_sort = denote_pretty_arena(expr, steps=steps, do_sort=True, use_morton=False)
    morton_sort = denote_pretty_arena(expr, steps=steps, do_sort=True, use_morton=True)
    assert no_sort == rank_sort
    assert no_sort == morton_sort
