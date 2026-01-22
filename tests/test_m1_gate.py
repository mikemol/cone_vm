import jax.numpy as jnp

import prism_vm as pv
from tests import harness


def _ledger_corrupt_flag(ledger):
    if hasattr(ledger, "corrupt"):
        return bool(ledger.corrupt)
    return bool(ledger.oom)


def test_add_zero_equivalence_baseline_vs_ledger():
    exprs = [
        "(add zero (suc zero))",
        "(add (suc zero) zero)",
    ]
    for expr in exprs:
        assert harness.pretty_baseline(expr) == harness.pretty_bsp_intrinsic(expr)


def test_intern_deterministic_ids_single_engine():
    vm = pv.PrismVM_BSP()
    expr = "(add (suc zero) (suc zero))"
    ptr1 = harness.parse_expr(vm, expr)
    count1 = int(vm.ledger.count)
    ptr2 = harness.parse_expr(vm, expr)
    count2 = int(vm.ledger.count)
    assert int(ptr1) == int(ptr2)
    assert count1 == count2


def test_univalence_no_alias_guard():
    assert hasattr(pv, "MAX_ID"), "MAX_ID must be defined for hard-cap mode"
    assert pv.MAX_NODES == pv.MAX_ID + 1
    assert pv.MAX_ID < (1 << 16)


def test_intern_corrupt_flag_trips():
    assert hasattr(pv, "MAX_ID"), "MAX_ID must be defined for hard-cap mode"
    ledger = pv.init_ledger()
    ledger = ledger._replace(count=jnp.array(pv.MAX_ID + 1, dtype=jnp.int32))
    ids, new_ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    assert _ledger_corrupt_flag(new_ledger)
    assert int(ids[0]) == 0
