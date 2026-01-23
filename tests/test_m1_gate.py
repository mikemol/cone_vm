import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

pytestmark = pytest.mark.m1


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
    assert not bool(new_ledger.oom)
    assert int(ids[0]) == 0


def test_intern_corrupt_flag_trips_on_a1_overflow():
    ledger = pv.init_ledger()
    ids, new_ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.MAX_ID + 1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    assert bool(new_ledger.corrupt)
    assert not bool(new_ledger.oom)
    assert int(ids[0]) == 0


def test_intern_corrupt_flag_trips_on_a2_overflow():
    ledger = pv.init_ledger()
    ids, new_ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.MAX_ID + 1], dtype=jnp.int32),
    )
    assert bool(new_ledger.corrupt)
    assert not bool(new_ledger.oom)
    assert int(ids[0]) == 0


@pytest.mark.parametrize("bad_a1, bad_a2", [(-1, 0), (0, -1)])
def test_intern_corrupt_flag_trips_on_negative_child_id(bad_a1, bad_a2):
    ledger = pv.init_ledger()
    ids, new_ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([bad_a1], dtype=jnp.int32),
        jnp.array([bad_a2], dtype=jnp.int32),
    )
    assert bool(new_ledger.corrupt)
    assert not bool(new_ledger.oom)
    assert int(ids[0]) == 0


@pytest.mark.parametrize("bad_op", [-1, 256])
def test_intern_corrupt_flag_trips_on_opcode_out_of_range(bad_op):
    ledger = pv.init_ledger()
    ids, new_ledger = pv.intern_nodes(
        ledger,
        jnp.array([bad_op], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    assert bool(new_ledger.corrupt)
    assert not bool(new_ledger.oom)
    assert int(ids[0]) == 0


def test_intern_raises_on_corrupt_host():
    vm = pv.PrismVM_BSP()
    vm.ledger = vm.ledger._replace(count=jnp.array(pv.MAX_ID + 1, dtype=jnp.int32))
    with pytest.raises(RuntimeError, match="CORRUPT"):
        vm._intern(pv.OP_SUC, pv.ZERO_PTR, 0)


def test_ledger_full_key_equality():
    ledger = pv.init_ledger()
    ops = jnp.array([pv.OP_ADD, pv.OP_ADD], dtype=jnp.int32)
    a1 = jnp.array([pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32)
    a2 = jnp.array([pv.ZERO_PTR, pv.ZERO_PTR + 1], dtype=jnp.int32)
    ids, ledger = pv.intern_nodes(ledger, ops, a1, a2)
    assert int(ids[0]) != int(ids[1])
    assert int(ledger.count) == 4


def test_key_width_no_alias():
    assert pv.MAX_ID >= 0x101
    ledger = pv.init_ledger()
    ops = jnp.array([pv.OP_SUC, pv.OP_SUC], dtype=jnp.int32)
    a1 = jnp.array([1, 1 + 256], dtype=jnp.int32)
    a2 = jnp.array([0, 0], dtype=jnp.int32)
    ids, ledger = pv.intern_nodes(ledger, ops, a1, a2)
    assert int(ids[0]) != int(ids[1])
    assert int(ledger.count) == 4
