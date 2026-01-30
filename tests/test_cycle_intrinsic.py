import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

i32 = harness.i32
intern_nodes = harness.intern_nodes
intern1 = harness.intern1

pytestmark = [
    pytest.mark.m1,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def _ledger_snapshot(ledger):
    return (
        jax.device_get(ledger.opcode),
        jax.device_get(ledger.arg1),
        jax.device_get(ledger.arg2),
        jax.device_get(ledger.keys_b0_sorted),
        jax.device_get(ledger.keys_b1_sorted),
        jax.device_get(ledger.keys_b2_sorted),
        jax.device_get(ledger.keys_b3_sorted),
        jax.device_get(ledger.keys_b4_sorted),
        jax.device_get(ledger.ids_sorted),
    )


def _assert_ledger_snapshot(ledger, snapshot):
    fields = _ledger_snapshot(ledger)
    for field, expected in zip(fields, snapshot):
        assert (field == expected).all()


def test_cycle_intrinsic_stop_path_on_corrupt():
    assert hasattr(pv, "cycle_intrinsic")
    ledger = pv.init_ledger()
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    frontier = i32([suc_id])
    ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))
    snapshot = _ledger_snapshot(ledger)
    with pytest.raises(RuntimeError, match="CORRUPT"):
        pv.cycle_intrinsic(ledger, frontier)
    _assert_ledger_snapshot(ledger, snapshot)


def test_cycle_intrinsic_stop_path_on_oom():
    assert hasattr(pv, "cycle_intrinsic")
    ledger = pv.init_ledger()
    suc_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    frontier = i32([suc_id])
    ledger = ledger._replace(oom=jnp.array(True, dtype=jnp.bool_))
    snapshot = _ledger_snapshot(ledger)
    with pytest.raises(RuntimeError, match="Ledger capacity exceeded"):
        pv.cycle_intrinsic(ledger, frontier)
    _assert_ledger_snapshot(ledger, snapshot)


@pytest.mark.m3
def test_cycle_intrinsic_does_not_mutate_preexisting_rows():
    assert hasattr(pv, "cycle_intrinsic")
    ledger = pv.init_ledger()
    suc_ids, ledger = intern_nodes(
        ledger, [pv.OP_SUC, pv.OP_SUC], [pv.ZERO_PTR, pv.ZERO_PTR], [0, 0]
    )
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_ids[0], suc_ids[1])
    frontier = i32([add_id])
    start_count = int(ledger.count)
    pre_ops = jax.device_get(ledger.opcode[:start_count])
    pre_a1 = jax.device_get(ledger.arg1[:start_count])
    pre_a2 = jax.device_get(ledger.arg2[:start_count])

    ledger, _ = pv.cycle_intrinsic(ledger, frontier)

    post_ops = jax.device_get(ledger.opcode[:start_count])
    post_a1 = jax.device_get(ledger.arg1[:start_count])
    post_a2 = jax.device_get(ledger.arg2[:start_count])
    assert bool(jnp.array_equal(pre_ops, post_ops))
    assert bool(jnp.array_equal(pre_a1, post_a1))
    assert bool(jnp.array_equal(pre_a2, post_a2))
