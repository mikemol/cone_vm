import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def test_fixed_point_after_simple_rewrite():
    ledger = pv.init_ledger()
    suc_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_SUC], [pv.ZERO_PTR], [0]
    )
    suc_id = int(suc_ids[0])
    add_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_ADD], [pv.ZERO_PTR], [suc_id]
    )
    add_id = int(add_ids[0])

    keys0 = set(mph.canon_state_ledger(ledger)[0])
    frontier = jnp.array([add_id], dtype=jnp.int32)

    ledger1, frontier1 = pv.cycle_intrinsic(ledger, frontier)
    keys1 = set(mph.canon_state_ledger(ledger1)[0])

    ledger2, frontier2 = pv.cycle_intrinsic(ledger1, frontier1)
    keys2 = set(mph.canon_state_ledger(ledger2)[0])

    assert keys0.issubset(keys1)
    assert keys1.issubset(keys2)
    assert mph.canon_state_ledger(ledger1) == mph.canon_state_ledger(ledger2)
    assert int(frontier1[0]) == int(frontier2[0])
