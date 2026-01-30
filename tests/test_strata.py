import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

intern1 = harness.intern1
i32 = harness.i32

pytestmark = pytest.mark.m2


def test_stratum_no_within_refs_passes():
    ledger = pv.init_ledger()
    y_id, ledger = intern1(ledger, pv.OP_SUC, 1, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, 1, y_id)
    frontier = i32([add_id])
    candidates = pv.emit_candidates(ledger, frontier)
    start = int(ledger.count)
    _, new_ledger, _ = pv.intern_candidates(ledger, candidates)
    new_count = int(new_ledger.count) - start
    stratum = pv.Stratum(start=jnp.int32(start), count=jnp.int32(new_count))
    assert pv.validate_stratum_no_within_refs(new_ledger, stratum)


def test_stratum_no_within_refs_detects_self_ref():
    ledger = pv.init_ledger()
    start = int(ledger.count)
    candidates = pv.CandidateBuffer(
        enabled=jnp.array([1], dtype=jnp.int32),
        opcode=jnp.array([pv.OP_SUC], dtype=jnp.int32),
        arg1=jnp.array([start], dtype=jnp.int32),
        arg2=jnp.array([0], dtype=jnp.int32),
    )
    _, new_ledger, _ = pv.intern_candidates(ledger, candidates)
    new_count = int(new_ledger.count) - start
    stratum = pv.Stratum(start=jnp.int32(start), count=jnp.int32(new_count))
    assert not pv.validate_stratum_no_within_refs(new_ledger, stratum)
