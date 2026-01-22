import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m2


def test_stratum_no_within_refs_passes():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    y_id = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    frontier = jnp.array([add_ids[0]], dtype=jnp.int32)
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
