import jax.numpy as jnp
import pytest

import prism_vm as pv


def _require_candidate_intern_api():
    missing = []
    if not hasattr(pv, "intern_candidates"):
        missing.append("intern_candidates")
    if missing:
        pytest.xfail("Candidate interning not implemented: " + ", ".join(missing))


def test_intern_candidates_compacts_enabled():
    _require_candidate_intern_api()
    enabled = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    opcode = jnp.array([pv.OP_ADD, pv.OP_MUL, pv.OP_SUC, pv.OP_ZERO], dtype=jnp.int32)
    arg1 = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    arg2 = jnp.array([9, 8, 7, 6], dtype=jnp.int32)
    candidates = pv.CandidateBuffer(
        enabled=enabled,
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
    )
    ledger = pv.init_ledger()
    ids, new_ledger, count = pv.intern_candidates(ledger, candidates)
    assert int(count) == 2
    expected_ids, expected_ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL, pv.OP_SUC], dtype=jnp.int32),
        jnp.array([2, 3], dtype=jnp.int32),
        jnp.array([8, 7], dtype=jnp.int32),
    )
    assert int(new_ledger.count) == int(expected_ledger.count)
    assert bool(jnp.array_equal(ids[: int(count)], expected_ids))


def test_intern_candidates_dedup():
    _require_candidate_intern_api()
    enabled = jnp.array([1, 1], dtype=jnp.int32)
    opcode = jnp.array([pv.OP_ADD, pv.OP_ADD], dtype=jnp.int32)
    arg1 = jnp.array([1, 1], dtype=jnp.int32)
    arg2 = jnp.array([1, 1], dtype=jnp.int32)
    candidates = pv.CandidateBuffer(
        enabled=enabled,
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
    )
    ledger = pv.init_ledger()
    ids, new_ledger, count = pv.intern_candidates(ledger, candidates)
    assert int(count) == 2
    assert int(ids[0]) == int(ids[1])
    assert int(new_ledger.count) == 3
