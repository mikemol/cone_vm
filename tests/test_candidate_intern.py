import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

intern_nodes = harness.intern_nodes

pytestmark = [
    pytest.mark.m2,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def _require_candidate_intern_api():
    assert hasattr(pv, "intern_candidates")


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
    expected_ids, expected_ledger = intern_nodes(
        ledger, [pv.OP_MUL, pv.OP_SUC], [2, 3], [8, 7]
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


def test_intern_candidates_sees_only_enabled():
    _require_candidate_intern_api()
    enabled = jnp.array([1, 0, 0], dtype=jnp.int32)
    opcode = jnp.array([pv.OP_SUC, pv.OP_ADD, pv.OP_MUL], dtype=jnp.int32)
    arg1 = jnp.array([pv.ZERO_PTR, pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32)
    arg2 = jnp.array([0, pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32)
    candidates = pv.CandidateBuffer(
        enabled=enabled,
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
    )
    ledger = pv.init_ledger()
    ids, new_ledger, count = pv.intern_candidates(ledger, candidates)
    assert int(count) == 1
    assert int(new_ledger.count) == 3
    assert bool(jnp.all(ids[int(count):] == 0))
