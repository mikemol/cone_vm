import jax.numpy as jnp
import pytest

import prism_vm as pv


def _require_candidate_api():
    missing = []
    if not hasattr(pv, "CandidateBuffer"):
        missing.append("CandidateBuffer")
    if not hasattr(pv, "emit_candidates"):
        missing.append("emit_candidates")
    if not hasattr(pv, "compact_candidates"):
        missing.append("compact_candidates")
    if missing:
        pytest.xfail("CNF-2 candidate pipeline not implemented: " + ", ".join(missing))


def test_candidate_emit_fixed_arity():
    _require_candidate_api()
    ledger = pv.init_ledger()
    frontier = jnp.array([1, 1], dtype=jnp.int32)
    candidates = pv.emit_candidates(ledger, frontier)
    expected = int(frontier.shape[0]) * 2
    assert candidates.enabled.shape[0] == expected
    assert candidates.opcode.shape[0] == expected
    assert candidates.arg1.shape[0] == expected
    assert candidates.arg2.shape[0] == expected


def test_candidate_compaction_enabled_only():
    _require_candidate_api()
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
    compacted, count = pv.compact_candidates(candidates)
    assert int(count) == 2
    assert int(compacted.opcode[0]) == pv.OP_MUL
    assert int(compacted.opcode[1]) == pv.OP_SUC
    assert int(compacted.arg1[0]) == 2
    assert int(compacted.arg1[1]) == 3
    assert int(compacted.arg2[0]) == 8
    assert int(compacted.arg2[1]) == 7
