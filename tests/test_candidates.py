import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m3


def _require_candidate_api():
    assert hasattr(pv, "CandidateBuffer")
    assert hasattr(pv, "emit_candidates")
    assert hasattr(pv, "compact_candidates")


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


def test_candidate_emit_add_zero_values():
    _require_candidate_api()
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
    candidates = pv.emit_candidates(ledger, jnp.array([add_ids[0]], dtype=jnp.int32))
    assert int(candidates.enabled[0]) == 1
    assert int(candidates.enabled[1]) == 0
    assert int(candidates.opcode[0]) == int(ledger.opcode[y_id])
    assert int(candidates.arg1[0]) == int(ledger.arg1[y_id])
    assert int(candidates.arg2[0]) == int(ledger.arg2[y_id])


def test_candidate_emit_mul_zero_values():
    _require_candidate_api()
    ledger = pv.init_ledger()
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )
    candidates = pv.emit_candidates(ledger, jnp.array([mul_ids[0]], dtype=jnp.int32))
    assert int(candidates.enabled[0]) == 1
    assert int(candidates.enabled[1]) == 0
    assert int(candidates.opcode[0]) == pv.OP_ZERO
    assert int(candidates.arg1[0]) == 0
    assert int(candidates.arg2[0]) == 0


def test_scatter_compacted_ids_does_not_wrap():
    size = 4
    comp_idx = jnp.array([3, 0, 0, 0], dtype=jnp.int32)
    ids_compact = jnp.array([7, 9, 9, 9], dtype=jnp.int32)
    count = jnp.int32(1)
    ids_full = pv._scatter_compacted_ids(comp_idx, ids_compact, count, size)
    assert int(ids_full[3]) == 7
    assert int(ids_full[0]) == 0


def test_candidate_emit_add_suc_values():
    _require_candidate_api()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    x_id = jnp.int32(1)
    suc_x_id = suc_ids[0]
    y_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    y_id = y_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    candidates = pv.emit_candidates(ledger, jnp.array([add_ids[0]], dtype=jnp.int32))
    assert int(candidates.enabled[0]) == 1
    assert int(candidates.enabled[1]) == 0
    assert int(candidates.opcode[0]) == pv.OP_ADD
    assert int(candidates.arg1[0]) == int(x_id)
    assert int(candidates.arg2[0]) == int(y_id)


def test_candidate_emit_add_suc_right_values():
    _require_candidate_api()
    ledger = pv.init_ledger()
    base_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    base_id = base_ids[0]
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([base_id], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    candidates = pv.emit_candidates(ledger, jnp.array([add_ids[0]], dtype=jnp.int32))
    assert int(candidates.enabled[0]) == 1
    assert int(candidates.enabled[1]) == 0
    assert int(candidates.opcode[0]) == pv.OP_ADD
    assert int(candidates.arg1[0]) == pv.ZERO_PTR
    assert int(candidates.arg2[0]) == int(base_id)


def test_candidate_emit_mul_suc_values():
    _require_candidate_api()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_x_id = suc_ids[0]
    y_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    y_id = y_ids[0]
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    candidates = pv.emit_candidates(ledger, jnp.array([mul_ids[0]], dtype=jnp.int32))
    assert int(candidates.enabled[0]) == 1
    assert int(candidates.enabled[1]) == 0
    assert int(candidates.opcode[0]) == pv.OP_MUL
    assert int(candidates.arg1[0]) == pv.ZERO_PTR
    assert int(candidates.arg2[0]) == int(y_id)


def test_candidate_emit_mul_suc_right_values():
    _require_candidate_api()
    ledger = pv.init_ledger()
    base_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    base_id = base_ids[0]
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_id = suc_ids[0]
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([base_id], dtype=jnp.int32),
        jnp.array([suc_id], dtype=jnp.int32),
    )
    candidates = pv.emit_candidates(ledger, jnp.array([mul_ids[0]], dtype=jnp.int32))
    assert int(candidates.enabled[0]) == 1
    assert int(candidates.enabled[1]) == 0
    assert int(candidates.opcode[0]) == pv.OP_MUL
    assert int(candidates.arg1[0]) == pv.ZERO_PTR
    assert int(candidates.arg2[0]) == int(base_id)


def test_candidate_slot1_disabled_for_all_frontier_nodes():
    _require_candidate_api()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC, pv.OP_SUC], dtype=jnp.int32),
        jnp.array([1, 1], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_zero_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    add_suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    mul_zero_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    mul_suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([suc_x_id], dtype=jnp.int32),
        jnp.array([y_id], dtype=jnp.int32),
    )
    frontier = jnp.array(
        [
            add_zero_ids[0],
            add_suc_ids[0],
            mul_zero_ids[0],
            mul_suc_ids[0],
            suc_x_id,
        ],
        dtype=jnp.int32,
    )
    candidates = pv.emit_candidates(ledger, frontier)
    assert int(jnp.sum(candidates.enabled[1::2])) == 0
