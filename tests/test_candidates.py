import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m2


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


def test_cnf2_slot_layout_indices():
    _require_candidate_api()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC, pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_zero_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
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
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
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
    _, count, comp_idx = pv.compact_candidates_with_index(candidates)
    size = candidates.enabled.shape[0]
    count_i = int(count)
    ids_compact = jnp.arange(size, dtype=jnp.int32) + 100
    ids_full = pv._scatter_compacted_ids(comp_idx, ids_compact, count, size)

    enabled_np = jax.device_get(candidates.enabled)
    comp_idx_np = jax.device_get(comp_idx[:count_i])
    ids_full_np = jax.device_get(ids_full)
    ids_compact_np = jax.device_get(ids_compact)
    pos_to_id = {int(pos): int(ids_compact_np[i]) for i, pos in enumerate(comp_idx_np)}
    slot0_positions = [2 * i for i in range(frontier.shape[0])]
    for pos in slot0_positions:
        if enabled_np[pos]:
            assert int(ids_full_np[pos]) == pos_to_id[pos]
        else:
            assert int(ids_full_np[pos]) == 0
        assert int(ids_full_np[pos + 1]) == 0


def test_candidate_emit_frontier_permutation_invariant():
    _require_candidate_api()
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC, pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR, pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    suc_x_id = suc_ids[0]
    y_id = suc_ids[1]
    add_zero_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
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
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
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
    perm = jnp.array([2, 0, 4, 1, 3], dtype=jnp.int32)
    inv_perm = jnp.argsort(perm)
    frontier_perm = frontier[perm]
    candidates = pv.emit_candidates(ledger, frontier)
    candidates_perm = pv.emit_candidates(ledger, frontier_perm)

    def slot0_payloads(cands):
        enabled = cands.enabled[0::2].astype(jnp.int32)
        return enabled, cands.opcode[0::2], cands.arg1[0::2], cands.arg2[0::2]

    enabled0, op0, a10, a20 = slot0_payloads(candidates)
    enabled1, op1, a11, a21 = slot0_payloads(candidates_perm)
    assert bool(jnp.array_equal(enabled0, enabled1[inv_perm]))
    assert bool(jnp.array_equal(op0, op1[inv_perm]))
    assert bool(jnp.array_equal(a10, a11[inv_perm]))
    assert bool(jnp.array_equal(a20, a21[inv_perm]))

    def slot0_ids(base_ledger, cands, frontier_len):
        compacted, count, comp_idx = pv.compact_candidates_with_index(cands)
        enabled = compacted.enabled.astype(jnp.int32)
        ops = jnp.where(enabled, compacted.opcode, jnp.int32(0))
        a1 = jnp.where(enabled, compacted.arg1, jnp.int32(0))
        a2 = jnp.where(enabled, compacted.arg2, jnp.int32(0))
        ids_compact, _ = pv.intern_nodes(base_ledger, ops, a1, a2)
        size = cands.enabled.shape[0]
        ids_full = pv._scatter_compacted_ids(comp_idx, ids_compact, count, size)
        idx0 = jnp.arange(frontier_len, dtype=jnp.int32) * 2
        return ids_full[idx0]

    slot0_ids_ref = slot0_ids(ledger, candidates, frontier.shape[0])
    slot0_ids_perm = slot0_ids(ledger, candidates_perm, frontier.shape[0])
    assert bool(jnp.array_equal(slot0_ids_ref, slot0_ids_perm[inv_perm]))


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
