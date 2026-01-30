import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests import harness

i32 = harness.i32
intern_nodes = harness.intern_nodes
intern1 = harness.intern1

pytestmark = [
    pytest.mark.m2,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def test_commit_stratum_identity():
    ledger = pv.init_ledger()
    node_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    stratum = pv.Stratum(start=jnp.int32(node_id), count=jnp.int32(1))
    ledger2, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, validate_mode=pv.ValidateMode.STRICT
    )
    assert int(ledger2.count) == int(ledger.count)
    assert int(canon_ids.a[0]) == int(node_id)
    mapped = q_map(pv._provisional_ids(i32([node_id])))
    assert int(mapped.a[0]) == int(node_id)


def test_commit_stratum_applies_prior_q_to_children():
    ledger = pv.init_ledger()
    suc_zero, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, suc_zero, pv.ZERO_PTR)
    add_zero, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, pv.ZERO_PTR)

    def prior_q(ids: pv.ProvisionalIds) -> pv.CommittedIds:
        mapped = jnp.where(
            ids.a == suc_zero, jnp.int32(pv.ZERO_PTR), ids.a
        )
        return pv._committed_ids(mapped)

    stratum = pv.Stratum(start=jnp.int32(add_id), count=jnp.int32(1))
    _, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, prior_q=prior_q, validate_mode=pv.ValidateMode.STRICT
    )
    assert int(canon_ids.a[0]) == int(add_zero)
    mapped = q_map(pv._provisional_ids(i32([add_id])))
    assert int(mapped.a[0]) == int(add_zero)


def test_commit_stratum_applies_prior_q_to_children_a2():
    ledger = pv.init_ledger()
    suc_zero, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    add_id, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, suc_zero)
    add_zero, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, pv.ZERO_PTR)

    def prior_q(ids: pv.ProvisionalIds) -> pv.CommittedIds:
        mapped = jnp.where(
            ids.a == suc_zero, jnp.int32(pv.ZERO_PTR), ids.a
        )
        return pv._committed_ids(mapped)

    stratum = pv.Stratum(start=jnp.int32(add_id), count=jnp.int32(1))
    _, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, prior_q=prior_q, validate_mode=pv.ValidateMode.STRICT
    )
    assert int(canon_ids.a[0]) == int(add_zero)
    mapped = q_map(pv._provisional_ids(i32([add_id])))
    assert int(mapped.a[0]) == int(add_zero)


def test_commit_stratum_count_mismatch_fails(monkeypatch):
    ledger = pv.init_ledger()
    node_id, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    stratum = pv.Stratum(start=jnp.int32(node_id), count=jnp.int32(1))
    real_intern = pv.intern_nodes

    def bad_intern(ledger_in, batch_or_ops, a1=None, a2=None):
        ids, new_ledger = real_intern(ledger_in, batch_or_ops, a1, a2)
        return ids[:0], new_ledger

    with pytest.raises(ValueError, match="Stratum count mismatch"):
        pv.commit_stratum(
            ledger,
            stratum,
            validate_mode=pv.ValidateMode.STRICT,
            intern_fn=bad_intern,
        )


def test_commit_stratum_validate_trips_on_within_refs():
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
    with pytest.raises(ValueError, match="Stratum contains within-tier references"):
        pv.commit_stratum(
            new_ledger, stratum, validate_mode=pv.ValidateMode.STRICT
        )


def test_commit_stratum_q_map_totality_on_mixed_ids():
    ledger = pv.init_ledger()
    suc_zero, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    node_id, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, pv.ZERO_PTR)
    node_id_i = int(node_id)

    def prior_q(ids: pv.ProvisionalIds) -> pv.CommittedIds:
        mapped = jnp.where(ids.a == pv.ZERO_PTR, jnp.int32(suc_zero), ids.a)
        return pv._committed_ids(mapped)

    stratum = pv.Stratum(start=jnp.int32(node_id_i), count=jnp.int32(1))
    _, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, prior_q=prior_q, validate_mode=pv.ValidateMode.STRICT
    )
    mixed = jnp.array([0, pv.ZERO_PTR, node_id_i, node_id_i + 1], dtype=jnp.int32)
    mapped = q_map(pv._provisional_ids(mixed)).a
    assert int(mapped[0]) == 0
    assert int(mapped[1]) == int(suc_zero)
    assert int(mapped[2]) == int(canon_ids.a[0])
    assert int(mapped[3]) == node_id_i + 1
    mapped2 = q_map(pv._provisional_ids(mapped)).a
    assert bool(jnp.array_equal(mapped, mapped2))


def test_commit_stratum_q_map_preserves_input_order():
    ledger = pv.init_ledger()
    suc_zero, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    node_id, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, pv.ZERO_PTR)
    node_id_i = int(node_id)

    def prior_q(ids: pv.ProvisionalIds) -> pv.CommittedIds:
        mapped = jnp.where(ids.a == pv.ZERO_PTR, jnp.int32(suc_zero), ids.a)
        return pv._committed_ids(mapped)

    stratum = pv.Stratum(start=jnp.int32(node_id_i), count=jnp.int32(1))
    _, _, q_map = pv.commit_stratum(
        ledger, stratum, prior_q=prior_q, validate_mode=pv.ValidateMode.STRICT
    )
    mixed = jnp.array([0, pv.ZERO_PTR, node_id_i, node_id_i + 1], dtype=jnp.int32)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    mapped = q_map(pv._provisional_ids(mixed)).a
    mapped_perm = q_map(pv._provisional_ids(mixed[perm])).a
    assert bool(jnp.array_equal(mapped_perm, mapped[perm]))
