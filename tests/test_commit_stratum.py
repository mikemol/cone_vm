import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = [
    pytest.mark.m2,
    pytest.mark.backend_matrix,
    pytest.mark.usefixtures("backend_device"),
]


def test_commit_stratum_identity():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    node_id = suc_ids[0]
    stratum = pv.Stratum(start=jnp.int32(node_id), count=jnp.int32(1))
    ledger2, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, validate=True
    )
    assert int(ledger2.count) == int(ledger.count)
    assert int(canon_ids.a[0]) == int(node_id)
    mapped = q_map(pv._provisional_ids(jnp.array([node_id], dtype=jnp.int32)))
    assert int(mapped.a[0]) == int(node_id)


def test_commit_stratum_applies_prior_q_to_children():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_zero = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([suc_zero], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    add_id = add_ids[0]
    add_zero_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    add_zero = add_zero_ids[0]

    def prior_q(ids: pv.ProvisionalIds) -> pv.CommittedIds:
        mapped = jnp.where(
            ids.a == suc_zero, jnp.int32(pv.ZERO_PTR), ids.a
        )
        return pv._committed_ids(mapped)

    stratum = pv.Stratum(start=jnp.int32(add_id), count=jnp.int32(1))
    _, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, prior_q=prior_q, validate=True
    )
    assert int(canon_ids.a[0]) == int(add_zero)
    mapped = q_map(pv._provisional_ids(jnp.array([add_id], dtype=jnp.int32)))
    assert int(mapped.a[0]) == int(add_zero)


def test_commit_stratum_count_mismatch_fails(monkeypatch):
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    node_id = suc_ids[0]
    stratum = pv.Stratum(start=jnp.int32(node_id), count=jnp.int32(1))
    real_intern = pv.intern_nodes

    def bad_intern(ledger_in, batch_or_ops, a1=None, a2=None):
        ids, new_ledger = real_intern(ledger_in, batch_or_ops, a1, a2)
        return ids[:0], new_ledger

    monkeypatch.setattr(pv, "intern_nodes", bad_intern)
    with pytest.raises(ValueError, match="Stratum count mismatch"):
        pv.commit_stratum(ledger, stratum, validate=True)


def test_commit_stratum_q_map_totality_on_mixed_ids():
    ledger = pv.init_ledger()
    suc_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc_zero = suc_ids[0]
    add_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    node_id = add_ids[0]
    node_id_i = int(node_id)

    def prior_q(ids: pv.ProvisionalIds) -> pv.CommittedIds:
        mapped = jnp.where(ids.a == pv.ZERO_PTR, jnp.int32(suc_zero), ids.a)
        return pv._committed_ids(mapped)

    stratum = pv.Stratum(start=jnp.int32(node_id_i), count=jnp.int32(1))
    _, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, prior_q=prior_q, validate=True
    )
    mixed = jnp.array([0, pv.ZERO_PTR, node_id_i, node_id_i + 1], dtype=jnp.int32)
    mapped = q_map(pv._provisional_ids(mixed)).a
    assert int(mapped[0]) == 0
    assert int(mapped[1]) == int(suc_zero)
    assert int(mapped[2]) == int(canon_ids.a[0])
    assert int(mapped[3]) == node_id_i + 1
    mapped2 = q_map(pv._provisional_ids(mapped)).a
    assert bool(jnp.array_equal(mapped, mapped2))
