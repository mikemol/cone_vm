import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m2


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
    assert int(canon_ids[0]) == int(node_id)
    mapped = q_map(jnp.array([node_id], dtype=jnp.int32))
    assert int(mapped[0]) == int(node_id)


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

    def prior_q(ids):
        return jnp.where(ids == suc_zero, jnp.int32(pv.ZERO_PTR), ids)

    stratum = pv.Stratum(start=jnp.int32(add_id), count=jnp.int32(1))
    _, canon_ids, q_map = pv.commit_stratum(
        ledger, stratum, prior_q=prior_q, validate=True
    )
    assert int(canon_ids[0]) == int(add_zero)
    mapped = q_map(jnp.array([add_id], dtype=jnp.int32))
    assert int(mapped[0]) == int(add_zero)
