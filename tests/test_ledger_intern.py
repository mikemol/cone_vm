import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m1


def _intern_once(ledger, triples):
    ops = jnp.array([t[0] for t in triples], dtype=jnp.int32)
    a1 = jnp.array([t[1] for t in triples], dtype=jnp.int32)
    a2 = jnp.array([t[2] for t in triples], dtype=jnp.int32)
    return pv.intern_nodes(ledger, ops, a1, a2)


def test_intern_nodes_dedup_batch():
    ledger = pv.init_ledger()
    triples = [
        (pv.OP_ADD, 1, 1),
        (pv.OP_ADD, 1, 1),
    ]
    ids, new_ledger = _intern_once(ledger, triples)
    assert int(ids[0]) == int(ids[1])
    assert int(new_ledger.count) == 3


def test_intern_nodes_reuses_existing():
    ledger = pv.init_ledger()
    triples = [(pv.OP_SUC, 1, 0)]
    ids1, ledger1 = _intern_once(ledger, triples)
    ids2, ledger2 = _intern_once(ledger1, triples)
    assert int(ids1[0]) == int(ids2[0])
    assert int(ledger1.count) == int(ledger2.count)


def test_intern_nodes_order_invariant():
    triples_a = [
        (pv.OP_SUC, 1, 0),
        (pv.OP_ADD, 1, 1),
        (pv.OP_MUL, 1, 1),
    ]
    ledger_a = pv.init_ledger()
    ids_a, ledger_a = _intern_once(ledger_a, triples_a)
    mapping_a = {triples_a[i]: int(ids_a[i]) for i in range(len(triples_a))}
    assert int(ledger_a.count) == 5

    triples_b = [
        (pv.OP_MUL, 1, 1),
        (pv.OP_SUC, 1, 0),
        (pv.OP_ADD, 1, 1),
    ]
    ledger_b = pv.init_ledger()
    ids_b, ledger_b = _intern_once(ledger_b, triples_b)
    mapping_b = {triples_b[i]: int(ids_b[i]) for i in range(len(triples_b))}
    assert int(ledger_b.count) == 5

    assert mapping_a == mapping_b
