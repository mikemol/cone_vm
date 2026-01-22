import jax.numpy as jnp
import pytest

import prism_vm as pv


def test_coord_leaf_canonicalization():
    ledger = pv.init_ledger()
    ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_COORD_ZERO, pv.OP_COORD_ZERO], dtype=jnp.int32),
        jnp.array([0, 7], dtype=jnp.int32),
        jnp.array([0, 9], dtype=jnp.int32),
    )
    assert int(ids[0]) == int(ids[1])
    assert int(ledger.count) == 3


def test_coord_pair_dedup():
    ledger = pv.init_ledger()
    ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_COORD_ZERO, pv.OP_COORD_ONE], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    zero_id = int(ids[0])
    one_id = int(ids[1])
    pair_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_COORD_PAIR, pv.OP_COORD_PAIR], dtype=jnp.int32),
        jnp.array([zero_id, zero_id], dtype=jnp.int32),
        jnp.array([one_id, one_id], dtype=jnp.int32),
    )
    assert int(pair_ids[0]) == int(pair_ids[1])
    assert int(ledger.count) == 5


def _coord_leaf(ledger, op):
    ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([op], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    return int(ids[0]), ledger


def _require_coord_norm():
    if not hasattr(pv, "coord_norm"):
        pytest.xfail("coord_norm not implemented")


def _require_coord_xor():
    if not hasattr(pv, "coord_xor"):
        pytest.xfail("coord_xor not implemented")


def test_coord_norm_idempotent():
    _require_coord_norm()
    ledger = pv.init_ledger()
    left, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    right, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    pair_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_COORD_PAIR], dtype=jnp.int32),
        jnp.array([left], dtype=jnp.int32),
        jnp.array([right], dtype=jnp.int32),
    )
    coord_id = int(pair_ids[0])
    norm1, ledger = pv.coord_norm(ledger, coord_id)
    norm2, ledger = pv.coord_norm(ledger, norm1)
    assert int(norm1) == int(norm2)


def test_coord_norm_confluent_small():
    _require_coord_norm()
    _require_coord_xor()
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    xor_id, ledger = pv.coord_xor(ledger, one_id, one_id)
    norm_xor, ledger = pv.coord_norm(ledger, xor_id)
    norm_zero, ledger = pv.coord_norm(ledger, zero_id)
    assert int(norm_xor) == int(norm_zero)
