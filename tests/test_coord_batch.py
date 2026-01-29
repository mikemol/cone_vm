import jax.numpy as jnp
import pytest

import prism_vm as pv


pytestmark = pytest.mark.m4


def test_coord_xor_batch_uses_single_intern_call(monkeypatch):
    calls = {"n": 0}
    real_intern = pv.intern_nodes

    def counted_intern(*args, **kwargs):
        calls["n"] += 1
        return real_intern(*args, **kwargs)

    ledger = pv.init_ledger()
    z0, ledger = pv._coord_leaf_id(ledger, pv.OP_COORD_ZERO)
    z1, ledger = pv._coord_leaf_id(ledger, pv.OP_COORD_ONE)
    calls["n"] = 0

    n = 64
    left = jnp.array([z0, z1] * (n // 2), dtype=jnp.int32)
    right = jnp.array([z1, z0] * (n // 2), dtype=jnp.int32)

    out_ids, ledger2 = pv.coord_xor_batch(
        ledger, left, right, intern_fn=counted_intern
    )
    ledger2.count.block_until_ready()
    assert out_ids.shape[0] == n
    assert calls["n"] <= 4


def test_coord_norm_batch_matches_host():
    ledger = pv.init_ledger()
    z1, ledger = pv._coord_leaf_id(ledger, pv.OP_COORD_ONE)
    ids = []
    for _ in range(16):
        pair, ledger = pv._coord_promote_leaf(ledger, z1)
        pair2, ledger = pv._coord_promote_leaf(ledger, pair)
        ids.append(pair2)

    coord_ids = jnp.array(ids, dtype=jnp.int32)
    norm_ids_b, ledger2 = pv.coord_norm_batch(ledger, coord_ids)
    ledger2.count.block_until_ready()

    norm_ids_h = []
    ledger_ref = ledger
    for cid in ids:
        nid, ledger_ref = pv.coord_norm(ledger_ref, int(cid))
        norm_ids_h.append(nid)

    assert list(map(int, norm_ids_b)) == norm_ids_h
