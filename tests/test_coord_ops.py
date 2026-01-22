import jax.numpy as jnp

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
