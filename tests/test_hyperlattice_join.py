import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def _intern_set_a(ledger):
    suc0_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    suc0 = int(suc0_ids[0])
    _, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_ADD], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([suc0], dtype=jnp.int32),
    )
    return ledger


def _intern_set_b(ledger):
    mul_ids, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_MUL], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
        jnp.array([pv.ZERO_PTR], dtype=jnp.int32),
    )
    mul_id = int(mul_ids[0])
    _, ledger = pv.intern_nodes(
        ledger,
        jnp.array([pv.OP_SUC], dtype=jnp.int32),
        jnp.array([mul_id], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    return ledger


def test_hyperlattice_join_order_independent():
    ledger_ab = pv.init_ledger()
    ledger_ab = _intern_set_a(ledger_ab)
    ledger_ab = _intern_set_b(ledger_ab)

    ledger_ba = pv.init_ledger()
    ledger_ba = _intern_set_b(ledger_ba)
    ledger_ba = _intern_set_a(ledger_ba)

    state_ab = mph.canon_state_ledger(ledger_ab)
    state_ba = mph.canon_state_ledger(ledger_ba)
    assert state_ab == state_ba

    union_keys = set(state_ab[0])
    rebuilt, _ = mph.rebuild_ledger_from_keys(union_keys)
    assert mph.canon_state_ledger(rebuilt) == state_ab
