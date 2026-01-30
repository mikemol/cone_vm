import jax.numpy as jnp
import pytest

import prism_vm as pv
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3


def _build_sample_ledger():
    ledger = pv.init_ledger()
    suc0_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_SUC], [pv.ZERO_PTR], [0]
    )
    suc0 = int(suc0_ids[0])
    add_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_ADD], [pv.ZERO_PTR], [suc0]
    )
    add_id = int(add_ids[0])
    mul_ids, ledger = mph.intern_nodes(
        ledger, [pv.OP_MUL], [suc0], [suc0]
    )
    mul_id = int(mul_ids[0])
    _, ledger = mph.intern_nodes(ledger, [pv.OP_SUC], [add_id], [0])
    _, ledger = mph.intern_nodes(
        ledger, [pv.OP_ADD], [mul_id], [pv.ZERO_PTR]
    )
    return ledger


def test_min_prism_enumeration_rebuilds_canonical():
    keys = mph.enumerate_keys(max_depth=2)
    rebuilt, _ = mph.rebuild_ledger_from_keys(keys)
    canon_keys = set(mph.canon_state_ledger(rebuilt)[0])
    assert canon_keys == set(keys)


def test_min_prism_cqrs_rebuild_matches_state():
    ledger = _build_sample_ledger()
    keys = set(mph.canon_state_ledger(ledger)[0])
    rebuilt, _ = mph.rebuild_ledger_from_keys(keys)
    assert mph.canon_state_ledger(rebuilt) == mph.canon_state_ledger(ledger)
