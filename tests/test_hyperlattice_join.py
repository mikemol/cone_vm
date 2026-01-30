import pytest

import prism_vm as pv
from tests import harness
from tests.min_prism import harness as mph

pytestmark = pytest.mark.m3

intern1 = harness.intern1


def _intern_set_a(ledger):
    suc0, ledger = intern1(ledger, pv.OP_SUC, pv.ZERO_PTR, 0)
    _, ledger = intern1(ledger, pv.OP_ADD, pv.ZERO_PTR, suc0)
    return ledger


def _intern_set_b(ledger):
    mul_id, ledger = intern1(ledger, pv.OP_MUL, pv.ZERO_PTR, pv.ZERO_PTR)
    _, ledger = intern1(ledger, pv.OP_SUC, mul_id, 0)
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
