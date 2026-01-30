import pytest

import prism_vm as pv
from tests import harness

intern1 = harness.intern1

pytestmark = pytest.mark.m4


def _coord_leaf(ledger, op):
    node_id, ledger = intern1(ledger, op, 0, 0)
    return int(node_id), ledger


def _coord_pair(ledger, left_id, right_id):
    node_id, ledger = intern1(ledger, pv.OP_COORD_PAIR, left_id, right_id)
    return int(node_id), ledger


def _intern_binary(ledger, op, left_id, right_id):
    node_id, ledger = intern1(ledger, op, left_id, right_id)
    return int(node_id), ledger


def test_cd_lift_add_over_pair():
    assert hasattr(pv, "cd_lift_binary")
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    pair_a, ledger = _coord_pair(ledger, one_id, zero_id)
    pair_b, ledger = _coord_pair(ledger, zero_id, one_id)

    lifted, ledger = pv.cd_lift_binary(ledger, pv.OP_ADD, pair_a, pair_b)
    left_id, ledger = _intern_binary(ledger, pv.OP_ADD, one_id, zero_id)
    right_id, ledger = _intern_binary(ledger, pv.OP_ADD, zero_id, one_id)
    expected, ledger = _coord_pair(ledger, left_id, right_id)
    assert int(lifted) == int(expected)


def test_cd_lift_mul_over_pair():
    assert hasattr(pv, "cd_lift_binary")
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    pair_a, ledger = _coord_pair(ledger, one_id, zero_id)
    pair_b, ledger = _coord_pair(ledger, zero_id, one_id)

    lifted, ledger = pv.cd_lift_binary(ledger, pv.OP_MUL, pair_a, pair_b)
    left_id, ledger = _intern_binary(ledger, pv.OP_MUL, one_id, zero_id)
    right_id, ledger = _intern_binary(ledger, pv.OP_MUL, zero_id, one_id)
    expected, ledger = _coord_pair(ledger, left_id, right_id)
    assert int(lifted) == int(expected)


def test_cd_lift_falls_back_to_op():
    assert hasattr(pv, "cd_lift_binary")
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    lifted, ledger = pv.cd_lift_binary(ledger, pv.OP_ADD, one_id, zero_id)
    expected, ledger = _intern_binary(ledger, pv.OP_ADD, one_id, zero_id)
    assert int(lifted) == int(expected)
