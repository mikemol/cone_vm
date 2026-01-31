import pytest

import prism_vm as pv
from tests import harness

intern1 = harness.intern1
committed_ids = harness.committed_ids

pytestmark = pytest.mark.m4


def _coord_leaf(ledger, op):
    node_id, ledger = intern1(ledger, op, 0, 0)
    return int(node_id), ledger


def _coord_frontier_with_op(op):
    ledger = pv.init_ledger()
    left, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    right, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    root_id, ledger = intern1(ledger, op, left, right)
    frontier = committed_ids(root_id)
    return ledger, frontier, left, right, root_id


def test_coord_add_aggregates_in_cycle_candidates():
    _require_cycle_candidates()
    ledger, frontier, left, right, _ = _coord_frontier_with_op(pv.OP_ADD)
    ledger, next_frontier_prov, _, q_map = harness.cycle_candidates_static_bound(
        ledger, frontier
    )
    next_id = int(pv.apply_q(q_map, next_frontier_prov).a[0])
    expected_id, _ = pv.coord_xor(ledger, left, right)
    assert next_id == int(expected_id)


def test_coord_mul_does_not_aggregate():
    _require_cycle_candidates()
    ledger, frontier, _, _, root_id = _coord_frontier_with_op(pv.OP_MUL)
    ledger, next_frontier_prov, _, q_map = harness.cycle_candidates_static_bound(
        ledger, frontier
    )
    next_id = int(pv.apply_q(q_map, next_frontier_prov).a[0])
    assert next_id == root_id


def _require_cycle_candidates():
    assert hasattr(pv, "cycle_candidates")
