import pytest

import prism_vm as pv
from tests import harness

i32 = harness.i32
intern_nodes = harness.intern_nodes
intern1 = harness.intern1

pytestmark = pytest.mark.m4


def test_coord_opcodes_exist():
    assert hasattr(pv, "OP_COORD_ZERO")
    assert hasattr(pv, "OP_COORD_ONE")
    assert hasattr(pv, "OP_COORD_PAIR")
    assert pv.OP_NAMES[pv.OP_COORD_ZERO] == "coord_zero"
    assert pv.OP_NAMES[pv.OP_COORD_ONE] == "coord_one"
    assert pv.OP_NAMES[pv.OP_COORD_PAIR] == "coord_pair"


def test_coord_pointer_equality():
    ledger = pv.init_ledger()
    ids, ledger = intern_nodes(
        ledger, [pv.OP_COORD_ZERO, pv.OP_COORD_ZERO], [0, 0], [0, 0]
    )
    assert int(ids[0]) == int(ids[1])


def test_coord_xor_parity_cancel():
    _require_coord_xor()
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    xor_id, ledger = pv.coord_xor(ledger, one_id, one_id)
    assert int(xor_id) == int(zero_id)
    xor_id, ledger = pv.coord_xor(ledger, zero_id, one_id)
    assert int(xor_id) == int(one_id)


def test_coord_leaf_requires_zero_args():
    ledger = pv.init_ledger()
    with pytest.raises(RuntimeError, match="key_safe_normalize.coord_leaf_args"):
        ids, _ = intern_nodes(
            ledger, [pv.OP_COORD_ZERO, pv.OP_COORD_ZERO], [0, 7], [0, 9]
        )
        ids.block_until_ready()


def test_coord_pair_dedup():
    ledger = pv.init_ledger()
    ids, ledger = intern_nodes(
        ledger, [pv.OP_COORD_ZERO, pv.OP_COORD_ONE], [0, 0], [0, 0]
    )
    zero_id = int(ids[0])
    one_id = int(ids[1])
    pair_ids, ledger = intern_nodes(
        ledger,
        [pv.OP_COORD_PAIR, pv.OP_COORD_PAIR],
        [zero_id, zero_id],
        [one_id, one_id],
    )
    assert int(pair_ids[0]) == int(pair_ids[1])
    assert int(ledger.count) == 5


def _coord_leaf(ledger, op):
    node_id, ledger = intern1(ledger, op, 0, 0)
    return int(node_id), ledger


def _require_coord_norm():
    assert hasattr(pv, "coord_norm"), "coord_norm not implemented"


def _require_coord_xor():
    assert hasattr(pv, "coord_xor"), "coord_xor not implemented"


def test_coord_norm_idempotent():
    _require_coord_norm()
    ledger = pv.init_ledger()
    left, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    right, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    coord_id, ledger = intern1(ledger, pv.OP_COORD_PAIR, left, right)
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


def test_coord_norm_commutes_with_xor():
    _require_coord_norm()
    _require_coord_xor()
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    pair_a, ledger = intern1(ledger, pv.OP_COORD_PAIR, one_id, zero_id)
    pair_b, ledger = intern1(ledger, pv.OP_COORD_PAIR, zero_id, one_id)
    xor_raw, ledger = pv.coord_xor(ledger, pair_a, pair_b)
    norm_raw, ledger = pv.coord_norm(ledger, xor_raw)
    norm_left, ledger = pv.coord_norm(ledger, pair_a)
    norm_right, ledger = pv.coord_norm(ledger, pair_b)
    xor_norm, ledger = pv.coord_xor(ledger, norm_left, norm_right)
    assert int(norm_raw) == int(xor_norm)


def test_coord_xor_distributes_over_pair():
    _require_coord_norm()
    _require_coord_xor()
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    pair_a, ledger = intern1(ledger, pv.OP_COORD_PAIR, one_id, zero_id)
    pair_b, ledger = intern1(ledger, pv.OP_COORD_PAIR, zero_id, one_id)
    xor_pair, ledger = pv.coord_xor(ledger, pair_a, pair_b)
    xor_left, ledger = pv.coord_xor(ledger, one_id, zero_id)
    xor_right, ledger = pv.coord_xor(ledger, zero_id, one_id)
    expected, ledger = intern1(ledger, pv.OP_COORD_PAIR, xor_left, xor_right)
    norm_xor, ledger = pv.coord_norm(ledger, xor_pair)
    norm_expected, ledger = pv.coord_norm(ledger, expected)
    assert int(norm_xor) == int(norm_expected)


def test_coord_xor_associative_small():
    _require_coord_norm()
    _require_coord_xor()
    ledger = pv.init_ledger()
    one_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ONE)
    zero_id, ledger = _coord_leaf(ledger, pv.OP_COORD_ZERO)
    a = one_id
    b = zero_id
    c = one_id
    ab, ledger = pv.coord_xor(ledger, a, b)
    left, ledger = pv.coord_xor(ledger, ab, c)
    bc, ledger = pv.coord_xor(ledger, b, c)
    right, ledger = pv.coord_xor(ledger, a, bc)
    norm_left, ledger = pv.coord_norm(ledger, left)
    norm_right, ledger = pv.coord_norm(ledger, right)
    assert int(norm_left) == int(norm_right)
