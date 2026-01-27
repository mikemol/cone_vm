import pytest

import interaction_combinator as ic

pytestmark = pytest.mark.m6


def test_init_ic_state_shapes():
    state = ic.init_ic_state(8)
    ic.validate_ic_state(state)
    assert state.node_type.shape == (8,)
    assert state.port.shape == (8, ic.PORT_ARITY)


def test_rule_table_empty_valid():
    table = ic.init_rule_table_empty()
    ic.validate_rule_table(table)
    assert table.lhs.shape == (0, 2)
    assert table.rhs_node_type.shape == (0, 2)
    assert table.rhs_port_map.shape == (0, 2, ic.PORT_ARITY)


def test_port_encode_decode_roundtrip():
    ref = ic.encode_port(7, ic.PORT_R)
    node, port = ic.decode_port(ref)
    assert node == 7
    assert port == ic.PORT_R
