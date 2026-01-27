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


def test_init_ic_arena_shapes():
    arena = ic.init_ic_arena(5)
    ic.validate_ic_arena(arena)
    assert arena.free_stack.shape == (5,)


def test_find_active_pairs_principal_only():
    state = ic.init_ic_state(2)
    port = state.port
    port = port.at[0, ic.PORT_P].set(ic.encode_port(1, ic.PORT_P))
    port = port.at[1, ic.PORT_P].set(ic.encode_port(0, ic.PORT_P))
    node_type = state.node_type
    node_type = node_type.at[0].set(ic.IC_CON)
    node_type = node_type.at[1].set(ic.IC_DUP)
    state = ic.ICState(node_type=node_type, port=port)
    pairs, count = ic.find_active_pairs(state)
    assert int(count) == 1
    left, right = (int(pairs[0, 0]), int(pairs[0, 1]))
    assert {left, right} == {0, 1}


def test_find_active_pairs_ignores_aux_links():
    state = ic.init_ic_state(2)
    port = state.port
    port = port.at[0, ic.PORT_L].set(ic.encode_port(1, ic.PORT_P))
    port = port.at[1, ic.PORT_P].set(ic.encode_port(0, ic.PORT_L))
    node_type = state.node_type
    node_type = node_type.at[0].set(ic.IC_CON)
    node_type = node_type.at[1].set(ic.IC_DUP)
    state = ic.ICState(node_type=node_type, port=port)
    pairs, count = ic.find_active_pairs(state)
    assert int(count) == 0
