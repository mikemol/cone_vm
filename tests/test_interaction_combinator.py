import pytest
import jax.numpy as jnp

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
    assert table.rhs_node_type.shape == (0, ic.MAX_NEW_NODES)
    assert table.rhs_port_map.shape == (0, ic.MAX_NEW_NODES, ic.PORT_ARITY)
    assert table.ext_port_map.shape == (0, ic.EXT_COUNT)


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


def test_match_rules_symmetric_pair():
    state = ic.init_ic_state(2)
    port = state.port
    port = port.at[0, ic.PORT_P].set(ic.encode_port(1, ic.PORT_P))
    port = port.at[1, ic.PORT_P].set(ic.encode_port(0, ic.PORT_P))
    node_type = state.node_type
    node_type = node_type.at[0].set(ic.IC_DUP)
    node_type = node_type.at[1].set(ic.IC_CON)
    state = ic.ICState(node_type=node_type, port=port)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    count = jnp.int32(1)
    table = ic.RuleTable(
        lhs=jnp.array([[ic.IC_CON, ic.IC_DUP]], dtype=jnp.int8),
        alloc_count=jnp.array([0], dtype=jnp.int32),
        rhs_node_type=jnp.zeros((1, ic.MAX_NEW_NODES), dtype=jnp.int8),
        rhs_port_map=jnp.zeros((1, ic.MAX_NEW_NODES, ic.PORT_ARITY), dtype=jnp.int32),
        ext_port_map=jnp.zeros((1, ic.EXT_COUNT), dtype=jnp.int32),
    )
    rule_idx, matched, swapped = ic.match_active_pairs(state, pairs, count, table)
    assert int(rule_idx[0]) == 0
    assert bool(matched[0])
    assert bool(swapped[0])


def test_match_rules_empty_table():
    state = ic.init_ic_state(2)
    node_type = state.node_type.at[0].set(ic.IC_CON).at[1].set(ic.IC_DUP)
    state = ic.ICState(node_type=node_type, port=state.port)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    count = jnp.int32(1)
    table = ic.init_rule_table_empty()
    rule_idx, matched, swapped = ic.match_active_pairs(state, pairs, count, table)
    assert int(rule_idx[0]) == -1
    assert not bool(matched[0])
    assert not bool(swapped[0])


def test_match_rules_respects_count():
    state = ic.init_ic_state(3)
    node_type = state.node_type.at[0].set(ic.IC_CON).at[1].set(ic.IC_DUP)
    state = ic.ICState(node_type=node_type, port=state.port)
    pairs = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    count = jnp.int32(1)
    table = ic.RuleTable(
        lhs=jnp.array([[ic.IC_CON, ic.IC_DUP]], dtype=jnp.int8),
        alloc_count=jnp.array([0], dtype=jnp.int32),
        rhs_node_type=jnp.zeros((1, ic.MAX_NEW_NODES), dtype=jnp.int8),
        rhs_port_map=jnp.zeros((1, ic.MAX_NEW_NODES, ic.PORT_ARITY), dtype=jnp.int32),
        ext_port_map=jnp.zeros((1, ic.EXT_COUNT), dtype=jnp.int32),
    )
    rule_idx, matched, swapped = ic.match_active_pairs(state, pairs, count, table)
    assert int(rule_idx[0]) == 0
    assert bool(matched[0])
    assert int(rule_idx[1]) == -1
    assert not bool(matched[1])
    assert not bool(swapped[1])


def test_rule_table_core_commutation_template():
    table = ic.init_rule_table_core()
    ic.validate_rule_table(table)
    lhs = table.lhs
    match = (lhs[:, 0] == ic.IC_DUP) & (lhs[:, 1] == ic.IC_CON)
    idx = int(jnp.argmax(match))
    assert int(table.alloc_count[idx]) == 4
    assert list(table.rhs_node_type[idx]) == [
        ic.IC_CON,
        ic.IC_CON,
        ic.IC_DUP,
        ic.IC_DUP,
    ]
    assert list(table.rhs_port_map[idx, 0]) == [ic.EXT_A_L, 2, 3]
    assert list(table.rhs_port_map[idx, 1]) == [ic.EXT_A_R, 2, 3]
    assert list(table.rhs_port_map[idx, 2]) == [ic.EXT_B_L, 0, 1]
    assert list(table.rhs_port_map[idx, 3]) == [ic.EXT_B_R, 0, 1]
    assert list(table.ext_port_map[idx]) == [0, 1, 2, 3]
