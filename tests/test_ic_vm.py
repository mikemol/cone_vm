import jax.numpy as jnp
import pytest

import ic_vm as ic

pytestmark = pytest.mark.m6


def test_ic_port_encode_decode_roundtrip():
    node = jnp.uint32(7)
    port = jnp.uint32(2)
    ptr = ic.encode_port(node, port)
    out_node, out_port = ic.decode_port(ptr)
    assert int(out_node) == int(node)
    assert int(out_port) == int(port)


def test_ic_init_free_stack():
    state = ic.ic_init(4)
    assert state.free_top == 4
    assert list(map(int, state.free_stack)) == [3, 2, 1, 0]
    assert state.ports.shape == (4, 3)


def test_ic_find_active_pairs_principal_link():
    state = ic.ic_init(2)
    state = ic.ic_wire(state, 0, 0, 1, 0)
    pairs, active = ic.ic_find_active_pairs(state)
    assert bool(active[0])
    assert int(pairs[0]) == 0


def test_ic_compact_active_pairs_empty():
    state = ic.ic_init(3)
    compacted, count, active = ic.ic_compact_active_pairs(state)
    assert int(count) == 0
    assert not bool(active[0])
    assert list(map(int, compacted)) == [0, 0, 0]


def test_ic_compact_active_pairs_multiple():
    state = ic.ic_init(4)
    state = ic.ic_wire(state, 0, 0, 1, 0)
    state = ic.ic_wire(state, 2, 0, 3, 0)
    compacted, count, active = ic.ic_compact_active_pairs(state)
    assert int(count) == 2
    assert bool(active[0])
    assert bool(active[2])
    assert list(map(int, compacted[:2])) == [0, 2]


def test_ic_rule_table_alloc_counts():
    alloc_same = ic.ic_rule_for_types(ic.TYPE_CON, ic.TYPE_CON)
    alloc_commute = ic.ic_rule_for_types(ic.TYPE_CON, ic.TYPE_DUP)
    alloc_erase = ic.ic_rule_for_types(ic.TYPE_ERA, ic.TYPE_CON)
    assert int(alloc_same[0]) == int(ic.RULE_ALLOC_ANNIHILATE)
    assert int(alloc_commute[0]) == int(ic.RULE_ALLOC_COMMUTE)
    assert int(alloc_erase[0]) == int(ic.RULE_ALLOC_ERASE)


def test_ic_rule_table_commutes_symmetry():
    left = ic.ic_rule_for_types(ic.TYPE_CON, ic.TYPE_DUP)
    right = ic.ic_rule_for_types(ic.TYPE_DUP, ic.TYPE_CON)
    assert tuple(map(int, left)) == tuple(map(int, right))
