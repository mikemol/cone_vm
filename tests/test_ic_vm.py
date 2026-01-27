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


def test_ic_select_template_annihilate():
    state = ic.ic_init(2)
    node_type = state.node_type
    node_type = node_type.at[0].set(ic.TYPE_CON)
    node_type = node_type.at[1].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    tmpl = ic.ic_select_template(state, 0, 1)
    assert int(tmpl) == int(ic.TEMPLATE_ANNIHILATE)


def test_ic_apply_annihilate_rewires_aux():
    state = ic.ic_init(4)
    node_type = state.node_type
    node_type = node_type.at[0].set(ic.TYPE_CON)
    node_type = node_type.at[1].set(ic.TYPE_CON)
    node_type = node_type.at[2].set(ic.TYPE_CON)
    node_type = node_type.at[3].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    state = ic.ic_wire(state, 0, ic.PORT_PRINCIPAL, 1, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 0, ic.PORT_AUX_LEFT, 2, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 1, ic.PORT_AUX_LEFT, 3, ic.PORT_PRINCIPAL)
    state = ic.ic_apply_annihilate(state, 0, 1)
    n2, p2 = ic.decode_port(state.ports[2, 0])
    n3, p3 = ic.decode_port(state.ports[3, 0])
    assert int(n2) == 3
    assert int(p2) == int(ic.PORT_PRINCIPAL)
    assert int(n3) == 2
    assert int(p3) == int(ic.PORT_PRINCIPAL)


def test_ic_apply_erase_frees_nodes():
    state = ic.ic_init(6)
    state, (bin_node,) = ic.ic_alloc(state, 1, ic.TYPE_CON)
    state, aux_nodes = ic.ic_alloc(state, 2, ic.TYPE_CON)
    state, (era_node,) = ic.ic_alloc(state, 1, ic.TYPE_ERA)
    aux_left, aux_right = map(int, aux_nodes)
    bin_node = int(bin_node)
    era_node = int(era_node)
    state = ic.ic_wire(state, bin_node, ic.PORT_PRINCIPAL, era_node, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, bin_node, ic.PORT_AUX_LEFT, aux_left, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, bin_node, ic.PORT_AUX_RIGHT, aux_right, ic.PORT_PRINCIPAL)
    free_top_before = int(state.free_top)
    state = ic.ic_apply_erase(state, era_node, bin_node)
    assert int(state.free_top) == free_top_before
    era_indices = jnp.nonzero(state.node_type == ic.TYPE_ERA, size=2)[0]
    assert int(era_indices[0]) != era_node
    assert int(era_indices[1]) != era_node
    for aux in (aux_left, aux_right):
        node, port = ic.decode_port(state.ports[aux, 0])
        assert int(port) == int(ic.PORT_PRINCIPAL)
        assert int(node) in {int(era_indices[0]), int(era_indices[1])}


def test_ic_apply_commute_allocates_nodes():
    state = ic.ic_init(6)
    state, (left,) = ic.ic_alloc(state, 1, ic.TYPE_CON)
    state, (right,) = ic.ic_alloc(state, 1, ic.TYPE_DUP)
    before_top = int(state.free_top)
    state = ic.ic_apply_commute(state, int(left), int(right))
    assert int(state.free_top) == before_top - 4


def test_ic_alloc_underflow_raises():
    state = ic.ic_init(2)
    state, _ = ic.ic_alloc(state, 2, ic.TYPE_CON)
    with pytest.raises(ValueError):
        ic.ic_alloc(state, 1, ic.TYPE_CON)
