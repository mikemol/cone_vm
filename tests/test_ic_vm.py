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


def test_ic_apply_commute_rewires_ports():
    state = ic.ic_init(10)
    state, ext = ic.ic_alloc(state, 4, ic.TYPE_CON)
    ext_left, ext_right, ext_dup_left, ext_dup_right = map(int, ext)
    state, (con_node,) = ic.ic_alloc(state, 1, ic.TYPE_CON)
    state, (dup_node,) = ic.ic_alloc(state, 1, ic.TYPE_DUP)
    con_node = int(con_node)
    dup_node = int(dup_node)
    state = ic.ic_wire(state, con_node, ic.PORT_PRINCIPAL, dup_node, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, con_node, ic.PORT_AUX_LEFT, ext_left, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, con_node, ic.PORT_AUX_RIGHT, ext_right, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, dup_node, ic.PORT_AUX_LEFT, ext_dup_left, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, dup_node, ic.PORT_AUX_RIGHT, ext_dup_right, ic.PORT_PRINCIPAL)
    before_top = int(state.free_top)
    expected_con = state.free_stack[before_top - 2:before_top]
    expected_dup = state.free_stack[before_top - 4:before_top - 2]
    state = ic.ic_apply_commute(state, con_node, dup_node)
    assert int(state.node_type[con_node]) == int(ic.TYPE_FREE)
    assert int(state.node_type[dup_node]) == int(ic.TYPE_FREE)
    assert list(map(int, state.ports[con_node])) == [0, 0, 0]
    assert list(map(int, state.ports[dup_node])) == [0, 0, 0]
    c0, c1 = map(int, expected_con)
    d0, d1 = map(int, expected_dup)
    node, port = ic.decode_port(state.ports[c0, 0])
    assert int(node) == d0
    assert int(port) == int(ic.PORT_PRINCIPAL)
    node, port = ic.decode_port(state.ports[c1, 0])
    assert int(node) == d1
    assert int(port) == int(ic.PORT_PRINCIPAL)
    node, port = ic.decode_port(state.ports[ext_dup_left, 0])
    assert int(node) == c0
    assert int(port) == int(ic.PORT_AUX_LEFT)
    node, port = ic.decode_port(state.ports[ext_dup_right, 0])
    assert int(node) == c1
    assert int(port) == int(ic.PORT_AUX_LEFT)
    node, port = ic.decode_port(state.ports[ext_left, 0])
    assert int(node) == d0
    assert int(port) == int(ic.PORT_AUX_LEFT)
    node, port = ic.decode_port(state.ports[ext_right, 0])
    assert int(node) == d1
    assert int(port) == int(ic.PORT_AUX_LEFT)
    node, port = ic.decode_port(state.ports[c0, ic.PORT_AUX_RIGHT])
    assert int(node) == d0
    assert int(port) == int(ic.PORT_AUX_RIGHT)
    node, port = ic.decode_port(state.ports[c1, ic.PORT_AUX_RIGHT])
    assert int(node) == d1
    assert int(port) == int(ic.PORT_AUX_RIGHT)


def test_ic_apply_active_pairs_batch():
    state = ic.ic_init(12)
    node_type = state.node_type.at[:12].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    # Pair 1
    state = ic.ic_wire(state, 0, ic.PORT_PRINCIPAL, 1, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 0, ic.PORT_AUX_LEFT, 2, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 0, ic.PORT_AUX_RIGHT, 3, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 1, ic.PORT_AUX_LEFT, 4, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 1, ic.PORT_AUX_RIGHT, 5, ic.PORT_PRINCIPAL)
    # Pair 2
    state = ic.ic_wire(state, 6, ic.PORT_PRINCIPAL, 7, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 6, ic.PORT_AUX_LEFT, 8, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 6, ic.PORT_AUX_RIGHT, 9, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 7, ic.PORT_AUX_LEFT, 10, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 7, ic.PORT_AUX_RIGHT, 11, ic.PORT_PRINCIPAL)
    state, stats = ic.ic_apply_active_pairs(state)
    assert int(stats.active_pairs) == 2
    assert int(stats.template_counts[ic.TEMPLATE_ANNIHILATE]) == 2
    for node in (0, 1, 6, 7):
        assert int(state.node_type[node]) == int(ic.TYPE_FREE)
    n2, p2 = ic.decode_port(state.ports[2, 0])
    n4, p4 = ic.decode_port(state.ports[4, 0])
    assert int(n2) == 4
    assert int(p2) == int(ic.PORT_PRINCIPAL)
    assert int(n4) == 2
    assert int(p4) == int(ic.PORT_PRINCIPAL)
    n3, p3 = ic.decode_port(state.ports[3, 0])
    n5, p5 = ic.decode_port(state.ports[5, 0])
    assert int(n3) == 5
    assert int(p3) == int(ic.PORT_PRINCIPAL)
    assert int(n5) == 3
    assert int(p5) == int(ic.PORT_PRINCIPAL)


def test_ic_apply_active_pairs_mixed_templates():
    state = ic.ic_init(20)
    state, ext = ic.ic_alloc(state, 4, ic.TYPE_CON)
    ext_left, ext_right, ext_dup_left, ext_dup_right = map(int, ext)
    state, (con_node,) = ic.ic_alloc(state, 1, ic.TYPE_CON)
    state, (dup_node,) = ic.ic_alloc(state, 1, ic.TYPE_DUP)
    con_node = int(con_node)
    dup_node = int(dup_node)
    state = ic.ic_wire(state, con_node, ic.PORT_PRINCIPAL, dup_node, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, con_node, ic.PORT_AUX_LEFT, ext_left, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, con_node, ic.PORT_AUX_RIGHT, ext_right, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, dup_node, ic.PORT_AUX_LEFT, ext_dup_left, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, dup_node, ic.PORT_AUX_RIGHT, ext_dup_right, ic.PORT_PRINCIPAL)
    node_type = state.node_type
    node_type = node_type.at[8].set(ic.TYPE_CON)
    node_type = node_type.at[9].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    state = ic.ic_wire(state, 8, ic.PORT_PRINCIPAL, 9, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 8, ic.PORT_AUX_LEFT, 10, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 8, ic.PORT_AUX_RIGHT, 11, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 9, ic.PORT_AUX_LEFT, 12, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 9, ic.PORT_AUX_RIGHT, 13, ic.PORT_PRINCIPAL)
    state, stats = ic.ic_apply_active_pairs(state)
    assert int(stats.active_pairs) == 2
    assert int(stats.template_counts[ic.TEMPLATE_ANNIHILATE]) == 1
    assert int(stats.template_counts[ic.TEMPLATE_COMMUTE]) == 1
    assert int(stats.alloc_nodes) == 4
    assert int(stats.freed_nodes) == 4
