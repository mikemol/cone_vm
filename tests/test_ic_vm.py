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


def test_ic_wire_jax_roundtrip():
    state = ic.ic_init(4)
    state = ic.ic_wire_jax(state, jnp.uint32(1), jnp.uint32(0), jnp.uint32(2), jnp.uint32(0))
    n1, p1 = ic.decode_port(state.ports[1, 0])
    n2, p2 = ic.decode_port(state.ports[2, 0])
    assert int(n1) == 2
    assert int(p1) == int(ic.PORT_PRINCIPAL)
    assert int(n2) == 1
    assert int(p2) == int(ic.PORT_PRINCIPAL)


def test_ic_wire_jax_safe_noop():
    state = ic.ic_init(4)
    ports_before = state.ports
    state = ic.ic_wire_jax_safe(
        state, jnp.uint32(0), jnp.uint32(0), jnp.uint32(2), jnp.uint32(0)
    )
    assert jnp.array_equal(state.ports, ports_before)


def test_ic_wire_pairs_jax_basic():
    state = ic.ic_init(6)
    state = ic.ic_wire_pairs_jax(
        state,
        jnp.array([1, 3], dtype=jnp.uint32),
        jnp.array([0, 1], dtype=jnp.uint32),
        jnp.array([2, 4], dtype=jnp.uint32),
        jnp.array([0, 2], dtype=jnp.uint32),
    )
    n1, p1 = ic.decode_port(state.ports[1, 0])
    n2, p2 = ic.decode_port(state.ports[2, 0])
    assert int(n1) == 2
    assert int(p1) == int(ic.PORT_PRINCIPAL)
    assert int(n2) == 1
    assert int(p2) == int(ic.PORT_PRINCIPAL)
    n3, p3 = ic.decode_port(state.ports[3, 1])
    n4, p4 = ic.decode_port(state.ports[4, 2])
    assert int(n3) == 4
    assert int(p3) == int(ic.PORT_AUX_RIGHT)
    assert int(n4) == 3
    assert int(p4) == int(ic.PORT_AUX_LEFT)


def test_ic_wire_ptr_pairs_jax_null_skip():
    state = ic.ic_init(4)
    ptr_a = jnp.array(
        [
            ic.encode_port(jnp.uint32(1), jnp.uint32(0)),
            jnp.uint32(0),
        ],
        dtype=jnp.uint32,
    )
    ptr_b = jnp.array(
        [
            ic.encode_port(jnp.uint32(2), jnp.uint32(0)),
            ic.encode_port(jnp.uint32(3), jnp.uint32(0)),
        ],
        dtype=jnp.uint32,
    )
    state = ic.ic_wire_ptr_pairs_jax(state, ptr_a, ptr_b)
    n1, p1 = ic.decode_port(state.ports[1, 0])
    n2, p2 = ic.decode_port(state.ports[2, 0])
    assert int(n1) == 2
    assert int(p1) == int(ic.PORT_PRINCIPAL)
    assert int(n2) == 1
    assert int(p2) == int(ic.PORT_PRINCIPAL)
    assert int(state.ports[3, 0]) == 0


def test_ic_wire_star_jax_leafs_connect():
    state = ic.ic_init(5)
    leaf_nodes = jnp.array([2, 3], dtype=jnp.uint32)
    leaf_ports = jnp.array([0, 0], dtype=jnp.uint32)
    state = ic.ic_wire_star_jax(
        state, jnp.uint32(1), jnp.uint32(0), leaf_nodes, leaf_ports
    )
    for leaf in (2, 3):
        node, port = ic.decode_port(state.ports[leaf, 0])
        assert int(node) == 1
        assert int(port) == int(ic.PORT_PRINCIPAL)


def test_ic_init_free_stack():
    state = ic.ic_init(4)
    assert state.free_top == 3
    assert list(map(int, state.free_stack)) == [3, 2, 1, 0]
    assert state.ports.shape == (4, 3)


def test_ic_find_active_pairs_principal_link():
    state = ic.ic_init(3)
    state = ic.ic_wire(state, 1, 0, 2, 0)
    pairs, active = ic.ic_find_active_pairs(state)
    assert bool(active[1])
    assert int(pairs[0]) == 1


def test_ic_corrupt_on_port3_halts():
    state = ic.ic_init(4)
    bad_ptr = ic.encode_port(jnp.uint32(1), jnp.uint32(3))
    ports = state.ports.at[1, ic.PORT_PRINCIPAL].set(bad_ptr)
    state = state._replace(ports=ports)
    state2, stats = ic.ic_apply_active_pairs(state)
    assert bool(state2.corrupt)
    assert int(stats.active_pairs) == 0
    assert int(state2.ports[1, ic.PORT_PRINCIPAL]) == int(bad_ptr)


def test_ic_compact_active_pairs_empty():
    state = ic.ic_init(3)
    compacted, count, active = ic.ic_compact_active_pairs(state)
    assert int(count) == 0
    assert not bool(active[0])
    assert list(map(int, compacted)) == [0, 0, 0]


def test_ic_compact_active_pairs_multiple():
    state = ic.ic_init(5)
    state = ic.ic_wire(state, 1, 0, 2, 0)
    state = ic.ic_wire(state, 3, 0, 4, 0)
    compacted, count, active = ic.ic_compact_active_pairs(state)
    assert int(count) == 2
    assert bool(active[1])
    assert bool(active[3])
    assert list(map(int, compacted[:2])) == [1, 3]


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
    state = ic.ic_init(3)
    node_type = state.node_type
    node_type = node_type.at[1].set(ic.TYPE_CON)
    node_type = node_type.at[2].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    tmpl = ic.ic_select_template(state, 1, 2)
    assert int(tmpl) == int(ic.TEMPLATE_ANNIHILATE)


def test_ic_apply_annihilate_rewires_aux():
    state = ic.ic_init(5)
    node_type = state.node_type
    node_type = node_type.at[1].set(ic.TYPE_CON)
    node_type = node_type.at[2].set(ic.TYPE_CON)
    node_type = node_type.at[3].set(ic.TYPE_CON)
    node_type = node_type.at[4].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    state = ic.ic_wire(state, 1, ic.PORT_PRINCIPAL, 2, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 1, ic.PORT_AUX_LEFT, 3, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 2, ic.PORT_AUX_LEFT, 4, ic.PORT_PRINCIPAL)
    state = ic.ic_apply_annihilate(state, 1, 2)
    n2, p2 = ic.decode_port(state.ports[3, 0])
    n3, p3 = ic.decode_port(state.ports[4, 0])
    assert int(n2) == 4
    assert int(p2) == int(ic.PORT_PRINCIPAL)
    assert int(n3) == 3
    assert int(p3) == int(ic.PORT_PRINCIPAL)


def test_ic_apply_erase_frees_nodes():
    state = ic.ic_init(7)
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
    state = ic.ic_init(7)
    state, (left,) = ic.ic_alloc(state, 1, ic.TYPE_CON)
    state, (right,) = ic.ic_alloc(state, 1, ic.TYPE_DUP)
    before_top = int(state.free_top)
    state = ic.ic_apply_commute(state, int(left), int(right))
    assert int(state.free_top) == before_top - 2


def test_ic_alloc_underflow_sets_oom():
    state = ic.ic_init(3)
    state, _ = ic.ic_alloc(state, 2, ic.TYPE_CON)
    state, nodes = ic.ic_alloc(state, 1, ic.TYPE_CON)
    assert bool(state.oom)
    assert list(map(int, nodes)) == [0]


def test_ic_apply_commute_rewires_ports():
    state = ic.ic_init(12)
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
    state = ic.ic_init(13)
    node_type = state.node_type.at[:13].set(ic.TYPE_CON)
    state = state._replace(node_type=node_type)
    # Pair 1
    state = ic.ic_wire(state, 1, ic.PORT_PRINCIPAL, 2, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 1, ic.PORT_AUX_LEFT, 3, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 1, ic.PORT_AUX_RIGHT, 4, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 2, ic.PORT_AUX_LEFT, 5, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 2, ic.PORT_AUX_RIGHT, 6, ic.PORT_PRINCIPAL)
    # Pair 2
    state = ic.ic_wire(state, 7, ic.PORT_PRINCIPAL, 8, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 7, ic.PORT_AUX_LEFT, 9, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 7, ic.PORT_AUX_RIGHT, 10, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 8, ic.PORT_AUX_LEFT, 11, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, 8, ic.PORT_AUX_RIGHT, 12, ic.PORT_PRINCIPAL)
    state, stats = ic.ic_apply_active_pairs(state)
    assert int(stats.active_pairs) == 2
    assert int(stats.template_counts[ic.TEMPLATE_ANNIHILATE]) == 2
    for node in (1, 2, 7, 8):
        assert int(state.node_type[node]) == int(ic.TYPE_FREE)
    n2, p2 = ic.decode_port(state.ports[3, 0])
    n4, p4 = ic.decode_port(state.ports[5, 0])
    assert int(n2) == 5
    assert int(p2) == int(ic.PORT_PRINCIPAL)
    assert int(n4) == 3
    assert int(p4) == int(ic.PORT_PRINCIPAL)
    n3, p3 = ic.decode_port(state.ports[4, 0])
    n5, p5 = ic.decode_port(state.ports[6, 0])
    assert int(n3) == 6
    assert int(p3) == int(ic.PORT_PRINCIPAL)
    assert int(n5) == 4
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
    state, ann_ext = ic.ic_alloc(state, 4, ic.TYPE_CON)
    ann_left, ann_right, ann_dup_left, ann_dup_right = map(int, ann_ext)
    state, pair_nodes = ic.ic_alloc(state, 2, ic.TYPE_CON)
    left_node, right_node = map(int, pair_nodes)
    state = ic.ic_wire(state, left_node, ic.PORT_PRINCIPAL, right_node, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, left_node, ic.PORT_AUX_LEFT, ann_left, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, left_node, ic.PORT_AUX_RIGHT, ann_right, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, right_node, ic.PORT_AUX_LEFT, ann_dup_left, ic.PORT_PRINCIPAL)
    state = ic.ic_wire(state, right_node, ic.PORT_AUX_RIGHT, ann_dup_right, ic.PORT_PRINCIPAL)
    state, stats = ic.ic_apply_active_pairs(state)
    assert int(stats.active_pairs) == 2
    assert int(stats.template_counts[ic.TEMPLATE_ANNIHILATE]) == 1
    assert int(stats.template_counts[ic.TEMPLATE_COMMUTE]) == 1
    assert int(stats.alloc_nodes) == 4
    assert int(stats.freed_nodes) == 4


def test_ic_apply_active_pairs_oom_on_alloc():
    state = ic.ic_init(8)
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
    free_top_before = int(state.free_top)
    state2, stats = ic.ic_apply_active_pairs(state)
    assert bool(state2.oom)
    assert int(state2.free_top) == free_top_before
    assert int(state2.node_type[con_node]) == int(ic.TYPE_CON)
    assert int(state2.node_type[dup_node]) == int(ic.TYPE_DUP)
    assert int(stats.active_pairs) == 0
    assert int(stats.alloc_nodes) == 0
    assert int(stats.freed_nodes) == 0


def test_ic_alloc_jax_success():
    state = ic.ic_init(6)
    state2, ids4, ok = ic.ic_alloc_jax(state, jnp.int32(2), ic.TYPE_CON)
    assert bool(ok)
    assert int(state2.free_top) == int(state.free_top) - 2
    assert int(ids4[0]) != 0
    assert int(ids4[1]) != 0
    assert int(ids4[2]) == 0
    assert int(ids4[3]) == 0
    for node in (int(ids4[0]), int(ids4[1])):
        assert int(state2.node_type[node]) == int(ic.TYPE_CON)


def test_ic_alloc_jax_fail_no_oom():
    state = ic.ic_init(3)
    state2, ids4, ok = ic.ic_alloc_jax(state, jnp.int32(4), ic.TYPE_CON)
    assert not bool(ok)
    assert int(state2.free_top) == int(state.free_top)
    assert not bool(state2.oom)
    assert int(ids4[0]) == 0


def test_ic_alloc_jax_fail_sets_oom():
    state = ic.ic_init(3)
    state2, _, ok = ic.ic_alloc_jax(
        state, jnp.int32(4), ic.TYPE_CON, set_oom_on_fail=True
    )
    assert not bool(ok)
    assert bool(state2.oom)
