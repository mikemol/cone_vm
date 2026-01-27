import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple

# Interaction-combinator (IC) backend scaffold (in-8).

TYPE_FREE = jnp.uint8(0)
TYPE_ERA = jnp.uint8(1)
TYPE_CON = jnp.uint8(2)
TYPE_DUP = jnp.uint8(3)

PORT_PRINCIPAL = jnp.uint32(0)
PORT_AUX_LEFT = jnp.uint32(1)
PORT_AUX_RIGHT = jnp.uint32(2)

RULE_ALLOC_ANNIHILATE = jnp.uint32(0)
RULE_ALLOC_ERASE = jnp.uint32(2)
RULE_ALLOC_COMMUTE = jnp.uint32(4)

TEMPLATE_NONE = jnp.uint32(0)
TEMPLATE_ANNIHILATE = jnp.uint32(1)
TEMPLATE_ERASE = jnp.uint32(2)
TEMPLATE_COMMUTE = jnp.uint32(3)

RULE_TABLE = jnp.array(
    [
        # FREE with anything is no-op.
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        # ERA interactions.
        [[0, 0], [0, 1], [2, 2], [2, 2]],
        # CON interactions.
        [[0, 0], [2, 2], [0, 1], [4, 3]],
        # DUP interactions.
        [[0, 0], [2, 2], [4, 3], [0, 1]],
    ],
    dtype=jnp.uint32,
)


def _connect_ptrs(
    ports: jnp.ndarray, ptr_a: jnp.ndarray, ptr_b: jnp.ndarray
) -> jnp.ndarray:
    ptr_a = jnp.asarray(ptr_a, dtype=jnp.uint32)
    ptr_b = jnp.asarray(ptr_b, dtype=jnp.uint32)
    do = (ptr_a != jnp.uint32(0)) & (ptr_b != jnp.uint32(0))

    def _do(p):
        node_a, port_a = decode_port(ptr_a)
        node_b, port_b = decode_port(ptr_b)
        p = p.at[node_a, port_a].set(ptr_b)
        p = p.at[node_b, port_b].set(ptr_a)
        return p

    return jax.lax.cond(do, _do, lambda p: p, ports)


class ICState(NamedTuple):
    node_type: jnp.ndarray
    ports: jnp.ndarray
    free_stack: jnp.ndarray
    free_top: jnp.ndarray
    oom: jnp.ndarray


class ICRewriteStats(NamedTuple):
    active_pairs: jnp.ndarray
    alloc_nodes: jnp.ndarray
    freed_nodes: jnp.ndarray
    template_counts: jnp.ndarray


def _safe_uint32(value) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.uint32)


def _alloc_raw(state: ICState, count: int) -> Tuple[ICState, jnp.ndarray, jnp.ndarray]:
    top = state.free_top.astype(jnp.int32)
    ok = (top >= count) & (~state.oom)

    def _do(s):
        start = top - count
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (count,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32), jnp.bool_(True)

    def _fail(s):
        return s, jnp.zeros((count,), dtype=jnp.uint32), jnp.bool_(False)

    return jax.lax.cond(ok, _do, _fail, state)


def _alloc_pad(state: ICState, count: int) -> Tuple[ICState, jnp.ndarray, jnp.ndarray]:
    if count > state.free_stack.shape[0]:
        return state, jnp.zeros((4,), dtype=jnp.uint32), jnp.bool_(False)
    state2, ids, ok = _alloc_raw(state, count)
    if count == 0:
        ids4 = jnp.zeros((4,), dtype=jnp.uint32)
    elif count == 1:
        ids4 = jnp.concatenate([ids, jnp.zeros((3,), dtype=jnp.uint32)], axis=0)
    elif count == 2:
        ids4 = jnp.concatenate([ids, jnp.zeros((2,), dtype=jnp.uint32)], axis=0)
    elif count == 4:
        ids4 = ids
    else:
        ids4 = jnp.zeros((4,), dtype=jnp.uint32)
        ok = jnp.bool_(False)
    return state2, ids4, ok


def _init_nodes_jax(
    state: ICState, ids4: jnp.ndarray, count: jnp.ndarray, node_type: jnp.uint8
) -> ICState:
    mask = jnp.arange(4, dtype=jnp.int32) < count.astype(jnp.int32)
    ids_safe = ids4
    node_type_curr = state.node_type[ids_safe]
    node_type_update = jnp.where(mask, node_type, node_type_curr)
    node_type_arr = state.node_type.at[ids_safe].set(node_type_update)
    ports_curr = state.ports[ids_safe]
    ports_update = jnp.where(mask[:, None], jnp.uint32(0), ports_curr)
    ports_arr = state.ports.at[ids_safe].set(ports_update)
    return state._replace(node_type=node_type_arr, ports=ports_arr)


@partial(jax.jit, static_argnames=("set_oom_on_fail",))
def ic_alloc_jax(
    state: ICState,
    count: jnp.ndarray,
    node_type: jnp.uint8,
    set_oom_on_fail: bool = False,
) -> Tuple[ICState, jnp.ndarray, jnp.ndarray]:
    """Device-only allocator for construction (no host sync)."""
    c = jnp.asarray(count, dtype=jnp.int32)
    idx = jnp.where(
        c == 0,
        0,
        jnp.where(
            c == 1, 1, jnp.where(c == 2, 2, jnp.where(c == 4, 3, 4))
        ),
    ).astype(jnp.int32)

    def _case0(s):
        return _alloc_pad(s, 0)

    def _case1(s):
        return _alloc_pad(s, 1)

    def _case2(s):
        return _alloc_pad(s, 2)

    def _case4(s):
        return _alloc_pad(s, 4)

    def _bad(s):
        return s, jnp.zeros((4,), dtype=jnp.uint32), jnp.bool_(False)

    state2, ids4, ok = jax.lax.switch(
        idx, (_case0, _case1, _case2, _case4, _bad), state
    )
    if set_oom_on_fail:
        state2 = state2._replace(oom=state2.oom | (~ok))

    def _do_init(s):
        return _init_nodes_jax(s, ids4, c, node_type)

    do_init = ok & (c > 0)
    state2 = jax.lax.cond(do_init, _do_init, lambda s: s, state2)
    return state2, ids4, ok

def encode_port(node_idx: jnp.ndarray, port_idx: jnp.ndarray) -> jnp.ndarray:
    return (node_idx.astype(jnp.uint32) << jnp.uint32(2)) | port_idx.astype(
        jnp.uint32
    )


def decode_port(ptr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    node = ptr >> jnp.uint32(2)
    port = ptr & jnp.uint32(0x3)
    return node.astype(jnp.uint32), port.astype(jnp.uint32)


def ic_init(capacity: int) -> ICState:
    node_type = jnp.full((capacity,), TYPE_FREE, dtype=jnp.uint8)
    ports = jnp.zeros((capacity, 3), dtype=jnp.uint32)
    free_stack = jnp.arange(capacity - 1, -1, -1, dtype=jnp.uint32)
    # Node 0 is reserved (encode_port(0, PORT_PRINCIPAL) == 0 sentinel).
    free_top = jnp.array(max(capacity - 1, 0), dtype=jnp.uint32)
    oom = jnp.array(False, dtype=jnp.bool_)
    return ICState(
        node_type=node_type,
        ports=ports,
        free_stack=free_stack,
        free_top=free_top,
        oom=oom,
    )


def ic_wire(
    state: ICState,
    node_a: int,
    port_a: int,
    node_b: int,
    port_b: int,
) -> ICState:
    ptr_a = encode_port(jnp.asarray(node_a), jnp.asarray(port_a))
    ptr_b = encode_port(jnp.asarray(node_b), jnp.asarray(port_b))
    ports = state.ports
    ports = ports.at[node_a, port_a].set(ptr_b)
    ports = ports.at[node_b, port_b].set(ptr_a)
    return state._replace(ports=ports)


@jax.jit
def ic_wire_jax(
    state: ICState,
    node_a: jnp.ndarray,
    port_a: jnp.ndarray,
    node_b: jnp.ndarray,
    port_b: jnp.ndarray,
) -> ICState:
    """Device-only wire: connect (node, port) <-> (node, port)."""
    node_a = jnp.asarray(node_a, dtype=jnp.uint32)
    port_a = jnp.asarray(port_a, dtype=jnp.uint32)
    node_b = jnp.asarray(node_b, dtype=jnp.uint32)
    port_b = jnp.asarray(port_b, dtype=jnp.uint32)
    ptr_a = encode_port(node_a, port_a)
    ptr_b = encode_port(node_b, port_b)
    ports = state.ports
    ports = ports.at[node_a, port_a].set(ptr_b)
    ports = ports.at[node_b, port_b].set(ptr_a)
    return state._replace(ports=ports)


@jax.jit
def ic_wire_ptrs_jax(
    state: ICState, ptr_a: jnp.ndarray, ptr_b: jnp.ndarray
) -> ICState:
    """Device-only wire given two encoded pointers (NULL-safe)."""
    ports = _connect_ptrs(state.ports, ptr_a, ptr_b)
    return state._replace(ports=ports)


@jax.jit
def ic_wire_jax_safe(
    state: ICState,
    node_a: jnp.ndarray,
    port_a: jnp.ndarray,
    node_b: jnp.ndarray,
    port_b: jnp.ndarray,
) -> ICState:
    """Device-only wire that no-ops on NULL endpoints."""
    ptr_a = encode_port(jnp.asarray(node_a, jnp.uint32), jnp.asarray(port_a, jnp.uint32))
    ptr_b = encode_port(jnp.asarray(node_b, jnp.uint32), jnp.asarray(port_b, jnp.uint32))
    return ic_wire_ptrs_jax(state, ptr_a, ptr_b)


@jax.jit
def ic_wire_pairs_jax(
    state: ICState,
    node_a: jnp.ndarray,
    port_a: jnp.ndarray,
    node_b: jnp.ndarray,
    port_b: jnp.ndarray,
) -> ICState:
    """Batch wire: connect (node_a[i], port_a[i]) <-> (node_b[i], port_b[i])."""
    node_a = jnp.asarray(node_a, dtype=jnp.uint32)
    port_a = jnp.asarray(port_a, dtype=jnp.uint32)
    node_b = jnp.asarray(node_b, dtype=jnp.uint32)
    port_b = jnp.asarray(port_b, dtype=jnp.uint32)

    n_nodes = state.ports.shape[0]
    n_nodes_u = jnp.uint32(n_nodes)
    na = jnp.minimum(node_a, n_nodes_u - jnp.uint32(1))
    nb = jnp.minimum(node_b, n_nodes_u - jnp.uint32(1))
    pa = port_a & jnp.uint32(0x3)
    pb = port_b & jnp.uint32(0x3)

    ptr_a = encode_port(na, pa)
    ptr_b = encode_port(nb, pb)
    do = (ptr_a != jnp.uint32(0)) & (ptr_b != jnp.uint32(0))

    safe_node = jnp.uint32(0)
    safe_port = jnp.uint32(0)
    na_s = jnp.where(do, na, safe_node)
    pa_s = jnp.where(do, pa, safe_port)
    nb_s = jnp.where(do, nb, safe_node)
    pb_s = jnp.where(do, pb, safe_port)

    ports = state.ports
    val_a = jnp.where(do, ptr_b, ports[safe_node, safe_port])
    val_b = jnp.where(do, ptr_a, ports[safe_node, safe_port])
    ports = ports.at[na_s, pa_s].set(val_a, mode="drop")
    ports = ports.at[nb_s, pb_s].set(val_b, mode="drop")
    return state._replace(ports=ports)


@jax.jit
def ic_wire_ptr_pairs_jax(
    state: ICState, ptr_a: jnp.ndarray, ptr_b: jnp.ndarray
) -> ICState:
    """Batch wire given encoded pointers (NULL-safe)."""
    ptr_a = jnp.asarray(ptr_a, dtype=jnp.uint32)
    ptr_b = jnp.asarray(ptr_b, dtype=jnp.uint32)
    do = (ptr_a != jnp.uint32(0)) & (ptr_b != jnp.uint32(0))

    na, pa = decode_port(ptr_a)
    nb, pb = decode_port(ptr_b)

    n_nodes = state.ports.shape[0]
    n_nodes_u = jnp.uint32(n_nodes)
    na = jnp.minimum(na, n_nodes_u - jnp.uint32(1))
    nb = jnp.minimum(nb, n_nodes_u - jnp.uint32(1))
    pa = pa & jnp.uint32(0x3)
    pb = pb & jnp.uint32(0x3)

    safe_node = jnp.uint32(0)
    safe_port = jnp.uint32(0)
    na_s = jnp.where(do, na, safe_node)
    pa_s = jnp.where(do, pa, safe_port)
    nb_s = jnp.where(do, nb, safe_node)
    pb_s = jnp.where(do, pb, safe_port)

    ports = state.ports
    val_a = jnp.where(do, ptr_b, ports[safe_node, safe_port])
    val_b = jnp.where(do, ptr_a, ports[safe_node, safe_port])
    ports = ports.at[na_s, pa_s].set(val_a, mode="drop")
    ports = ports.at[nb_s, pb_s].set(val_b, mode="drop")
    return state._replace(ports=ports)


@jax.jit
def ic_wire_star_jax(
    state: ICState,
    center_node: jnp.ndarray,
    center_port: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    leaf_ports: jnp.ndarray,
) -> ICState:
    """Wire a single center endpoint to many leaf endpoints (device-only)."""
    center_node = jnp.asarray(center_node, dtype=jnp.uint32)
    center_port = jnp.asarray(center_port, dtype=jnp.uint32)
    leaf_nodes = jnp.asarray(leaf_nodes, dtype=jnp.uint32)
    leaf_ports = jnp.asarray(leaf_ports, dtype=jnp.uint32)
    k = leaf_nodes.shape[0]
    node_a = jnp.full((k,), center_node, dtype=jnp.uint32)
    port_a = jnp.full((k,), center_port, dtype=jnp.uint32)
    return ic_wire_pairs_jax(state, node_a, port_a, leaf_nodes, leaf_ports)
@jax.jit
def ic_find_active_pairs(state: ICState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return indices of nodes in active principal-principal pairs."""
    ports = state.ports
    n = ports.shape[0]
    idx = jnp.arange(n, dtype=jnp.uint32)
    ptr = ports[:, 0]
    is_connected = ptr != jnp.uint32(0)
    tgt_node, tgt_port = decode_port(ptr)
    is_principal = tgt_port == PORT_PRINCIPAL
    in_bounds = tgt_node < jnp.uint32(n)
    safe_tgt = jnp.where(
        is_connected & is_principal & in_bounds, tgt_node, jnp.uint32(0)
    )
    back = ports[safe_tgt, 0]
    back_node, back_port = decode_port(back)
    mutual = (back_node == idx) & (back_port == PORT_PRINCIPAL)
    active = is_connected & is_principal & in_bounds & mutual & (idx < tgt_node)
    pairs = jnp.nonzero(active, size=n, fill_value=0)[0]
    return pairs.astype(jnp.uint32), active


def _compact_mask(mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    size = mask.shape[0]
    count = jnp.sum(mask).astype(jnp.uint32)
    idx = jnp.nonzero(mask, size=size, fill_value=0)[0].astype(jnp.uint32)
    valid = jnp.arange(size, dtype=jnp.uint32) < count
    return idx, valid, count


@jax.jit
def ic_compact_active_pairs(
    state: ICState,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return compacted active pair indices and a count."""
    _, active = ic_find_active_pairs(state)
    idx, valid, count = _compact_mask(active)
    compacted = jnp.where(valid, idx, jnp.uint32(0))
    return compacted, count, active


@jax.jit
def ic_rule_for_types(type_a: jnp.ndarray, type_b: jnp.ndarray) -> jnp.ndarray:
    """Lookup rule vector [alloc_count, template_id] for a type pair."""
    a = type_a.astype(jnp.uint32)
    b = type_b.astype(jnp.uint32)
    return RULE_TABLE[a, b]


def ic_select_template(state: ICState, node_a: int, node_b: int) -> jnp.ndarray:
    type_a = state.node_type[node_a]
    type_b = state.node_type[node_b]
    return ic_rule_for_types(type_a, type_b)[1]


def ic_apply_annihilate(state: ICState, node_a: int, node_b: int) -> ICState:
    ports = state.ports
    a_left = ports[node_a, 1]
    a_right = ports[node_a, 2]
    b_left = ports[node_b, 1]
    b_right = ports[node_b, 2]
    ports = _connect_ptrs(ports, a_left, b_left)
    ports = _connect_ptrs(ports, a_right, b_right)
    ports = ports.at[node_a].set(jnp.uint32(0))
    ports = ports.at[node_b].set(jnp.uint32(0))
    node_type = state.node_type
    node_type = node_type.at[node_a].set(TYPE_FREE)
    node_type = node_type.at[node_b].set(TYPE_FREE)
    return state._replace(ports=ports, node_type=node_type)


def _init_nodes(state: ICState, nodes: jnp.ndarray, node_type: jnp.uint8) -> ICState:
    node_type_arr = state.node_type.at[nodes].set(node_type)
    ports = state.ports.at[nodes].set(jnp.uint32(0))
    return state._replace(node_type=node_type_arr, ports=ports)


def _alloc2(state: ICState) -> Tuple[ICState, jnp.ndarray]:
    top = state.free_top.astype(jnp.int32)
    ok = (top >= 2) & (~state.oom)

    def _do(s):
        start = top - 2
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (2,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32)

    def _fail(s):
        return s._replace(oom=jnp.bool_(True)), jnp.zeros((2,), dtype=jnp.uint32)

    return jax.lax.cond(ok, _do, _fail, state)


def _alloc4(state: ICState) -> Tuple[ICState, jnp.ndarray]:
    top = state.free_top.astype(jnp.int32)
    ok = (top >= 4) & (~state.oom)

    def _do(s):
        start = top - 4
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (4,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32)

    def _fail(s):
        return s._replace(oom=jnp.bool_(True)), jnp.zeros((4,), dtype=jnp.uint32)

    return jax.lax.cond(ok, _do, _fail, state)


def _free2(state: ICState, nodes: jnp.ndarray) -> ICState:
    top = state.free_top.astype(jnp.int32)
    cap = state.free_stack.shape[0]
    ok = (top + 2) <= cap

    def _do(s):
        fs = s.free_stack
        fs = fs.at[top + 0].set(nodes[0])
        fs = fs.at[top + 1].set(nodes[1])
        return s._replace(free_stack=fs, free_top=jnp.uint32(top + 2))

    def _fail(s):
        return s._replace(oom=jnp.bool_(True))

    return jax.lax.cond(ok & (~state.oom), _do, _fail, state)


def ic_apply_erase(state: ICState, node_a: jnp.ndarray, node_b: jnp.ndarray) -> ICState:
    type_a = state.node_type[node_a]
    is_era_a = type_a == TYPE_ERA
    era = jnp.where(is_era_a, node_a, node_b).astype(jnp.uint32)
    target = jnp.where(is_era_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    aux_left = ports[target, 1]
    aux_right = ports[target, 2]
    state2, eras = _alloc2(state)

    def _do(s):
        s = _init_nodes(s, eras, TYPE_ERA)
        ports = s.ports
        ports = _connect_ptrs(
            ports, encode_port(eras[0], PORT_PRINCIPAL), aux_left
        )
        ports = _connect_ptrs(
            ports, encode_port(eras[1], PORT_PRINCIPAL), aux_right
        )
        ports = ports.at[era].set(jnp.uint32(0))
        ports = ports.at[target].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[era].set(TYPE_FREE)
        node_type = node_type.at[target].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return _free2(s, jnp.stack([era, target]).astype(jnp.uint32))

    return jax.lax.cond(state2.oom, lambda s: s, _do, state2)


def _ic_apply_erase_with_ids(
    state: ICState, node_a: jnp.ndarray, node_b: jnp.ndarray, eras: jnp.ndarray
) -> ICState:
    type_a = state.node_type[node_a]
    is_era_a = type_a == TYPE_ERA
    era = jnp.where(is_era_a, node_a, node_b).astype(jnp.uint32)
    target = jnp.where(is_era_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    aux_left = ports[target, 1]
    aux_right = ports[target, 2]

    def _do(s):
        s = _init_nodes(s, eras, TYPE_ERA)
        ports = s.ports
        ports = _connect_ptrs(
            ports, encode_port(eras[0], PORT_PRINCIPAL), aux_left
        )
        ports = _connect_ptrs(
            ports, encode_port(eras[1], PORT_PRINCIPAL), aux_right
        )
        ports = ports.at[era].set(jnp.uint32(0))
        ports = ports.at[target].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[era].set(TYPE_FREE)
        node_type = node_type.at[target].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return _free2(s, jnp.stack([era, target]).astype(jnp.uint32))

    return jax.lax.cond(state.oom, lambda s: s, _do, state)


def ic_apply_commute(state: ICState, node_a: jnp.ndarray, node_b: jnp.ndarray) -> ICState:
    type_a = state.node_type[node_a]
    is_con_a = type_a == TYPE_CON
    con = jnp.where(is_con_a, node_a, node_b).astype(jnp.uint32)
    dup = jnp.where(is_con_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    con_left = ports[con, 1]
    con_right = ports[con, 2]
    dup_left = ports[dup, 1]
    dup_right = ports[dup, 2]
    state2, ids4 = _alloc4(state)

    def _do(s):
        dup_nodes = ids4[:2]
        con_nodes = ids4[2:]
        s = _init_nodes(s, con_nodes, TYPE_CON)
        s = _init_nodes(s, dup_nodes, TYPE_DUP)
        c0, c1 = con_nodes
        d0, d1 = dup_nodes
        ports = s.ports
        ports = _connect_ptrs(
            ports, encode_port(c0, PORT_PRINCIPAL), encode_port(d0, PORT_PRINCIPAL)
        )
        ports = _connect_ptrs(
            ports, encode_port(c1, PORT_PRINCIPAL), encode_port(d1, PORT_PRINCIPAL)
        )
        ports = _connect_ptrs(ports, encode_port(c0, PORT_AUX_LEFT), dup_left)
        ports = _connect_ptrs(ports, encode_port(c1, PORT_AUX_LEFT), dup_right)
        ports = _connect_ptrs(ports, encode_port(d0, PORT_AUX_LEFT), con_left)
        ports = _connect_ptrs(ports, encode_port(d1, PORT_AUX_LEFT), con_right)
        ports = _connect_ptrs(
            ports, encode_port(c0, PORT_AUX_RIGHT), encode_port(d0, PORT_AUX_RIGHT)
        )
        ports = _connect_ptrs(
            ports, encode_port(c1, PORT_AUX_RIGHT), encode_port(d1, PORT_AUX_RIGHT)
        )
        ports = ports.at[con].set(jnp.uint32(0))
        ports = ports.at[dup].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[con].set(TYPE_FREE)
        node_type = node_type.at[dup].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return _free2(s, jnp.stack([con, dup]).astype(jnp.uint32))

    return jax.lax.cond(state2.oom, lambda s: s, _do, state2)


def _ic_apply_commute_with_ids(
    state: ICState, node_a: jnp.ndarray, node_b: jnp.ndarray, ids4: jnp.ndarray
) -> ICState:
    type_a = state.node_type[node_a]
    is_con_a = type_a == TYPE_CON
    con = jnp.where(is_con_a, node_a, node_b).astype(jnp.uint32)
    dup = jnp.where(is_con_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    con_left = ports[con, 1]
    con_right = ports[con, 2]
    dup_left = ports[dup, 1]
    dup_right = ports[dup, 2]

    def _do(s):
        dup_nodes = ids4[:2]
        con_nodes = ids4[2:]
        s = _init_nodes(s, con_nodes, TYPE_CON)
        s = _init_nodes(s, dup_nodes, TYPE_DUP)
        c0, c1 = con_nodes
        d0, d1 = dup_nodes
        ports = s.ports
        ports = _connect_ptrs(
            ports, encode_port(c0, PORT_PRINCIPAL), encode_port(d0, PORT_PRINCIPAL)
        )
        ports = _connect_ptrs(
            ports, encode_port(c1, PORT_PRINCIPAL), encode_port(d1, PORT_PRINCIPAL)
        )
        ports = _connect_ptrs(ports, encode_port(c0, PORT_AUX_LEFT), dup_left)
        ports = _connect_ptrs(ports, encode_port(c1, PORT_AUX_LEFT), dup_right)
        ports = _connect_ptrs(ports, encode_port(d0, PORT_AUX_LEFT), con_left)
        ports = _connect_ptrs(ports, encode_port(d1, PORT_AUX_LEFT), con_right)
        ports = _connect_ptrs(
            ports, encode_port(c0, PORT_AUX_RIGHT), encode_port(d0, PORT_AUX_RIGHT)
        )
        ports = _connect_ptrs(
            ports, encode_port(c1, PORT_AUX_RIGHT), encode_port(d1, PORT_AUX_RIGHT)
        )
        ports = ports.at[con].set(jnp.uint32(0))
        ports = ports.at[dup].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[con].set(TYPE_FREE)
        node_type = node_type.at[dup].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return _free2(s, jnp.stack([con, dup]).astype(jnp.uint32))

    return jax.lax.cond(state.oom, lambda s: s, _do, state)


def ic_apply_template(
    state: ICState, node_a: jnp.ndarray, node_b: jnp.ndarray, template_id: jnp.ndarray
) -> ICState:
    template_id = template_id.astype(jnp.int32)

    def _noop(s):
        return s

    def _ann(s):
        return ic_apply_annihilate(s, node_a, node_b)

    def _erase(s):
        return ic_apply_erase(s, node_a, node_b)

    def _comm(s):
        return ic_apply_commute(s, node_a, node_b)

    def _apply(s):
        return jax.lax.switch(
            template_id, (_noop, _ann, _erase, _comm), s
        )

    return jax.lax.cond(state.oom, _noop, _apply, state)


def _alloc_plan(
    state: ICState, pairs: jnp.ndarray, count: jnp.ndarray
) -> Tuple[ICState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = pairs.shape[0]
    idx = jnp.arange(n, dtype=jnp.uint32)
    valid = idx < count
    node_a = jnp.where(valid, pairs, jnp.uint32(0))
    node_b = decode_port(state.ports[node_a, PORT_PRINCIPAL])[0]
    rule = RULE_TABLE[state.node_type[node_a].astype(jnp.uint32),
                      state.node_type[node_b].astype(jnp.uint32)]
    alloc_counts = jnp.where(valid, rule[:, 0], jnp.uint32(0))
    template_ids = jnp.where(valid, rule[:, 1], TEMPLATE_NONE)
    offsets = jnp.cumsum(alloc_counts) - alloc_counts
    total_alloc = jnp.sum(alloc_counts).astype(jnp.int32)
    free_top = state.free_top.astype(jnp.int32)
    ok = (free_top >= total_alloc) & (~state.oom)
    base = free_top - total_alloc
    base_safe = jnp.where(ok, base, jnp.int32(0))
    free_top_new = jnp.where(ok, _safe_uint32(base), state.free_top)
    oom_new = jnp.where(ok, state.oom, jnp.bool_(True))
    state2 = state._replace(free_top=free_top_new, oom=oom_new)

    zeros4 = jnp.zeros((4,), dtype=jnp.uint32)
    alloc_ids = jnp.zeros((n, 4), dtype=jnp.uint32)

    def build(i, buf):
        count_i = alloc_counts[i]
        offset = offsets[i].astype(jnp.int32)
        start = base_safe + offset

        def take2(_):
            ids2 = jax.lax.dynamic_slice(state.free_stack, (start,), (2,))
            return jnp.concatenate([ids2, zeros4[:2]], axis=0)

        def take4(_):
            return jax.lax.dynamic_slice(state.free_stack, (start,), (4,))

        ids = jax.lax.cond(
            count_i == jnp.uint32(2),
            take2,
            lambda _: jax.lax.cond(
                count_i == jnp.uint32(4),
                take4,
                lambda __: zeros4,
                operand=None,
            ),
            operand=None,
        )
        return buf.at[i].set(ids)

    alloc_ids = jax.lax.cond(
        ok, lambda _: jax.lax.fori_loop(0, n, build, alloc_ids), lambda _: alloc_ids, operand=None
    )
    return state2, template_ids, alloc_counts, alloc_ids, valid


def _apply_template_planned(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    template_id: jnp.ndarray,
    alloc_ids: jnp.ndarray,
) -> ICState:
    template_id = template_id.astype(jnp.int32)

    def _noop(s):
        return s

    def _ann(s):
        return ic_apply_annihilate(s, node_a, node_b)

    def _erase(s):
        return _ic_apply_erase_with_ids(s, node_a, node_b, alloc_ids[:2])

    def _comm(s):
        return _ic_apply_commute_with_ids(s, node_a, node_b, alloc_ids)

    def _apply(s):
        return jax.lax.switch(template_id, (_noop, _ann, _erase, _comm), s)

    return jax.lax.cond(state.oom, _noop, _apply, state)


@jax.jit
def ic_apply_active_pairs(state: ICState) -> Tuple[ICState, ICRewriteStats]:
    pairs, count, _ = ic_compact_active_pairs(state)
    count_i = count.astype(jnp.int32)
    zero_stats = ICRewriteStats(
        active_pairs=jnp.uint32(0),
        alloc_nodes=jnp.uint32(0),
        freed_nodes=jnp.uint32(0),
        template_counts=jnp.zeros((4,), dtype=jnp.uint32),
    )
    def body(i, carry):
        s, alloc, freed, tmpl_counts, tmpl_ids, alloc_counts, alloc_ids = carry
        node_a = pairs[i]
        node_b = decode_port(s.ports[node_a, PORT_PRINCIPAL])[0]
        tmpl = tmpl_ids[i]
        s2 = _apply_template_planned(s, node_a, node_b, tmpl, alloc_ids[i])
        ok = (~s.oom) & (~s2.oom)
        tmpl_i = tmpl.astype(jnp.int32)
        tmpl_counts = tmpl_counts.at[tmpl_i].add(ok.astype(jnp.uint32))
        alloc_delta = jnp.where(ok, alloc_counts[i], jnp.uint32(0))
        freed_delta = jnp.where(
            (tmpl != TEMPLATE_NONE) & ok, jnp.uint32(2), jnp.uint32(0)
        )
        return s2, alloc + alloc_delta, freed + freed_delta, tmpl_counts, tmpl_ids, alloc_counts, alloc_ids

    def _apply(s):
        s2, tmpl_ids, alloc_counts, alloc_ids, valid = _alloc_plan(s, pairs, count)
        def _run(s_in):
            init = (
                s_in,
                jnp.uint32(0),
                jnp.uint32(0),
                zero_stats.template_counts,
                tmpl_ids,
                alloc_counts,
                alloc_ids,
            )
            s_out, alloc, freed, tmpl_counts, _, _, _ = jax.lax.fori_loop(
                0, count_i, body, init
            )
            stats = ICRewriteStats(
                active_pairs=count,
                alloc_nodes=alloc,
                freed_nodes=freed,
                template_counts=tmpl_counts,
            )
            return s_out, stats

        return jax.lax.cond(s2.oom, lambda s_in: (s_in, zero_stats), _run, s2)

    return jax.lax.cond(count_i == 0, lambda s: (s, zero_stats), _apply, state)


@jax.jit
def ic_reduce(
    state: ICState, max_steps: int
) -> Tuple[ICState, ICRewriteStats, jnp.ndarray]:
    max_steps_i = jnp.int32(max_steps)
    zero_stats = ICRewriteStats(
        active_pairs=jnp.uint32(0),
        alloc_nodes=jnp.uint32(0),
        freed_nodes=jnp.uint32(0),
        template_counts=jnp.zeros((4,), dtype=jnp.uint32),
    )

    def cond(carry):
        s, stats, steps, last_active = carry
        return (steps < max_steps_i) & (last_active > 0) & (~s.oom)

    def body(carry):
        s, stats, steps, _ = carry
        s2, batch = ic_apply_active_pairs(s)
        stats = ICRewriteStats(
            active_pairs=stats.active_pairs + batch.active_pairs,
            alloc_nodes=stats.alloc_nodes + batch.alloc_nodes,
            freed_nodes=stats.freed_nodes + batch.freed_nodes,
            template_counts=stats.template_counts + batch.template_counts,
        )
        return s2, stats, steps + 1, batch.active_pairs

    init = (state, zero_stats, jnp.int32(0), jnp.int32(1))
    s_out, stats_out, steps_out, last_active = jax.lax.while_loop(
        cond, body, init
    )
    return s_out, stats_out, steps_out


def _alloc_nodes(state: ICState, count: jnp.ndarray) -> Tuple[ICState, jnp.ndarray]:
    n = int(count)
    if n == 0:
        return state, jnp.zeros((0,), dtype=jnp.uint32)
    free_top = int(state.free_top)
    if free_top < n:
        raise ValueError("free_stack underflow")
    idx = state.free_stack[free_top - n:free_top]
    free_top = free_top - n
    return state._replace(free_top=jnp.uint32(free_top)), idx


def ic_alloc(
    state: ICState, count: int, node_type: jnp.uint8
) -> Tuple[ICState, jnp.ndarray]:
    state, nodes = _alloc_nodes(state, jnp.asarray(count, dtype=jnp.uint32))
    if nodes.size == 0:
        return state, nodes
    node_type_arr = state.node_type.at[nodes].set(node_type)
    ports = state.ports.at[nodes].set(jnp.uint32(0))
    return state._replace(node_type=node_type_arr, ports=ports), nodes


def _free_nodes(state: ICState, nodes: jnp.ndarray) -> ICState:
    if nodes.size == 0:
        return state
    count = int(nodes.shape[0])
    free_top = int(state.free_top)
    cap = int(state.free_stack.shape[0])
    if free_top + count > cap:
        raise ValueError(
            "free_stack overflow (double-free or freeing unallocated nodes)"
        )
    free_stack = state.free_stack
    free_stack = free_stack.at[free_top:free_top + count].set(nodes)
    return state._replace(free_stack=free_stack, free_top=jnp.uint32(free_top + count))
