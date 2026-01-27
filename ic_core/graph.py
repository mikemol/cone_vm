import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple

# Interaction-combinator (IC) graph + safety helpers.

TYPE_FREE = jnp.uint8(0)
TYPE_ERA = jnp.uint8(1)
TYPE_CON = jnp.uint8(2)
TYPE_DUP = jnp.uint8(3)

PORT_PRINCIPAL = jnp.uint32(0)
PORT_AUX_LEFT = jnp.uint32(1)
PORT_AUX_RIGHT = jnp.uint32(2)


class ICState(NamedTuple):
    node_type: jnp.ndarray
    ports: jnp.ndarray
    free_stack: jnp.ndarray
    free_top: jnp.ndarray
    oom: jnp.ndarray
    corrupt: jnp.ndarray


def _halted(state: ICState) -> jnp.ndarray:
    return state.oom | state.corrupt


def _scan_corrupt_ports(state: ICState) -> ICState:
    ptrs = state.ports.reshape(-1)
    bad = (ptrs != jnp.uint32(0)) & ((ptrs & jnp.uint32(0x3)) == jnp.uint32(3))
    return state._replace(corrupt=state.corrupt | jnp.any(bad))


def _safe_uint32(value) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.uint32)


def _alloc_raw(state: ICState, count: int) -> Tuple[ICState, jnp.ndarray, jnp.ndarray]:
    if count > state.free_stack.shape[0]:
        return state, jnp.zeros((count,), dtype=jnp.uint32), jnp.bool_(False)
    top = state.free_top.astype(jnp.int32)
    ok = (top >= count) & (~state.oom) & (~state.corrupt)

    def _do(s):
        start = top - count
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (count,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32), jnp.bool_(True)

    def _fail(s):
        return s, jnp.zeros((count,), dtype=jnp.uint32), jnp.bool_(False)

    return jax.lax.cond(ok, _do, _fail, state)


def _alloc_pad(state: ICState, count: int) -> Tuple[ICState, jnp.ndarray, jnp.ndarray]:
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
    node_type_curr = state.node_type[ids4]
    node_type_update = jnp.where(mask, node_type, node_type_curr)
    node_type_arr = state.node_type.at[ids4].set(node_type_update)
    ports_curr = state.ports[ids4]
    ports_update = jnp.where(mask[:, None], jnp.uint32(0), ports_curr)
    ports_arr = state.ports.at[ids4].set(ports_update)
    return state._replace(node_type=node_type_arr, ports=ports_arr)


def _init_nodes(state: ICState, nodes: jnp.ndarray, node_type: jnp.uint8) -> ICState:
    node_type_arr = state.node_type.at[nodes].set(node_type)
    ports = state.ports.at[nodes].set(jnp.uint32(0))
    return state._replace(node_type=node_type_arr, ports=ports)


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
        jnp.where(c == 1, 1, jnp.where(c == 2, 2, jnp.where(c == 4, 3, 4))),
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

    def _run(s):
        return jax.lax.switch(idx, (_case0, _case1, _case2, _case4, _bad), s)

    def _halt(s):
        return s, jnp.zeros((4,), dtype=jnp.uint32), jnp.bool_(False)

    state2, ids4, ok = jax.lax.cond(state.corrupt, _halt, _run, state)
    if set_oom_on_fail:
        state2 = state2._replace(oom=state2.oom | ((~ok) & (~state.corrupt)))

    def _do_init(s):
        return _init_nodes_jax(s, ids4, c, node_type)

    do_init = ok & (c > 0)
    state2 = jax.lax.cond(do_init, _do_init, lambda s: s, state2)
    return state2, ids4, ok


def encode_port(node_idx: jnp.ndarray, port_idx: jnp.ndarray) -> jnp.ndarray:
    return (node_idx.astype(jnp.uint32) << jnp.uint32(2)) | port_idx.astype(jnp.uint32)


def decode_port(ptr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    node = ptr >> jnp.uint32(2)
    port = ptr & jnp.uint32(0x3)
    return node.astype(jnp.uint32), port.astype(jnp.uint32)


def ic_init(capacity: int) -> ICState:
    node_type = jnp.full((capacity,), TYPE_FREE, dtype=jnp.uint8)
    ports = jnp.zeros((capacity, 3), dtype=jnp.uint32)
    free_stack = jnp.arange(capacity - 1, -1, -1, dtype=jnp.uint32)
    # Node 0 is reserved (encode_port(0, PORT_PRINCIPAL) == 0 sentinel).
    free_top = jnp.array(
        capacity if capacity < 3 else max(capacity - 1, 0),
        dtype=jnp.uint32,
    )
    oom = jnp.array(False, dtype=jnp.bool_)
    corrupt = jnp.array(False, dtype=jnp.bool_)
    return ICState(
        node_type=node_type,
        ports=ports,
        free_stack=free_stack,
        free_top=free_top,
        oom=oom,
        corrupt=corrupt,
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


def _connect_ptrs(ports: jnp.ndarray, ptr_a: jnp.ndarray, ptr_b: jnp.ndarray) -> jnp.ndarray:
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


@jax.jit
def ic_wire_jax(
    state: ICState,
    node_a: jnp.ndarray,
    port_a: jnp.ndarray,
    node_b: jnp.ndarray,
    port_b: jnp.ndarray,
) -> ICState:
    """Device-only wire: connect (node, port) <-> (node, port)."""

    def _do(s):
        node_a_u = jnp.asarray(node_a, dtype=jnp.uint32)
        port_a_u = jnp.asarray(port_a, dtype=jnp.uint32)
        node_b_u = jnp.asarray(node_b, dtype=jnp.uint32)
        port_b_u = jnp.asarray(port_b, dtype=jnp.uint32)
        ptr_a = encode_port(node_a_u, port_a_u)
        ptr_b = encode_port(node_b_u, port_b_u)
        ports = s.ports
        ports = ports.at[node_a_u, port_a_u].set(ptr_b)
        ports = ports.at[node_b_u, port_b_u].set(ptr_a)
        return s._replace(ports=ports)

    return jax.lax.cond(_halted(state), lambda s: s, _do, state)


@jax.jit
def ic_wire_ptrs_jax(state: ICState, ptr_a: jnp.ndarray, ptr_b: jnp.ndarray) -> ICState:
    """Device-only wire given two encoded pointers (NULL-safe)."""

    def _do(s):
        ports = _connect_ptrs(s.ports, ptr_a, ptr_b)
        return s._replace(ports=ports)

    return jax.lax.cond(_halted(state), lambda s: s, _do, state)


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

    def _do(s):
        node_a_u = jnp.asarray(node_a, dtype=jnp.uint32)
        port_a_u = jnp.asarray(port_a, dtype=jnp.uint32)
        node_b_u = jnp.asarray(node_b, dtype=jnp.uint32)
        port_b_u = jnp.asarray(port_b, dtype=jnp.uint32)

        n_nodes = s.ports.shape[0]
        n_nodes_u = jnp.uint32(n_nodes)
        na = jnp.minimum(node_a_u, n_nodes_u - jnp.uint32(1))
        nb = jnp.minimum(node_b_u, n_nodes_u - jnp.uint32(1))
        pa = port_a_u & jnp.uint32(0x3)
        pb = port_b_u & jnp.uint32(0x3)

        ptr_a = encode_port(na, pa)
        ptr_b = encode_port(nb, pb)
        do = (ptr_a != jnp.uint32(0)) & (ptr_b != jnp.uint32(0))

        safe_node = jnp.uint32(0)
        safe_port = jnp.uint32(0)
        na_s = jnp.where(do, na, safe_node)
        pa_s = jnp.where(do, pa, safe_port)
        nb_s = jnp.where(do, nb, safe_node)
        pb_s = jnp.where(do, pb, safe_port)

        ports = s.ports
        val_a = jnp.where(do, ptr_b, ports[safe_node, safe_port])
        val_b = jnp.where(do, ptr_a, ports[safe_node, safe_port])
        ports = ports.at[na_s, pa_s].set(val_a, mode="drop")
        ports = ports.at[nb_s, pb_s].set(val_b, mode="drop")
        return s._replace(ports=ports)

    return jax.lax.cond(_halted(state), lambda s: s, _do, state)


@jax.jit
def ic_wire_ptr_pairs_jax(state: ICState, ptr_a: jnp.ndarray, ptr_b: jnp.ndarray) -> ICState:
    """Batch wire given encoded pointers (NULL-safe)."""

    def _do(s):
        ptr_a_u = jnp.asarray(ptr_a, dtype=jnp.uint32)
        ptr_b_u = jnp.asarray(ptr_b, dtype=jnp.uint32)
        do = (ptr_a_u != jnp.uint32(0)) & (ptr_b_u != jnp.uint32(0))

        na, pa = decode_port(ptr_a_u)
        nb, pb = decode_port(ptr_b_u)

        n_nodes = s.ports.shape[0]
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

        ports = s.ports
        val_a = jnp.where(do, ptr_b_u, ports[safe_node, safe_port])
        val_b = jnp.where(do, ptr_a_u, ports[safe_node, safe_port])
        ports = ports.at[na_s, pa_s].set(val_a, mode="drop")
        ports = ports.at[nb_s, pb_s].set(val_b, mode="drop")
        return s._replace(ports=ports)

    return jax.lax.cond(_halted(state), lambda s: s, _do, state)


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
    safe_tgt = jnp.where(is_connected & is_principal & in_bounds, tgt_node, jnp.uint32(0))
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


def _alloc2(state: ICState) -> Tuple[ICState, jnp.ndarray]:
    top = state.free_top.astype(jnp.int32)
    ok = (top >= 2) & (~state.oom) & (~state.corrupt)

    def _do(s):
        start = top - 2
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (2,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32)

    def _fail(s):
        def _keep(s_in):
            return s_in

        def _oom(s_in):
            return s_in._replace(oom=jnp.bool_(True))

        s2 = jax.lax.cond(s.corrupt, _keep, _oom, s)
        return s2, jnp.zeros((2,), dtype=jnp.uint32)

    return jax.lax.cond(ok, _do, _fail, state)


def _alloc4(state: ICState) -> Tuple[ICState, jnp.ndarray]:
    top = state.free_top.astype(jnp.int32)
    ok = (top >= 4) & (~state.oom) & (~state.corrupt)

    def _do(s):
        start = top - 4
        ids = jax.lax.dynamic_slice(s.free_stack, (start,), (4,))
        return s._replace(free_top=jnp.uint32(start)), ids.astype(jnp.uint32)

    def _fail(s):
        def _keep(s_in):
            return s_in

        def _oom(s_in):
            return s_in._replace(oom=jnp.bool_(True))

        s2 = jax.lax.cond(s.corrupt, _keep, _oom, s)
        return s2, jnp.zeros((4,), dtype=jnp.uint32)

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
        def _keep(s_in):
            return s_in

        def _corrupt(s_in):
            return s_in._replace(corrupt=jnp.bool_(True))

        return jax.lax.cond(s.corrupt, _keep, _corrupt, s)

    return jax.lax.cond(ok & (~state.oom) & (~state.corrupt), _do, _fail, state)


def _host_flag(value: jnp.ndarray) -> bool:
    return bool(jax.device_get(value))


def _alloc_nodes(state: ICState, count: int) -> Tuple[ICState, jnp.ndarray]:
    n = int(count)
    if n == 0:
        return state, jnp.zeros((0,), dtype=jnp.uint32)
    if _host_flag(state.corrupt):
        return state, jnp.zeros((n,), dtype=jnp.uint32)
    free_top = int(state.free_top)
    if free_top < n or _host_flag(state.oom):
        return state._replace(oom=jnp.bool_(True)), jnp.zeros((n,), dtype=jnp.uint32)
    idx = state.free_stack[free_top - n:free_top]
    free_top = free_top - n
    return state._replace(free_top=jnp.uint32(free_top)), idx


def ic_alloc(state: ICState, count: int, node_type: jnp.uint8) -> Tuple[ICState, jnp.ndarray]:
    state, nodes = _alloc_nodes(state, count)
    if nodes.size == 0 or _host_flag(state.oom) or _host_flag(state.corrupt):
        return state, nodes
    node_type_arr = state.node_type.at[nodes].set(node_type)
    ports = state.ports.at[nodes].set(jnp.uint32(0))
    return state._replace(node_type=node_type_arr, ports=ports), nodes


def _free_nodes(state: ICState, nodes: jnp.ndarray) -> ICState:
    if nodes.size == 0:
        return state
    if _host_flag(state.corrupt):
        return state
    count = int(nodes.shape[0])
    free_top = int(state.free_top)
    cap = int(state.free_stack.shape[0])
    if free_top + count > cap:
        return state._replace(corrupt=jnp.bool_(True))
    free_stack = state.free_stack
    free_stack = free_stack.at[free_top:free_top + count].set(nodes)
    return state._replace(free_stack=free_stack, free_top=jnp.uint32(free_top + count))


__all__ = [
    "TYPE_FREE",
    "TYPE_ERA",
    "TYPE_CON",
    "TYPE_DUP",
    "PORT_PRINCIPAL",
    "PORT_AUX_LEFT",
    "PORT_AUX_RIGHT",
    "ICState",
    "ic_alloc_jax",
    "encode_port",
    "decode_port",
    "ic_init",
    "ic_wire",
    "ic_wire_jax",
    "ic_wire_ptrs_jax",
    "ic_wire_jax_safe",
    "ic_wire_pairs_jax",
    "ic_wire_ptr_pairs_jax",
    "ic_wire_star_jax",
    "ic_find_active_pairs",
    "ic_compact_active_pairs",
    "ic_alloc",
]
