import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from prism_core import alloc as _alloc

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


def _init_nodes(state: ICState, nodes: jnp.ndarray, node_type: jnp.uint8) -> ICState:
    node_type_arr = state.node_type.at[nodes].set(node_type)
    ports = state.ports.at[nodes].set(jnp.uint32(0))
    return state._replace(node_type=node_type_arr, ports=ports)

# Allocator aliases (shared implementation in prism_core.alloc).
ic_alloc_jax = _alloc.alloc_jax
_alloc2 = _alloc.alloc2
_alloc4 = _alloc.alloc4
_free2 = _alloc.free2
_host_flag = _alloc.host_flag
_alloc_nodes = _alloc.alloc_nodes
ic_alloc = _alloc.alloc_host
_free_nodes = _alloc.free_nodes


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
