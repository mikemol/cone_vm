import jax
import jax.numpy as jnp
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


def _connect_ptrs(ports: jnp.ndarray, ptr_a: jnp.ndarray, ptr_b: jnp.ndarray) -> jnp.ndarray:
    node_a, port_a = decode_port(ptr_a)
    node_b, port_b = decode_port(ptr_b)
    ports = ports.at[node_a, port_a].set(ptr_b)
    ports = ports.at[node_b, port_b].set(ptr_a)
    return ports


class ICState(NamedTuple):
    node_type: jnp.ndarray
    ports: jnp.ndarray
    free_stack: jnp.ndarray
    free_top: jnp.ndarray


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
    free_top = jnp.array(capacity, dtype=jnp.uint32)
    return ICState(
        node_type=node_type,
        ports=ports,
        free_stack=free_stack,
        free_top=free_top,
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
def ic_find_active_pairs(state: ICState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return indices of nodes in active principal-principal pairs."""
    ports = state.ports
    n = ports.shape[0]
    idx = jnp.arange(n, dtype=jnp.uint32)
    ptr = ports[:, 0]
    tgt_node, tgt_port = decode_port(ptr)
    is_principal = tgt_port == PORT_PRINCIPAL
    back = ports[tgt_node, 0]
    back_node, back_port = decode_port(back)
    mutual = (back_node == idx) & (back_port == PORT_PRINCIPAL)
    active = is_principal & mutual & (idx < tgt_node)
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


def ic_apply_template(
    state: ICState, node_a: int, node_b: int, template_id: jnp.ndarray
) -> ICState:
    tmpl = int(template_id)
    if tmpl == int(TEMPLATE_NONE):
        return state
    if tmpl == int(TEMPLATE_ANNIHILATE):
        return ic_apply_annihilate(state, node_a, node_b)
    if tmpl == int(TEMPLATE_ERASE):
        return ic_apply_erase(state, node_a, node_b)
    if tmpl == int(TEMPLATE_COMMUTE):
        return ic_apply_commute(state, node_a, node_b)
    raise ValueError(f"Unsupported template_id={tmpl}")


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
    free_stack = state.free_stack
    free_stack = free_stack.at[free_top:free_top + count].set(nodes)
    return state._replace(free_stack=free_stack, free_top=jnp.uint32(free_top + count))


def ic_apply_erase(state: ICState, node_a: int, node_b: int) -> ICState:
    # Erase: replace binary node with two erasers on auxiliary wires.
    type_a = state.node_type[node_a]
    type_b = state.node_type[node_b]
    if type_a == TYPE_ERA:
        era = node_a
        target = node_b
    elif type_b == TYPE_ERA:
        era = node_b
        target = node_a
    else:
        raise ValueError("ic_apply_erase expects an ERA node")
    ports = state.ports
    aux_left = ports[target, 1]
    aux_right = ports[target, 2]
    state, eras = ic_alloc(state, 2, TYPE_ERA)
    if eras.size == 2:
        ports = state.ports
        ports = _connect_ptrs(ports, encode_port(eras[0], PORT_PRINCIPAL), aux_left)
        ports = _connect_ptrs(ports, encode_port(eras[1], PORT_PRINCIPAL), aux_right)
        ports = ports.at[era].set(jnp.uint32(0))
        ports = ports.at[target].set(jnp.uint32(0))
        node_type = state.node_type
        node_type = node_type.at[era].set(TYPE_FREE)
        node_type = node_type.at[target].set(TYPE_FREE)
        state = state._replace(node_type=node_type, ports=ports)
        state = _free_nodes(state, jnp.asarray([era, target], dtype=jnp.uint32))
    return state


def ic_apply_commute(state: ICState, node_a: int, node_b: int) -> ICState:
    # Commute: con/dup cross-wiring (partial scaffold).
    type_a = state.node_type[node_a]
    type_b = state.node_type[node_b]
    if type_a == TYPE_CON and type_b == TYPE_DUP:
        con = node_a
        dup = node_b
    elif type_a == TYPE_DUP and type_b == TYPE_CON:
        con = node_b
        dup = node_a
    else:
        raise ValueError("ic_apply_commute expects a CON/DUP pair")
    ports = state.ports
    con_left = ports[con, 1]
    con_right = ports[con, 2]
    dup_left = ports[dup, 1]
    dup_right = ports[dup, 2]
    state, con_nodes = ic_alloc(state, 2, TYPE_CON)
    state, dup_nodes = ic_alloc(state, 2, TYPE_DUP)
    if con_nodes.size == 2 and dup_nodes.size == 2:
        c0, c1 = con_nodes
        d0, d1 = dup_nodes
        ports = state.ports
        ports = _connect_ptrs(
            ports,
            encode_port(c0, PORT_PRINCIPAL),
            encode_port(d0, PORT_PRINCIPAL),
        )
        ports = _connect_ptrs(
            ports,
            encode_port(c1, PORT_PRINCIPAL),
            encode_port(d1, PORT_PRINCIPAL),
        )
        ports = _connect_ptrs(ports, encode_port(c0, PORT_AUX_LEFT), dup_left)
        ports = _connect_ptrs(ports, encode_port(c1, PORT_AUX_LEFT), dup_right)
        ports = _connect_ptrs(ports, encode_port(d0, PORT_AUX_LEFT), con_left)
        ports = _connect_ptrs(ports, encode_port(d1, PORT_AUX_LEFT), con_right)
        ports = _connect_ptrs(
            ports,
            encode_port(c0, PORT_AUX_RIGHT),
            encode_port(d0, PORT_AUX_RIGHT),
        )
        ports = _connect_ptrs(
            ports,
            encode_port(c1, PORT_AUX_RIGHT),
            encode_port(d1, PORT_AUX_RIGHT),
        )
        ports = ports.at[con].set(jnp.uint32(0))
        ports = ports.at[dup].set(jnp.uint32(0))
        node_type = state.node_type
        node_type = node_type.at[con].set(TYPE_FREE)
        node_type = node_type.at[dup].set(TYPE_FREE)
        state = state._replace(node_type=node_type, ports=ports)
        state = _free_nodes(state, jnp.asarray([con, dup], dtype=jnp.uint32))
    return state
