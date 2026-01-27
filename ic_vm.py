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
    free_top = jnp.array(
        capacity if capacity < 3 else max(capacity - 1, 0), dtype=jnp.uint32
    )
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
        s, alloc, freed, tmpl_counts = carry
        node_a = pairs[i]
        node_b = decode_port(s.ports[node_a, PORT_PRINCIPAL])[0]
        tmpl = ic_rule_for_types(s.node_type[node_a], s.node_type[node_b])[1]
        s2 = ic_apply_template(s, node_a, node_b, tmpl)
        ok = (~s.oom) & (~s2.oom)
        tmpl_i = tmpl.astype(jnp.int32)
        tmpl_counts = tmpl_counts.at[tmpl_i].add(ok.astype(jnp.uint32))
        alloc_delta = jnp.where(
            (tmpl == TEMPLATE_ERASE) & ok, jnp.uint32(2), jnp.uint32(0)
        )
        alloc_delta = jnp.where(
            (tmpl == TEMPLATE_COMMUTE) & ok, jnp.uint32(4), alloc_delta
        )
        freed_delta = jnp.where(
            (tmpl != TEMPLATE_NONE) & ok, jnp.uint32(2), jnp.uint32(0)
        )
        return s2, alloc + alloc_delta, freed + freed_delta, tmpl_counts

    def _apply(s):
        init = (s, jnp.uint32(0), jnp.uint32(0), zero_stats.template_counts)
        s2, alloc, freed, tmpl_counts = jax.lax.fori_loop(
            0, count_i, body, init
        )
        stats = ICRewriteStats(
            active_pairs=count,
            alloc_nodes=alloc,
            freed_nodes=freed,
            template_counts=tmpl_counts,
        )
        return s2, stats

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
