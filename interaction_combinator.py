"""
Interaction Combinator (IC) engine scaffolding.

This is a minimal data-model placeholder for the in-8 tensor/rule-table track.
It intentionally avoids operational semantics until the rule table and port
wiring pipeline are formalized.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp

IC_FREE = 0
IC_ERA = 1
IC_CON = 2
IC_DUP = 3

PORT_P = 0
PORT_L = 1
PORT_R = 2
PORT_ARITY = 3

MAX_NEW_NODES = 4
EXT_A_L = 4
EXT_A_R = 5
EXT_B_L = 6
EXT_B_R = 7
EXT_COUNT = 4
TEMPLATE_NONE = -1


@dataclass(frozen=True)
class ICState:
    # node_type: int8 array [N]
    # port: int32 array [N,3] (encoded port refs; see encode_port)
    node_type: jnp.ndarray
    port: jnp.ndarray


@dataclass(frozen=True)
class RuleTable:
    # lhs: int8 array [R,2] (principal pair types)
    # alloc_count: int32 array [R] (0,2,4)
    # rhs_node_type: int8 array [R,4] (scaffold: up to 4 nodes)
    # rhs_port_map: int32 array [R,4,3] (wiring template slots)
    # ext_port_map: int32 array [R,4] (external neighbor rewiring)
    lhs: jnp.ndarray
    alloc_count: jnp.ndarray
    rhs_node_type: jnp.ndarray
    rhs_port_map: jnp.ndarray
    ext_port_map: jnp.ndarray


@dataclass(frozen=True)
class ICArena:
    state: ICState
    free_stack: jnp.ndarray
    free_count: jnp.ndarray


@dataclass(frozen=True)
class RewritePlan:
    alloc_ok: jnp.ndarray
    alloc_count: jnp.ndarray
    new_ids: jnp.ndarray
    new_types: jnp.ndarray
    new_ports: jnp.ndarray
    ext_targets: jnp.ndarray
    ext_values: jnp.ndarray


@dataclass(frozen=True)
class RewritePlanBatch:
    alloc_ok: jnp.ndarray
    alloc_count: jnp.ndarray
    new_ids: jnp.ndarray
    new_types: jnp.ndarray
    new_ports: jnp.ndarray
    ext_targets: jnp.ndarray
    ext_values: jnp.ndarray


def apply_rewrite_plan(arena: ICArena, plan: RewritePlan) -> ICArena:
    state = arena.state
    if not bool(plan.alloc_ok):
        return arena
    active_mask = jnp.arange(MAX_NEW_NODES, dtype=jnp.int32) < plan.alloc_count
    ids = plan.new_ids
    valid_ids = active_mask & (ids >= 0)
    safe_ids = jnp.where(valid_ids, ids, jnp.int32(0))
    new_node_type = state.node_type.at[safe_ids].set(
        jnp.where(valid_ids, plan.new_types, state.node_type[0])
    )
    new_ports = state.port
    for port_idx in range(PORT_ARITY):
        values = plan.new_ports[:, port_idx]
        new_ports = new_ports.at[safe_ids, port_idx].set(
            jnp.where(valid_ids, values, new_ports[0, port_idx])
        )
    ext_nodes = plan.ext_targets[:, 0]
    ext_ports = plan.ext_targets[:, 1]
    ext_valid = (ext_nodes >= 0) & (plan.ext_values != jnp.int32(0))
    ext_nodes_safe = jnp.where(ext_valid, ext_nodes, jnp.int32(0))
    ext_ports_safe = jnp.where(ext_valid, ext_ports, jnp.int32(0))
    ext_values_safe = jnp.where(ext_valid, plan.ext_values, new_ports[0, 0])
    new_ports = new_ports.at[ext_nodes_safe, ext_ports_safe].set(ext_values_safe)
    new_state = ICState(node_type=new_node_type, port=new_ports)
    return ICArena(state=new_state, free_stack=arena.free_stack, free_count=arena.free_count)


def apply_rewrite_plan_batch(arena: ICArena, plan: RewritePlanBatch) -> ICArena:
    state = arena.state
    if plan.alloc_ok.size == 0:
        return arena
    active_mask = (
        jnp.arange(MAX_NEW_NODES, dtype=jnp.int32)[None, :] < plan.alloc_count[:, None]
    )
    active_mask = active_mask & plan.alloc_ok[:, None]
    ids = plan.new_ids
    valid_ids = active_mask & (ids >= 0)
    safe_ids = jnp.where(valid_ids, ids, jnp.int32(0))
    safe_ids_flat = safe_ids.reshape(-1)
    valid_flat = valid_ids.reshape(-1)
    new_types_flat = plan.new_types.reshape(-1)
    new_node_type = state.node_type.at[safe_ids_flat].set(
        jnp.where(valid_flat, new_types_flat, state.node_type[0])
    )
    new_ports = state.port
    for port_idx in range(PORT_ARITY):
        values_flat = plan.new_ports[:, :, port_idx].reshape(-1)
        new_ports = new_ports.at[safe_ids_flat, port_idx].set(
            jnp.where(valid_flat, values_flat, new_ports[0, port_idx])
        )
    ext_nodes = plan.ext_targets[:, :, 0]
    ext_ports = plan.ext_targets[:, :, 1]
    ext_valid = plan.alloc_ok[:, None] & (ext_nodes >= 0) & (plan.ext_values != 0)
    ext_nodes_safe = jnp.where(ext_valid, ext_nodes, jnp.int32(0)).reshape(-1)
    ext_ports_safe = jnp.where(ext_valid, ext_ports, jnp.int32(0)).reshape(-1)
    ext_values_safe = jnp.where(
        ext_valid, plan.ext_values, new_ports[0, 0]
    ).reshape(-1)
    new_ports = new_ports.at[ext_nodes_safe, ext_ports_safe].set(ext_values_safe)
    new_state = ICState(node_type=new_node_type, port=new_ports)
    return ICArena(state=new_state, free_stack=arena.free_stack, free_count=arena.free_count)


def rewrite_one_step(
    arena: ICArena, table: RuleTable
) -> tuple[ICArena, jnp.ndarray]:
    pairs, count = find_active_pairs(arena.state)
    if int(count) == 0:
        return arena, jnp.array(False)
    pair = pairs[0]
    rule_idx, matched, swapped = match_active_pairs(
        arena.state, pairs, count, table
    )
    if not bool(matched[0]):
        return arena, jnp.array(False)
    arena, plan = build_rewrite_plan(arena, pair, rule_idx[0], swapped[0], table)
    if not bool(plan.alloc_ok):
        return arena, jnp.array(False)
    arena = apply_rewrite_plan(arena, plan)
    arena = free_nodes(arena, pair)
    return arena, jnp.array(True)


def rewrite_batch(
    arena: ICArena, table: RuleTable, max_pairs: int | None = None
) -> tuple[ICArena, jnp.ndarray]:
    pairs, count = find_active_pairs(arena.state)
    limit = int(count)
    if max_pairs is not None:
        limit = min(limit, int(max_pairs))
    rewrites = 0
    for i in range(limit):
        pair = pairs[i]
        rule_idx, matched, swapped = match_active_pairs(
            arena.state, pairs[i : i + 1], jnp.int32(1), table
        )
        if not bool(matched[0]):
            continue
        arena, plan = build_rewrite_plan(
            arena, pair, rule_idx[0], swapped[0], table
        )
        if not bool(plan.alloc_ok):
            continue
        arena = apply_rewrite_plan(arena, plan)
        arena = free_nodes(arena, pair)
        rewrites += 1
    return arena, jnp.array(rewrites, dtype=jnp.int32)


def rewrite_batch_vectorized(
    arena: ICArena, table: RuleTable, max_pairs: int | None = None
) -> tuple[ICArena, jnp.ndarray]:
    pairs, count = find_active_pairs(arena.state)
    limit = int(count)
    if max_pairs is not None:
        limit = min(limit, int(max_pairs))
    if limit == 0:
        return arena, jnp.array(0, dtype=jnp.int32)
    pairs = pairs[:limit]
    rule_idx, matched, swapped = match_active_pairs(
        arena.state, pairs, jnp.int32(limit), table
    )
    arena, plan = build_rewrite_plan_batch(arena, pairs, rule_idx, swapped, table)
    if plan.alloc_ok.size == 0:
        return arena, jnp.array(0, dtype=jnp.int32)
    arena = apply_rewrite_plan_batch(arena, plan)
    arena = free_nodes_masked(arena, pairs, plan.alloc_ok)
    rewrites = jnp.sum(plan.alloc_ok.astype(jnp.int32))
    return arena, rewrites


def rewrite_n_steps(
    arena: ICArena,
    table: RuleTable,
    steps: int,
    max_pairs: int | None = None,
) -> tuple[ICArena, jnp.ndarray]:
    total = 0
    for _ in range(int(steps)):
        arena, rewrites = rewrite_batch(arena, table, max_pairs=max_pairs)
        step_count = int(rewrites)
        total += step_count
        if step_count == 0:
            break
    return arena, jnp.array(total, dtype=jnp.int32)


def encode_port(node_idx: int, port_idx: int) -> int:
    if port_idx < 0 or port_idx >= PORT_ARITY:
        raise ValueError("port_idx out of range")
    if node_idx < 0:
        raise ValueError("node_idx out of range")
    return node_idx * PORT_ARITY + port_idx


def decode_port(ref: int) -> tuple[int, int]:
    if ref < 0:
        raise ValueError("ref out of range")
    return ref // PORT_ARITY, ref % PORT_ARITY


def init_ic_state(capacity: int) -> ICState:
    node_type = jnp.full((capacity,), IC_FREE, dtype=jnp.int8)
    port = jnp.zeros((capacity, PORT_ARITY), dtype=jnp.int32)
    return ICState(node_type=node_type, port=port)


def init_ic_arena(capacity: int) -> ICArena:
    state = init_ic_state(capacity)
    free_stack = jnp.arange(capacity, dtype=jnp.int32)
    free_count = jnp.array(capacity, dtype=jnp.int32)
    return ICArena(state=state, free_stack=free_stack, free_count=free_count)


def validate_ic_state(state: ICState) -> None:
    if state.node_type.ndim != 1:
        raise ValueError("ICState.node_type must be 1D")
    if state.port.ndim != 2 or state.port.shape[1] != PORT_ARITY:
        raise ValueError("ICState.port must be [N,3]")
    if state.port.shape[0] != state.node_type.shape[0]:
        raise ValueError("ICState node_type/port length mismatch")


def find_active_pairs(state: ICState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return active principal-port pairs.

    Returns:
        pairs: int32 array [N,2], only the first `count` rows are valid.
        count: int32 scalar count of active pairs.
    """
    node_count = state.node_type.shape[0]
    node_ids = jnp.arange(node_count, dtype=jnp.int32)
    port_p = state.port[:, PORT_P]
    neighbor = port_p // PORT_ARITY
    neighbor_port = port_p % PORT_ARITY
    neighbor_valid = (neighbor >= 0) & (neighbor < node_count)
    neighbor_safe = jnp.clip(neighbor, 0, node_count - 1)
    back_ref = state.port[neighbor_safe, PORT_P]
    expected_back = node_ids * PORT_ARITY + PORT_P
    self_active = state.node_type != IC_FREE
    neighbor_active = jnp.where(
        neighbor_valid, state.node_type[neighbor_safe] != IC_FREE, False
    )
    mutual = neighbor_valid & (neighbor_port == PORT_P) & (back_ref == expected_back)
    pair_mask = self_active & neighbor_active & mutual & (node_ids < neighbor)
    count = jnp.sum(pair_mask).astype(jnp.int32)
    idx = jnp.nonzero(pair_mask, size=node_count, fill_value=0)[0].astype(jnp.int32)
    left = idx
    right = neighbor_safe[idx]
    pairs = jnp.stack([left, right], axis=1)
    return pairs, count


def match_rule_indices(
    left_types: jnp.ndarray, right_types: jnp.ndarray, table: RuleTable
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rule_count = table.lhs.shape[0]
    if rule_count == 0:
        rule_idx = jnp.full(left_types.shape, -1, dtype=jnp.int32)
        matched = jnp.zeros(left_types.shape, dtype=jnp.bool_)
        swapped = jnp.zeros(left_types.shape, dtype=jnp.bool_)
        return rule_idx, matched, swapped

    lhs0 = table.lhs[:, 0]
    lhs1 = table.lhs[:, 1]
    left = left_types[:, None]
    right = right_types[:, None]
    eq_direct = (left == lhs0) & (right == lhs1)
    eq_swap = (left == lhs1) & (right == lhs0)
    match = eq_direct | eq_swap
    matched = jnp.any(match, axis=1)
    rule_idx = jnp.argmax(match, axis=1).astype(jnp.int32)
    rule_idx = jnp.where(matched, rule_idx, jnp.int32(-1))
    direct_hit = jnp.any(eq_direct, axis=1)
    swapped = jnp.any(eq_swap, axis=1) & (~direct_hit) & matched
    return rule_idx, matched, swapped


def match_active_pairs(
    state: ICState, pairs: jnp.ndarray, count: jnp.ndarray, table: RuleTable
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    left = pairs[:, 0]
    right = pairs[:, 1]
    left_types = state.node_type[left]
    right_types = state.node_type[right]
    rule_idx, matched, swapped = match_rule_indices(left_types, right_types, table)
    valid = jnp.arange(pairs.shape[0], dtype=jnp.int32) < count
    rule_idx = jnp.where(valid, rule_idx, jnp.int32(-1))
    matched = matched & valid
    swapped = swapped & valid
    return rule_idx, matched, swapped


def validate_ic_arena(arena: ICArena) -> None:
    validate_ic_state(arena.state)
    if arena.free_stack.ndim != 1:
        raise ValueError("ICArena.free_stack must be 1D")
    if arena.free_stack.shape[0] != arena.state.node_type.shape[0]:
        raise ValueError("ICArena.free_stack length mismatch")
    if arena.free_count.ndim != 0:
        raise ValueError("ICArena.free_count must be scalar")


def _pair_roles(left: jnp.ndarray, right: jnp.ndarray, swapped: jnp.ndarray):
    a = jnp.where(swapped, right, left)
    b = jnp.where(swapped, left, right)
    return a, b


def _extract_external_refs(state: ICState, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    ext = jnp.stack(
        [
            state.port[a, PORT_L],
            state.port[a, PORT_R],
            state.port[b, PORT_L],
            state.port[b, PORT_R],
        ],
        axis=0,
    )
    return ext.astype(jnp.int32)


def _extract_external_refs_batch(
    state: ICState, a: jnp.ndarray, b: jnp.ndarray
) -> jnp.ndarray:
    ext = jnp.stack(
        [
            state.port[a, PORT_L],
            state.port[a, PORT_R],
            state.port[b, PORT_L],
            state.port[b, PORT_R],
        ],
        axis=1,
    )
    return ext.astype(jnp.int32)


def _resolve_template_ports(
    rhs_port_map: jnp.ndarray, new_ids: jnp.ndarray, ext_refs: jnp.ndarray
) -> jnp.ndarray:
    port_idx = jnp.arange(PORT_ARITY, dtype=jnp.int32)[None, :]
    internal = (rhs_port_map >= 0) & (rhs_port_map < MAX_NEW_NODES)
    ext = (rhs_port_map >= MAX_NEW_NODES) & (
        rhs_port_map < MAX_NEW_NODES + EXT_COUNT
    )
    internal_node = jnp.clip(rhs_port_map, 0, MAX_NEW_NODES - 1)
    internal_ref = new_ids[internal_node] * jnp.int32(PORT_ARITY) + port_idx
    ext_idx = jnp.clip(rhs_port_map - MAX_NEW_NODES, 0, EXT_COUNT - 1)
    ext_ref = ext_refs[ext_idx]
    out = jnp.where(internal, internal_ref, jnp.where(ext, ext_ref, jnp.int32(0)))
    return out.astype(jnp.int32)


def _resolve_external_updates(
    ext_port_map: jnp.ndarray, new_ids: jnp.ndarray, ext_refs: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ext_nodes = ext_refs // jnp.int32(PORT_ARITY)
    ext_ports = ext_refs % jnp.int32(PORT_ARITY)
    ext_targets = jnp.stack([ext_nodes, ext_ports], axis=1).astype(jnp.int32)
    use_new = (ext_port_map >= 0) & (ext_port_map < MAX_NEW_NODES)
    use_ext = (ext_port_map >= MAX_NEW_NODES) & (
        ext_port_map < MAX_NEW_NODES + EXT_COUNT
    )
    new_idx = jnp.clip(ext_port_map, 0, MAX_NEW_NODES - 1)
    new_ref = new_ids[new_idx] * jnp.int32(PORT_ARITY) + jnp.int32(PORT_P)
    ext_idx = jnp.clip(ext_port_map - MAX_NEW_NODES, 0, EXT_COUNT - 1)
    ext_ref = ext_refs[ext_idx]
    values = jnp.where(use_new, new_ref, jnp.where(use_ext, ext_ref, jnp.int32(0)))
    return ext_targets, values.astype(jnp.int32)


def allocate_from_arena(
    arena: ICArena, count: jnp.ndarray
) -> tuple[ICArena, jnp.ndarray, jnp.ndarray]:
    count = jnp.minimum(count, jnp.int32(MAX_NEW_NODES))
    available = arena.free_count
    alloc_ok = available >= count
    safe_count = jnp.where(alloc_ok, count, jnp.int32(0))
    start = jnp.maximum(available - safe_count, 0)
    idx = jnp.arange(MAX_NEW_NODES, dtype=jnp.int32)
    use = idx < safe_count
    stack_idx = start + idx
    safe_idx = jnp.clip(stack_idx, 0, arena.free_stack.shape[0] - 1)
    new_ids = jnp.where(use, arena.free_stack[safe_idx], jnp.int32(0))
    new_free_count = jnp.where(alloc_ok, available - safe_count, available)
    new_arena = ICArena(
        state=arena.state, free_stack=arena.free_stack, free_count=new_free_count
    )
    return new_arena, new_ids.astype(jnp.int32), alloc_ok


def allocate_batch_from_arena(
    arena: ICArena, counts: jnp.ndarray
) -> tuple[ICArena, jnp.ndarray, jnp.ndarray]:
    counts = jnp.asarray(counts, dtype=jnp.int32)
    counts = jnp.minimum(counts, jnp.int32(MAX_NEW_NODES))
    available = arena.free_count
    prefix = jnp.cumsum(counts)
    alloc_ok = prefix <= available
    alloc_counts = jnp.where(alloc_ok, counts, jnp.int32(0))
    total_alloc = jnp.sum(alloc_counts).astype(jnp.int32)
    start = available - prefix
    idx = jnp.arange(MAX_NEW_NODES, dtype=jnp.int32)[None, :]
    use = (idx < counts[:, None]) & alloc_ok[:, None]
    stack_idx = start[:, None] + idx
    safe_idx = jnp.clip(stack_idx, 0, arena.free_stack.shape[0] - 1)
    new_ids = jnp.where(use, arena.free_stack[safe_idx], jnp.int32(0))
    new_free_count = available - total_alloc
    new_arena = ICArena(
        state=arena.state, free_stack=arena.free_stack, free_count=new_free_count
    )
    return new_arena, new_ids.astype(jnp.int32), alloc_ok


def _free_nodes_with_count(
    arena: ICArena, ids: jnp.ndarray, num: jnp.ndarray
) -> ICArena:
    ids = jnp.asarray(ids, dtype=jnp.int32)
    num = jnp.minimum(jnp.asarray(num, dtype=jnp.int32), jnp.int32(ids.shape[0]))
    cap = jnp.asarray(arena.free_stack.shape[0], dtype=jnp.int32)
    free_count = arena.free_count
    space = jnp.maximum(cap - free_count, 0)
    num_free = jnp.minimum(num, space)
    idx = jnp.arange(ids.shape[0], dtype=jnp.int32)
    use = idx < num_free
    stack_pos = free_count + idx
    safe_pos = jnp.clip(stack_pos, 0, cap - 1)
    free_stack = arena.free_stack
    new_vals = jnp.where(use, ids, free_stack[safe_pos])
    free_stack = free_stack.at[safe_pos].set(new_vals)
    free_count = free_count + num_free

    safe_ids = jnp.where(use, ids, jnp.int32(0))
    node_type = arena.state.node_type.at[safe_ids].set(
        jnp.where(use, IC_FREE, arena.state.node_type[0])
    )
    port = arena.state.port
    for port_idx in range(PORT_ARITY):
        port = port.at[safe_ids, port_idx].set(
            jnp.where(use, jnp.int32(0), port[0, port_idx])
        )
    state = ICState(node_type=node_type, port=port)
    return ICArena(state=state, free_stack=free_stack, free_count=free_count)


def free_nodes(arena: ICArena, ids: jnp.ndarray) -> ICArena:
    num = jnp.asarray(ids.shape[0], dtype=jnp.int32)
    return _free_nodes_with_count(arena, ids, num)


def free_nodes_masked(
    arena: ICArena, ids: jnp.ndarray, mask: jnp.ndarray
) -> ICArena:
    ids = jnp.asarray(ids, dtype=jnp.int32)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    mask = jnp.broadcast_to(mask, ids.shape)
    ids_flat = ids.reshape(-1)
    mask_flat = mask.reshape(-1)
    idx = jnp.nonzero(
        mask_flat, size=ids_flat.shape[0], fill_value=0
    )[0].astype(jnp.int32)
    ids_packed = ids_flat[idx]
    num = jnp.sum(mask_flat).astype(jnp.int32)
    return _free_nodes_with_count(arena, ids_packed, num)


def init_rule_table_empty() -> RuleTable:
    lhs = jnp.zeros((0, 2), dtype=jnp.int8)
    alloc_count = jnp.zeros((0,), dtype=jnp.int32)
    rhs_node_type = jnp.zeros((0, MAX_NEW_NODES), dtype=jnp.int8)
    rhs_port_map = jnp.zeros((0, MAX_NEW_NODES, PORT_ARITY), dtype=jnp.int32)
    ext_port_map = jnp.zeros((0, EXT_COUNT), dtype=jnp.int32)
    return RuleTable(
        lhs=lhs,
        alloc_count=alloc_count,
        rhs_node_type=rhs_node_type,
        rhs_port_map=rhs_port_map,
        ext_port_map=ext_port_map,
    )


def init_rule_table_core() -> RuleTable:
    lhs = jnp.array(
        [
            [IC_DUP, IC_CON],  # commutation
            [IC_CON, IC_CON],  # annihilation
            [IC_DUP, IC_DUP],  # annihilation
            [IC_ERA, IC_CON],  # erasure
            [IC_ERA, IC_DUP],  # erasure
        ],
        dtype=jnp.int8,
    )
    alloc_count = jnp.array([4, 0, 0, 2, 2], dtype=jnp.int32)
    rhs_node_type = jnp.full((lhs.shape[0], MAX_NEW_NODES), IC_FREE, dtype=jnp.int8)
    rhs_port_map = jnp.full(
        (lhs.shape[0], MAX_NEW_NODES, PORT_ARITY), TEMPLATE_NONE, dtype=jnp.int32
    )
    ext_port_map = jnp.full((lhs.shape[0], EXT_COUNT), TEMPLATE_NONE, dtype=jnp.int32)

    # Commutation template: DUP (A) with CON (B).
    rhs_node_type = rhs_node_type.at[0].set(
        jnp.array([IC_CON, IC_CON, IC_DUP, IC_DUP], dtype=jnp.int8)
    )
    rhs_port_map = rhs_port_map.at[0, 0].set(jnp.array([EXT_A_L, 2, 3]))
    rhs_port_map = rhs_port_map.at[0, 1].set(jnp.array([EXT_A_R, 2, 3]))
    rhs_port_map = rhs_port_map.at[0, 2].set(jnp.array([EXT_B_L, 0, 1]))
    rhs_port_map = rhs_port_map.at[0, 3].set(jnp.array([EXT_B_R, 0, 1]))
    ext_port_map = ext_port_map.at[0].set(
        jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    )

    # Annihilation templates: wire A.L <-> B.L, A.R <-> B.R.
    ext_port_map = ext_port_map.at[1].set(
        jnp.array([EXT_B_L, EXT_B_R, EXT_A_L, EXT_A_R], dtype=jnp.int32)
    )
    ext_port_map = ext_port_map.at[2].set(
        jnp.array([EXT_B_L, EXT_B_R, EXT_A_L, EXT_A_R], dtype=jnp.int32)
    )

    # Erasure templates: spawn two erasers for B's aux ports.
    rhs_node_type = rhs_node_type.at[3].set(
        jnp.array([IC_ERA, IC_ERA, IC_FREE, IC_FREE], dtype=jnp.int8)
    )
    rhs_node_type = rhs_node_type.at[4].set(
        jnp.array([IC_ERA, IC_ERA, IC_FREE, IC_FREE], dtype=jnp.int8)
    )
    rhs_port_map = rhs_port_map.at[3, 0].set(jnp.array([EXT_B_L, TEMPLATE_NONE, TEMPLATE_NONE]))
    rhs_port_map = rhs_port_map.at[3, 1].set(jnp.array([EXT_B_R, TEMPLATE_NONE, TEMPLATE_NONE]))
    rhs_port_map = rhs_port_map.at[4, 0].set(jnp.array([EXT_B_L, TEMPLATE_NONE, TEMPLATE_NONE]))
    rhs_port_map = rhs_port_map.at[4, 1].set(jnp.array([EXT_B_R, TEMPLATE_NONE, TEMPLATE_NONE]))
    ext_port_map = ext_port_map.at[3].set(
        jnp.array([TEMPLATE_NONE, TEMPLATE_NONE, 0, 1], dtype=jnp.int32)
    )
    ext_port_map = ext_port_map.at[4].set(
        jnp.array([TEMPLATE_NONE, TEMPLATE_NONE, 0, 1], dtype=jnp.int32)
    )

    return RuleTable(
        lhs=lhs,
        alloc_count=alloc_count,
        rhs_node_type=rhs_node_type,
        rhs_port_map=rhs_port_map,
        ext_port_map=ext_port_map,
    )


def validate_rule_table(table: RuleTable) -> None:
    if table.lhs.ndim != 2 or table.lhs.shape[1] != 2:
        raise ValueError("RuleTable.lhs must be [R,2]")
    if table.alloc_count.ndim != 1:
        raise ValueError("RuleTable.alloc_count must be [R]")
    if table.rhs_node_type.ndim != 2 or table.rhs_node_type.shape[1] != MAX_NEW_NODES:
        raise ValueError("RuleTable.rhs_node_type must be [R,4]")
    if table.rhs_port_map.ndim != 3 or table.rhs_port_map.shape[1:] != (
        MAX_NEW_NODES,
        PORT_ARITY,
    ):
        raise ValueError("RuleTable.rhs_port_map must be [R,4,3]")
    if table.ext_port_map.ndim != 2 or table.ext_port_map.shape[1] != EXT_COUNT:
        raise ValueError("RuleTable.ext_port_map must be [R,4]")
    if table.lhs.shape[0] != table.alloc_count.shape[0]:
        raise ValueError("RuleTable lhs/alloc_count length mismatch")
    if table.lhs.shape[0] != table.rhs_node_type.shape[0]:
        raise ValueError("RuleTable lhs/rhs_node_type length mismatch")
    if table.lhs.shape[0] != table.rhs_port_map.shape[0]:
        raise ValueError("RuleTable lhs/rhs_port_map length mismatch")
    if table.lhs.shape[0] != table.ext_port_map.shape[0]:
        raise ValueError("RuleTable lhs/ext_port_map length mismatch")


def build_rewrite_plan(
    arena: ICArena,
    pair: jnp.ndarray,
    rule_idx: jnp.ndarray,
    swapped: jnp.ndarray,
    table: RuleTable,
) -> tuple[ICArena, RewritePlan]:
    if table.lhs.shape[0] == 0:
        empty = jnp.zeros((MAX_NEW_NODES,), dtype=jnp.int32)
        plan = RewritePlan(
            alloc_ok=jnp.bool_(False),
            alloc_count=jnp.int32(0),
            new_ids=empty,
            new_types=jnp.full((MAX_NEW_NODES,), IC_FREE, dtype=jnp.int8),
            new_ports=jnp.zeros((MAX_NEW_NODES, PORT_ARITY), dtype=jnp.int32),
            ext_targets=jnp.zeros((EXT_COUNT, 2), dtype=jnp.int32),
            ext_values=jnp.zeros((EXT_COUNT,), dtype=jnp.int32),
        )
        return arena, plan
    left = pair[0]
    right = pair[1]
    a, b = _pair_roles(left, right, swapped)
    ext_refs = _extract_external_refs(arena.state, a, b)
    rule_valid = rule_idx >= 0
    safe_rule_idx = jnp.where(rule_valid, rule_idx, jnp.int32(0))
    alloc_count = jnp.where(
        rule_valid, table.alloc_count[safe_rule_idx], jnp.int32(0)
    )
    arena, new_ids, alloc_ok = allocate_from_arena(arena, alloc_count)
    rhs_node_type = table.rhs_node_type[safe_rule_idx]
    rhs_port_map = table.rhs_port_map[safe_rule_idx]
    ext_port_map = table.ext_port_map[safe_rule_idx]
    active_mask = jnp.arange(MAX_NEW_NODES, dtype=jnp.int32) < alloc_count
    new_types = jnp.where(active_mask, rhs_node_type, IC_FREE)
    new_ports = _resolve_template_ports(rhs_port_map, new_ids, ext_refs)
    ext_targets, ext_values = _resolve_external_updates(ext_port_map, new_ids, ext_refs)
    plan = RewritePlan(
        alloc_ok=alloc_ok & rule_valid,
        alloc_count=alloc_count,
        new_ids=new_ids,
        new_types=new_types.astype(jnp.int8),
        new_ports=new_ports,
        ext_targets=ext_targets,
        ext_values=ext_values,
    )
    return arena, plan


def build_rewrite_plan_batch(
    arena: ICArena,
    pairs: jnp.ndarray,
    rule_idx: jnp.ndarray,
    swapped: jnp.ndarray,
    table: RuleTable,
) -> tuple[ICArena, RewritePlanBatch]:
    if table.lhs.shape[0] == 0:
        empty_ids = jnp.zeros((0, MAX_NEW_NODES), dtype=jnp.int32)
        empty_ports = jnp.zeros((0, MAX_NEW_NODES, PORT_ARITY), dtype=jnp.int32)
        empty_ext = jnp.zeros((0, EXT_COUNT, 2), dtype=jnp.int32)
        plan = RewritePlanBatch(
            alloc_ok=jnp.zeros((0,), dtype=jnp.bool_),
            alloc_count=jnp.zeros((0,), dtype=jnp.int32),
            new_ids=empty_ids,
            new_types=jnp.zeros((0, MAX_NEW_NODES), dtype=jnp.int8),
            new_ports=empty_ports,
            ext_targets=empty_ext,
            ext_values=jnp.zeros((0, EXT_COUNT), dtype=jnp.int32),
        )
        return arena, plan
    left = pairs[:, 0]
    right = pairs[:, 1]
    a, b = _pair_roles(left, right, swapped)
    ext_refs = _extract_external_refs_batch(arena.state, a, b)
    rule_valid = rule_idx >= 0
    safe_rule_idx = jnp.where(rule_valid, rule_idx, jnp.int32(0))
    alloc_count = jnp.where(
        rule_valid, table.alloc_count[safe_rule_idx], jnp.int32(0)
    )
    arena, new_ids, alloc_ok = allocate_batch_from_arena(arena, alloc_count)
    rhs_node_type = table.rhs_node_type[safe_rule_idx]
    rhs_port_map = table.rhs_port_map[safe_rule_idx]
    ext_port_map = table.ext_port_map[safe_rule_idx]
    active_mask = (
        jnp.arange(MAX_NEW_NODES, dtype=jnp.int32)[None, :] < alloc_count[:, None]
    )
    new_types = jnp.where(active_mask, rhs_node_type, IC_FREE)
    new_ports = jax.vmap(_resolve_template_ports)(rhs_port_map, new_ids, ext_refs)
    ext_targets, ext_values = jax.vmap(_resolve_external_updates)(
        ext_port_map, new_ids, ext_refs
    )
    plan = RewritePlanBatch(
        alloc_ok=alloc_ok & rule_valid,
        alloc_count=alloc_count,
        new_ids=new_ids,
        new_types=new_types.astype(jnp.int8),
        new_ports=new_ports,
        ext_targets=ext_targets,
        ext_values=ext_values,
    )
    return arena, plan
