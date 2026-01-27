"""
Interaction Combinator (IC) engine scaffolding.

This is a minimal data-model placeholder for the in-8 tensor/rule-table track.
It intentionally avoids operational semantics until the rule table and port
wiring pipeline are formalized.
"""

from __future__ import annotations

from dataclasses import dataclass
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
