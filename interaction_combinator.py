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


@dataclass(frozen=True)
class ICState:
    # node_type: int8 array [N]
    # port: int32 array [N,3] (encoded port refs; see encode_port)
    node_type: jnp.ndarray
    port: jnp.ndarray


@dataclass(frozen=True)
class RuleTable:
    # lhs: int8 array [R,2] (principal pair types)
    # rhs_node_type: int8 array [R,2] (scaffold: up to 2 nodes)
    # rhs_port_map: int32 array [R,2,3] (scaffold port wiring template)
    lhs: jnp.ndarray
    rhs_node_type: jnp.ndarray
    rhs_port_map: jnp.ndarray


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
    rhs_node_type = jnp.zeros((0, 2), dtype=jnp.int8)
    rhs_port_map = jnp.zeros((0, 2, PORT_ARITY), dtype=jnp.int32)
    return RuleTable(lhs=lhs, rhs_node_type=rhs_node_type, rhs_port_map=rhs_port_map)


def validate_rule_table(table: RuleTable) -> None:
    if table.lhs.ndim != 2 or table.lhs.shape[1] != 2:
        raise ValueError("RuleTable.lhs must be [R,2]")
    if table.rhs_node_type.ndim != 2 or table.rhs_node_type.shape[1] != 2:
        raise ValueError("RuleTable.rhs_node_type must be [R,2]")
    if table.rhs_port_map.ndim != 3 or table.rhs_port_map.shape[1:] != (
        2,
        PORT_ARITY,
    ):
        raise ValueError("RuleTable.rhs_port_map must be [R,2,3]")
    if table.lhs.shape[0] != table.rhs_node_type.shape[0]:
        raise ValueError("RuleTable lhs/rhs_node_type length mismatch")
    if table.lhs.shape[0] != table.rhs_port_map.shape[0]:
        raise ValueError("RuleTable lhs/rhs_port_map length mismatch")
