"""
Interaction Combinator (IC) engine scaffolding.

This is a minimal data-model placeholder for the in-8 tensor/rule-table track.
It intentionally avoids operational semantics until the rule table and port
wiring pipeline are formalized.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp

IC_CON = 0
IC_DUP = 1
IC_ERA = 2

PORT_P = 0
PORT_L = 1
PORT_R = 2
PORT_ARITY = 3


@dataclass(frozen=True)
class ICState:
    # node_type: int8 array [N]
    # port: int32 array [N,3] (adjacency by port index)
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


def init_ic_state(capacity: int) -> ICState:
    node_type = jnp.full((capacity,), IC_ERA, dtype=jnp.int8)
    port = jnp.zeros((capacity, PORT_ARITY), dtype=jnp.int32)
    return ICState(node_type=node_type, port=port)


def validate_ic_state(state: ICState) -> None:
    if state.node_type.ndim != 1:
        raise ValueError("ICState.node_type must be 1D")
    if state.port.ndim != 2 or state.port.shape[1] != PORT_ARITY:
        raise ValueError("ICState.port must be [N,3]")
    if state.port.shape[0] != state.node_type.shape[0]:
        raise ValueError("ICState node_type/port length mismatch")


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
