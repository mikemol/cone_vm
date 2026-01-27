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
