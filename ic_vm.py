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
