from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class WireEndpoints(NamedTuple):
    """Bundle of endpoints for node/port wiring."""

    node_a: jnp.ndarray
    port_a: jnp.ndarray
    node_b: jnp.ndarray
    port_b: jnp.ndarray


class WirePtrPair(NamedTuple):
    """Bundle of encoded pointer endpoints."""

    ptr_a: jnp.ndarray
    ptr_b: jnp.ndarray


class WireStarEndpoints(NamedTuple):
    """Bundle of endpoints for star wiring."""

    center_node: jnp.ndarray
    center_port: jnp.ndarray
    leaf_nodes: jnp.ndarray
    leaf_ports: jnp.ndarray


__all__ = [
    "WireEndpoints",
    "WirePtrPair",
    "WireStarEndpoints",
]
