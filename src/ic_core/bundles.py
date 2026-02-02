from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

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


if TYPE_CHECKING:
    from ic_core.graph import ICState


@dataclass(frozen=True)
class TemplateApplyArgs:
    """Bundle of inputs for template application."""

    state: "ICState"
    node_a: jnp.ndarray
    node_b: jnp.ndarray
    template_id: jnp.ndarray


__all__ = [
    "TemplateApplyArgs",
    "WireEndpoints",
    "WirePtrPair",
    "WireStarEndpoints",
]
