"""Type-surface re-exports for IC.

This mirrors prism_vm_core/types.py for pointer-domain and state types.
"""

from ic_core.domains import (
    ICNodeId,
    ICPortId,
    ICPtr,
    HostInt,
    HostBool,
    _node_id,
    _port_id,
    _ic_ptr,
    _require_node_id,
    _require_port_id,
    _require_ic_ptr,
    _require_ptr_domain,
    _host_int,
    _host_bool,
    _host_int_value,
    _host_bool_value,
)
from ic_core.graph import (
    ICState,
    TYPE_FREE,
    TYPE_ERA,
    TYPE_CON,
    TYPE_DUP,
    PORT_PRINCIPAL,
    PORT_AUX_LEFT,
    PORT_AUX_RIGHT,
)
from ic_core.engine import ICRewriteStats
from prism_core.compact import (
    CompactResult,
    CompactConfig,
    DEFAULT_COMPACT_CONFIG,
)
from prism_core.alloc import AllocConfig, DEFAULT_ALLOC_CONFIG

__all__ = [
    "ICNodeId",
    "ICPortId",
    "ICPtr",
    "HostInt",
    "HostBool",
    "_node_id",
    "_port_id",
    "_ic_ptr",
    "_require_node_id",
    "_require_port_id",
    "_require_ic_ptr",
    "_require_ptr_domain",
    "_host_int",
    "_host_bool",
    "_host_int_value",
    "_host_bool_value",
    "ICState",
    "ICRewriteStats",
    "CompactResult",
    "CompactConfig",
    "DEFAULT_COMPACT_CONFIG",
    "AllocConfig",
    "DEFAULT_ALLOC_CONFIG",
    "TYPE_FREE",
    "TYPE_ERA",
    "TYPE_CON",
    "TYPE_DUP",
    "PORT_PRINCIPAL",
    "PORT_AUX_LEFT",
    "PORT_AUX_RIGHT",
]
