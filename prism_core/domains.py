"""Shared domain types and sentinel conventions.

This module is intentionally free of runtime behavior. It exists to document
and type-tag pointer domains across engines without changing execution.
"""

from typing import NewType

# Host-only domain tags (type checkers only).
NodeId = NewType("NodeId", int)
PortId = NewType("PortId", int)
Ptr = NewType("Ptr", int)

LedgerId = NewType("LedgerId", int)
ArenaPtr = NewType("ArenaPtr", int)
ManifestPtr = NewType("ManifestPtr", int)

# Sentinel conventions (documentation only).
NULL_PTR = 0  # Encoded pointer sentinel.
RESERVED_NODE = 0  # Reserved node index for NULL pointer encoding.

__all__ = [
    "NodeId",
    "PortId",
    "Ptr",
    "LedgerId",
    "ArenaPtr",
    "ManifestPtr",
    "NULL_PTR",
    "RESERVED_NODE",
]
