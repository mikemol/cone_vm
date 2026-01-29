from __future__ import annotations

from dataclasses import dataclass

from prism_core.host import (
    HostBool,
    HostInt,
    _host_bool,
    _host_bool_value,
    _host_int,
    _host_int_value,
)


@dataclass(frozen=True)
class ICNodeId:
    i: int

    def __int__(self) -> int:
        return int(self.i)

    def __index__(self) -> int:
        return int(self.i)


@dataclass(frozen=True)
class ICPortId:
    i: int

    def __int__(self) -> int:
        return int(self.i)

    def __index__(self) -> int:
        return int(self.i)


@dataclass(frozen=True)
class ICPtr:
    i: int

    def __int__(self) -> int:
        return int(self.i)

    def __index__(self) -> int:
        return int(self.i)


def _node_id(value) -> ICNodeId:
    if isinstance(value, ICNodeId):
        return value
    if isinstance(value, (ICPortId, ICPtr)):
        raise TypeError("expected ICNodeId, got different pointer domain")
    return ICNodeId(_host_int_value(value))


def _port_id(value) -> ICPortId:
    if isinstance(value, ICPortId):
        return value
    if isinstance(value, (ICNodeId, ICPtr)):
        raise TypeError("expected ICPortId, got different pointer domain")
    return ICPortId(_host_int_value(value))


def _ic_ptr(value) -> ICPtr:
    if isinstance(value, ICPtr):
        return value
    if isinstance(value, (ICNodeId, ICPortId)):
        raise TypeError("expected ICPtr, got different pointer domain")
    return ICPtr(_host_int_value(value))


def _require_node_id(ptr: ICNodeId, label: str) -> ICNodeId:
    if not isinstance(ptr, ICNodeId):
        raise TypeError(f"{label} expected ICNodeId")
    return ptr


def _require_port_id(ptr: ICPortId, label: str) -> ICPortId:
    if not isinstance(ptr, ICPortId):
        raise TypeError(f"{label} expected ICPortId")
    return ptr


def _require_ic_ptr(ptr: ICPtr, label: str) -> ICPtr:
    if not isinstance(ptr, ICPtr):
        raise TypeError(f"{label} expected ICPtr")
    return ptr


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
    "_host_int",
    "_host_bool",
    "_host_int_value",
    "_host_bool_value",
]
