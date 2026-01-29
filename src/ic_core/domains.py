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


def _require_ptr_domain(ptr, label: str, expected_type):
    if not isinstance(ptr, expected_type):
        raise TypeError(f"{label} expected {expected_type.__name__}")
    return ptr


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
    return _require_ptr_domain(ptr, label, ICNodeId)


def _require_port_id(ptr: ICPortId, label: str) -> ICPortId:
    return _require_ptr_domain(ptr, label, ICPortId)


def _require_ic_ptr(ptr: ICPtr, label: str) -> ICPtr:
    return _require_ptr_domain(ptr, label, ICPtr)


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
]
