from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class HostInt:
    v: int

    def __int__(self) -> int:
        return int(self.v)

    def __index__(self) -> int:
        return int(self.v)


@dataclass(frozen=True)
class HostBool:
    v: bool

    def __bool__(self) -> bool:
        return bool(self.v)


def _host_int(value) -> HostInt:
    if isinstance(value, HostInt):
        return value
    if isinstance(value, HostBool):
        raise TypeError("expected HostInt, got HostBool")
    return HostInt(int(jax.device_get(value)))


def _host_bool(value) -> HostBool:
    if isinstance(value, HostBool):
        return value
    if isinstance(value, HostInt):
        raise TypeError("expected HostBool, got HostInt")
    return HostBool(bool(jax.device_get(value)))


def _host_int_value(value) -> int:
    return int(_host_int(value))


def _host_bool_value(value) -> bool:
    return bool(_host_bool(value))


__all__ = [
    "HostInt",
    "HostBool",
    "_host_int",
    "_host_bool",
    "_host_int_value",
    "_host_bool_value",
]
