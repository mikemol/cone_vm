from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class PrismExportMissingError(AttributeError):
    name: str
    module: str = "prism_vm_core.exports"
    available: tuple[str, ...] | None = None

    def __str__(self) -> str:
        # Preserve historical message for tests/logs.
        return f"prism_vm export missing: {self.name}"


@dataclass
class PrismPolicyModeError(ValueError):
    mode: object
    allowed: tuple[str, ...] = ("static", "value")
    context: str | None = None

    def __str__(self) -> str:
        # Preserve the legacy message shape where possible.
        return f"unknown policy_mode={self.mode!r}"


@dataclass
class PrismPolicyBindingError(ValueError):
    message: str
    context: str | None = None
    policy_mode: str | None = None

    def __str__(self) -> str:
        return self.message


@dataclass
class PrismValidateModeError(ValueError):
    mode: object
    allowed: tuple[str, ...] = ("none", "strict", "hyper")
    context: str | None = None

    def __str__(self) -> str:
        return f"Unknown validate_mode={self.mode!r}"


@dataclass
class PrismBspModeError(ValueError):
    mode: object
    allowed: tuple[str, ...] = ("intrinsic", "cnf2", "auto")
    context: str | None = None

    def __str__(self) -> str:
        return f"Unknown bsp_mode={self.mode!r}"


@dataclass
class PrismCnf2ModeError(ValueError):
    mode: object
    allowed: tuple[str, ...] = ("off", "base", "slot1", "auto")
    context: str | None = None

    def __str__(self) -> str:
        return f"Unknown cnf2_mode={self.mode!r}"


@dataclass
class PrismCnf2ModeConflictError(ValueError):
    message: str
    context: str | None = None

    def __str__(self) -> str:
        return self.message


@dataclass
class PrismSafetyModeError(ValueError):
    mode: object
    allowed: tuple[str, ...] = ("corrupt", "clamp", "drop")
    context: str | None = None

    def __str__(self) -> str:
        return f"Unknown safety mode: {self.mode!r}"


def _available_tuple(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(values)


__all__ = [
    "PrismExportMissingError",
    "PrismPolicyModeError",
    "PrismPolicyBindingError",
    "PrismValidateModeError",
    "PrismBspModeError",
    "PrismCnf2ModeError",
    "PrismCnf2ModeConflictError",
    "PrismSafetyModeError",
    "_available_tuple",
]
