from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PrismExportMissingError(AttributeError):
    name: str
    module: str = "prism_vm_core.exports"
    available: tuple[str, ...] | None = None

    def __str__(self) -> str:
        # Preserve historical message for tests/logs.
        return f"prism_vm export missing: {self.name}"


@dataclass(frozen=True)
class PrismPolicyModeError(ValueError):
    mode: object
    allowed: tuple[str, ...] = ("static", "value")
    context: str | None = None

    def __str__(self) -> str:
        # Preserve the legacy message shape where possible.
        return f"unknown policy_mode={self.mode!r}"


@dataclass(frozen=True)
class PrismPolicyBindingError(ValueError):
    message: str
    context: str | None = None
    policy_mode: str | None = None

    def __str__(self) -> str:
        return self.message


def _available_tuple(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(values)


__all__ = [
    "PrismExportMissingError",
    "PrismPolicyModeError",
    "PrismPolicyBindingError",
    "_available_tuple",
]
