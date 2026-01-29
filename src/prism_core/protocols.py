from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SafeGatherFn(Protocol):
    def __call__(
        self, arr, idx, label: str, *, policy=None, return_ok: bool = False
    ):
        ...


@runtime_checkable
class SafeGatherOkFn(Protocol):
    def __call__(self, arr, idx, label: str, *, policy=None):
        ...


@runtime_checkable
class SafeIndexFn(Protocol):
    def __call__(self, idx, size, label: str, *, policy=None):
        ...


__all__ = [
    "SafeGatherFn",
    "SafeGatherOkFn",
    "SafeIndexFn",
]
