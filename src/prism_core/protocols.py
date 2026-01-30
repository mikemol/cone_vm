from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import jax.numpy as jnp

PolicyValue: TypeAlias = jnp.ndarray


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
class SafeGatherValueFn(Protocol):
    def __call__(self, arr, idx, label: str, *, policy_value: PolicyValue):
        ...


@runtime_checkable
class SafeGatherOkValueFn(Protocol):
    def __call__(self, arr, idx, label: str, *, policy_value: PolicyValue):
        ...


@runtime_checkable
class SafeGatherOkBoundFn(Protocol):
    def __call__(self, arr, idx, label: str):
        ...


@runtime_checkable
class SafeGatherOkDynamicFn(Protocol):
    def __call__(self, arr, idx, label: str, *, policy):
        ...


@runtime_checkable
class SafeIndexFn(Protocol):
    def __call__(self, idx, size, label: str, *, policy=None):
        ...


@runtime_checkable
class SafeIndexValueFn(Protocol):
    def __call__(self, idx, size, label: str, *, policy_value: PolicyValue):
        ...


__all__ = [
    "PolicyValue",
    "SafeGatherFn",
    "SafeGatherOkFn",
    "SafeGatherValueFn",
    "SafeGatherOkValueFn",
    "SafeGatherOkBoundFn",
    "SafeGatherOkDynamicFn",
    "SafeIndexFn",
    "SafeIndexValueFn",
]
