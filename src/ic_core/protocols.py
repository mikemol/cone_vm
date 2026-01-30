from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import jax.numpy as jnp

from ic_core.graph import ICState
from prism_core.compact import CompactResult
from prism_core.protocols import SafeIndexFn


@runtime_checkable
class CompactPairsFn(Protocol):
    def __call__(
        self, state: ICState
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...


@runtime_checkable
class CompactPairsResultFn(Protocol):
    def __call__(self, state: ICState) -> Tuple[CompactResult, jnp.ndarray]:
        ...


@runtime_checkable
class DecodePortFn(Protocol):
    def __call__(self, ptr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...


@runtime_checkable
class AllocPlanFn(Protocol):
    def __call__(
        self, state: ICState, pairs: jnp.ndarray, count: jnp.ndarray
    ) -> Tuple[ICState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...


@runtime_checkable
class ApplyTemplatePlannedFn(Protocol):
    def __call__(
        self,
        state: ICState,
        node_a: jnp.ndarray,
        node_b: jnp.ndarray,
        template_id: jnp.ndarray,
        alloc_ids: jnp.ndarray,
    ) -> ICState:
        ...


@runtime_checkable
class HaltedFn(Protocol):
    def __call__(self, state: ICState) -> jnp.ndarray:
        ...


@runtime_checkable
class ScanCorruptFn(Protocol):
    def __call__(self, state: ICState) -> ICState:
        ...


@runtime_checkable
class RuleForTypesFn(Protocol):
    def __call__(
        self, type_a: jnp.ndarray, type_b: jnp.ndarray
    ) -> jnp.ndarray:
        ...


@runtime_checkable
class ApplyAnnFn(Protocol):
    def __call__(self, state: ICState, node_a, node_b) -> ICState:
        ...


@runtime_checkable
class ApplyEraseFn(Protocol):
    def __call__(self, state: ICState, node_a, node_b) -> ICState:
        ...


@runtime_checkable
class ApplyCommuteFn(Protocol):
    def __call__(self, state: ICState, node_a, node_b) -> ICState:
        ...


@runtime_checkable
class ApplyTemplateFn(Protocol):
    def __call__(
        self,
        state: ICState,
        node_a: jnp.ndarray,
        node_b: jnp.ndarray,
        template_id: jnp.ndarray,
    ) -> ICState:
        ...


__all__ = [
    "CompactPairsFn",
    "CompactPairsResultFn",
    "DecodePortFn",
    "AllocPlanFn",
    "ApplyTemplatePlannedFn",
    "HaltedFn",
    "ScanCorruptFn",
    "RuleForTypesFn",
    "ApplyAnnFn",
    "ApplyEraseFn",
    "ApplyCommuteFn",
    "ApplyTemplateFn",
    "SafeIndexFn",
]
