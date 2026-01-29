from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import jax.numpy as jnp

from prism_vm_core.structures import CandidateBuffer, Ledger, NodeBatch, Stratum


@runtime_checkable
class InternFn(Protocol):
    def __call__(
        self, ledger: Ledger, batch_or_ops, a1=None, a2=None
    ) -> Tuple[jnp.ndarray, Ledger]:
        ...


@runtime_checkable
class EmitCandidatesFn(Protocol):
    def __call__(self, ledger: Ledger, rewrite_ids) -> CandidateBuffer:
        ...


@runtime_checkable
class HostRaiseFn(Protocol):
    def __call__(self, ledger: Ledger, message: str) -> None:
        ...


@runtime_checkable
class NodeBatchFn(Protocol):
    def __call__(self, ops, a1, a2) -> NodeBatch:
        ...


@runtime_checkable
class CoordXorBatchFn(Protocol):
    def __call__(
        self, ledger: Ledger, left_ids, right_ids
    ) -> Tuple[jnp.ndarray, Ledger]:
        ...


@runtime_checkable
class CandidateIndicesFn(Protocol):
    def __call__(
        self, enabled
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...


@runtime_checkable
class ScatterDropFn(Protocol):
    def __call__(self, arr, idx, updates, label: str):
        ...


@runtime_checkable
class GuardsEnabledFn(Protocol):
    def __call__(self) -> bool:
        ...


@runtime_checkable
class HostIntValueFn(Protocol):
    def __call__(self, value) -> int:
        ...


@runtime_checkable
class HostBoolValueFn(Protocol):
    def __call__(self, value) -> bool:
        ...


@runtime_checkable
class LedgerRootsHashFn(Protocol):
    def __call__(self, ledger: Ledger, root_ids):
        ...


@runtime_checkable
class CommitStratumFn(Protocol):
    def __call__(
        self,
        ledger: Ledger,
        stratum: Stratum,
        prior_q=None,
        validate: bool = False,
        validate_mode: str = "strict",
        *,
        intern_fn: InternFn | None = None,
    ):
        ...


@runtime_checkable
class ApplyQFn(Protocol):
    def __call__(self, q, ids):
        ...


@runtime_checkable
class IdentityQFn(Protocol):
    def __call__(self, ids):
        ...


__all__ = [
    "InternFn",
    "EmitCandidatesFn",
    "HostRaiseFn",
    "NodeBatchFn",
    "CoordXorBatchFn",
    "CandidateIndicesFn",
    "ScatterDropFn",
    "GuardsEnabledFn",
    "HostIntValueFn",
    "HostBoolValueFn",
    "LedgerRootsHashFn",
    "CommitStratumFn",
    "ApplyQFn",
    "IdentityQFn",
]
