from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import jax.numpy as jnp

from prism_core.compact import CompactResult
from prism_vm_core.structures import Arena, CandidateBuffer, Ledger, NodeBatch, Stratum


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
    def __call__(self, enabled) -> CompactResult:
        ...


@runtime_checkable
class ScatterDropFn(Protocol):
    def __call__(self, arr, idx, updates, label: str):
        ...


@runtime_checkable
class SafeGatherFn(Protocol):
    def __call__(
        self, arr, idx, label: str, *, policy=None, return_ok: bool = False
    ):
        ...


@runtime_checkable
class GuardMaxFn(Protocol):
    def __call__(self, value, max_value, label: str) -> None:
        ...


@runtime_checkable
class OpRankFn(Protocol):
    def __call__(self, arena: Arena):
        ...


@runtime_checkable
class ServoEnabledFn(Protocol):
    def __call__(self) -> bool:
        ...


@runtime_checkable
class ServoUpdateFn(Protocol):
    def __call__(self, arena: Arena) -> Arena:
        ...


@runtime_checkable
class OpMortonFn(Protocol):
    def __call__(self, arena: Arena):
        ...


@runtime_checkable
class OpSortWithPermFn(Protocol):
    def __call__(self, arena: Arena, *args, **kwargs) -> Tuple[Arena, jnp.ndarray]:
        ...


@runtime_checkable
class ArenaRootHashFn(Protocol):
    def __call__(self, arena: Arena, root_ptr):
        ...


@runtime_checkable
class DamageTileSizeFn(Protocol):
    def __call__(self, block_size, l2_block_size, l1_block_size):
        ...


@runtime_checkable
class DamageMetricsUpdateFn(Protocol):
    def __call__(self, arena: Arena, tile_size) -> None:
        ...


@runtime_checkable
class OpInteractFn(Protocol):
    def __call__(self, arena: Arena) -> Arena:
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
        safe_gather_policy=None,
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
    "SafeGatherFn",
    "GuardMaxFn",
    "OpRankFn",
    "ServoEnabledFn",
    "ServoUpdateFn",
    "OpMortonFn",
    "OpSortWithPermFn",
    "ArenaRootHashFn",
    "DamageTileSizeFn",
    "DamageMetricsUpdateFn",
    "OpInteractFn",
    "GuardsEnabledFn",
    "HostIntValueFn",
    "HostBoolValueFn",
    "LedgerRootsHashFn",
    "CommitStratumFn",
    "ApplyQFn",
    "IdentityQFn",
]
