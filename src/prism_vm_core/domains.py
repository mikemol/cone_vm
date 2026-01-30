from typing import Protocol

import jax.numpy as jnp

from prism_core.host import (
    HostBool,
    HostInt,
    _host_bool,
    _host_bool_value,
    _host_int,
    _host_int_value,
)
from prism_core.domains import _require_ptr_domain

from prism_vm_core.ontology import (
    ArenaPtr,
    CommittedIds,
    LedgerId,
    ManifestPtr,
    ProvisionalIds,
)


class QMap(Protocol):
    def __call__(self, ids: ProvisionalIds) -> CommittedIds: ...


def _manifest_ptr(value) -> ManifestPtr:
    if isinstance(value, ManifestPtr):
        return value
    if isinstance(value, (LedgerId, ArenaPtr, ProvisionalIds, CommittedIds)):
        raise TypeError("expected ManifestPtr, got different pointer domain")
    return ManifestPtr(_host_int_value(value))


def _ledger_id(value) -> LedgerId:
    if isinstance(value, LedgerId):
        return value
    if isinstance(value, (ManifestPtr, ArenaPtr, ProvisionalIds, CommittedIds)):
        raise TypeError("expected LedgerId, got different pointer domain")
    return LedgerId(_host_int_value(value))


def _arena_ptr(value) -> ArenaPtr:
    if isinstance(value, ArenaPtr):
        return value
    if isinstance(value, (ManifestPtr, LedgerId, ProvisionalIds, CommittedIds)):
        raise TypeError("expected ArenaPtr, got different pointer domain")
    return ArenaPtr(_host_int_value(value))


def _require_manifest_ptr(ptr: ManifestPtr, label: str) -> ManifestPtr:
    return _require_ptr_domain(ptr, label, ManifestPtr)


def _require_ledger_id(ptr: LedgerId, label: str) -> LedgerId:
    return _require_ptr_domain(ptr, label, LedgerId)


def _require_arena_ptr(ptr: ArenaPtr, label: str) -> ArenaPtr:
    return _require_ptr_domain(ptr, label, ArenaPtr)


def _host_raise_if_bad(
    ledger, oom_message: str = "Ledger capacity exceeded", oom_exc=RuntimeError
) -> None:
    # SYNC: host check after device-side mutations (m1).
    ledger.count.block_until_ready()
    if _host_bool_value(ledger.corrupt):
        raise RuntimeError(
            "CORRUPT: key encoding alias risk (id width exceeded)"
        )
    if _host_bool_value(ledger.oom):
        raise oom_exc(oom_message)


def _provisional_ids(value) -> ProvisionalIds:
    if isinstance(value, ProvisionalIds):
        return value
    if isinstance(value, CommittedIds):
        raise TypeError("expected ProvisionalIds, got CommittedIds")
    return ProvisionalIds(jnp.asarray(value))


def _committed_ids(value) -> CommittedIds:
    if isinstance(value, CommittedIds):
        return value
    if isinstance(value, ProvisionalIds):
        raise TypeError("expected CommittedIds, got ProvisionalIds")
    return CommittedIds(jnp.asarray(value))


__all__ = [
    "QMap",
    "_manifest_ptr",
    "_ledger_id",
    "_arena_ptr",
    "_require_manifest_ptr",
    "_require_ledger_id",
    "_require_arena_ptr",
    "_require_ptr_domain",
    "_host_int",
    "_host_bool",
    "_host_int_value",
    "_host_bool_value",
    "_host_raise_if_bad",
    "_provisional_ids",
    "_committed_ids",
]
