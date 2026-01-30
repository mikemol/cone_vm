from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from prism_core import jax_safe as _jax_safe
from prism_core.safety import SafetyPolicy
from prism_core.di import call_with_optional_kwargs
from prism_core.guards import (
    GuardConfig,
    resolve_safe_gather_fn,
    resolve_safe_gather_ok_fn,
)
from prism_ledger.intern import intern_nodes
from prism_vm_core.constants import _PREFIX_SCAN_CHUNK
from prism_vm_core.domains import (
    QMap,
    _committed_ids,
    _host_bool,
    _host_int_value,
    _host_raise_if_bad,
    _provisional_ids,
)
from prism_vm_core.guards import _guards_enabled
from prism_vm_core.structures import NodeBatch, Stratum
from prism_core.protocols import SafeGatherOkFn
from prism_vm_core.protocols import HostRaiseFn, InternFn, NodeBatchFn

safe_gather_1d = _jax_safe.safe_gather_1d
safe_gather_1d_ok = _jax_safe.safe_gather_1d_ok


def _node_batch(op, a1, a2):
    return NodeBatch(op=op, a1=a1, a2=a2)


@jax.jit
def validate_stratum_no_within_refs_jax(ledger, stratum):
    # Strict strata rule: new nodes may only reference ids < stratum.start (m2 gate).
    # See IMPLEMENTATION_PLAN.md (m2 strata discipline).
    start = stratum.start
    count = jnp.maximum(stratum.count, 0)
    ledger_count = ledger.count.astype(jnp.int32)
    chunk = jnp.int32(_PREFIX_SCAN_CHUNK)
    max_start = jnp.int32(ledger.arg1.shape[0] - _PREFIX_SCAN_CHUNK)
    max_start = jnp.maximum(max_start, jnp.int32(0))
    num_chunks = (ledger_count + chunk - 1) // chunk

    def _scan_chunk(i, ok):
        base = jnp.minimum(i * chunk, max_start)
        a1 = lax.dynamic_slice_in_dim(ledger.arg1, base, _PREFIX_SCAN_CHUNK)
        a2 = lax.dynamic_slice_in_dim(ledger.arg2, base, _PREFIX_SCAN_CHUNK)
        ids = base + jnp.arange(_PREFIX_SCAN_CHUNK, dtype=jnp.int32)
        live = ids < ledger_count
        mask = live & (ids >= start) & (ids < start + count)
        ok_a1 = jnp.all(jnp.where(mask, a1 < start, True))
        ok_a2 = jnp.all(jnp.where(mask, a2 < start, True))
        return ok & ok_a1 & ok_a2

    return lax.fori_loop(0, num_chunks, _scan_chunk, jnp.bool_(True))


def validate_stratum_no_within_refs(
    ledger,
    stratum,
    *,
    guards_enabled_fn=_guards_enabled,
    host_int_value_fn=_host_int_value,
    host_bool_fn=_host_bool,
):
    if guards_enabled_fn():
        start = max(0, host_int_value_fn(stratum.start))
        count = max(0, host_int_value_fn(stratum.count))
        if count == 0:
            return True
        ledger_count = host_int_value_fn(ledger.count)
        end = min(start + count, ledger_count)
        if end <= start:
            return True
        # SYNC: host reads only the stratum slice for validation (m2).
        a1 = jax.device_get(ledger.arg1[start:end])
        a2 = jax.device_get(ledger.arg2[start:end])
        ok_a1 = bool((a1 < start).all())
        ok_a2 = bool((a2 < start).all())
        return host_bool_fn(ok_a1 and ok_a2)
    # SYNC: host bool() reads device result for validation (m1).
    return host_bool_fn(validate_stratum_no_within_refs_jax(ledger, stratum))


@jax.jit
def validate_stratum_no_future_refs_jax(ledger, stratum):
    # Hyperstrata rule: new nodes may only reference ids < their own id.
    start = stratum.start
    count = jnp.maximum(stratum.count, 0)
    ledger_count = ledger.count.astype(jnp.int32)
    chunk = jnp.int32(_PREFIX_SCAN_CHUNK)
    max_start = jnp.int32(ledger.arg1.shape[0] - _PREFIX_SCAN_CHUNK)
    max_start = jnp.maximum(max_start, jnp.int32(0))
    num_chunks = (ledger_count + chunk - 1) // chunk

    def _scan_chunk(i, ok):
        base = jnp.minimum(i * chunk, max_start)
        a1 = lax.dynamic_slice_in_dim(ledger.arg1, base, _PREFIX_SCAN_CHUNK)
        a2 = lax.dynamic_slice_in_dim(ledger.arg2, base, _PREFIX_SCAN_CHUNK)
        ids = base + jnp.arange(_PREFIX_SCAN_CHUNK, dtype=jnp.int32)
        live = ids < ledger_count
        mask = live & (ids >= start) & (ids < start + count)
        ok_a1 = jnp.all(jnp.where(mask, a1 < ids, True))
        ok_a2 = jnp.all(jnp.where(mask, a2 < ids, True))
        return ok & ok_a1 & ok_a2

    return lax.fori_loop(0, num_chunks, _scan_chunk, jnp.bool_(True))


def validate_stratum_no_future_refs(
    ledger,
    stratum,
    *,
    guards_enabled_fn=_guards_enabled,
    host_int_value_fn=_host_int_value,
    host_bool_fn=_host_bool,
):
    if guards_enabled_fn():
        start = max(0, host_int_value_fn(stratum.start))
        count = max(0, host_int_value_fn(stratum.count))
        if count == 0:
            return True
        ledger_count = host_int_value_fn(ledger.count)
        end = min(start + count, ledger_count)
        if end <= start:
            return True
        ids = jnp.arange(start, end, dtype=jnp.int32)
        a1 = jax.device_get(ledger.arg1[start:end])
        a2 = jax.device_get(ledger.arg2[start:end])
        ok_a1 = bool((a1 < ids).all())
        ok_a2 = bool((a2 < ids).all())
        return host_bool_fn(ok_a1 and ok_a2)
    return host_bool_fn(validate_stratum_no_future_refs_jax(ledger, stratum))


def _identity_q(ids, *, committed_ids_fn=_committed_ids):
    return committed_ids_fn(ids.a)


@dataclass(frozen=True)
class QMapMeta:
    stratum: Stratum
    canon_len: jnp.ndarray
    safe_gather_policy: SafetyPolicy | None


def _q_map_ok(ids, meta: QMapMeta):
    ids_arr = ids.a
    start = jnp.asarray(meta.stratum.start, dtype=jnp.int32)
    count = jnp.asarray(meta.stratum.count, dtype=jnp.int32)
    in_range = (ids_arr >= start) & (ids_arr < start + count)
    idx = jnp.where(in_range, ids_arr - start, jnp.int32(0))
    ok_idx = idx < meta.canon_len
    ok = jnp.where(in_range, ok_idx, True)
    policy = meta.safe_gather_policy or SafetyPolicy()
    if policy.mode == "clamp":
        ok = jnp.ones_like(ok, dtype=jnp.bool_)
    return ok


def apply_q(
    q: QMap,
    ids,
    *,
    provisional_ids_fn=_provisional_ids,
    return_ok: bool = False,
):
    # Collapseʰ: homomorphic projection q.
    ids_in = provisional_ids_fn(ids)
    out = q(ids_in)
    if not return_ok:
        return out
    meta = getattr(q, "_prism_meta", None)
    if meta is None:
        ok = jnp.ones_like(out.a, dtype=jnp.bool_)
        return out, ok
    ok = _q_map_ok(ids_in, meta)
    return out, ok


def _apply_stratum_q(
    ids,
    stratum: Stratum,
    canon_ids,
    label: str,
    *,
    safe_gather_fn=safe_gather_1d,
    safe_gather_ok_fn: SafeGatherOkFn = safe_gather_1d_ok,
    guards_enabled_fn=_guards_enabled,
    provisional_ids_fn=_provisional_ids,
    return_ok: bool = False,
    oob_policy: SafetyPolicy | None = None,
):
    expected = jnp.maximum(jnp.asarray(stratum.count, dtype=jnp.int32), 0)
    actual = jnp.asarray(canon_ids.a.shape[0], dtype=jnp.int32)
    if guards_enabled_fn():
        mismatch = expected != actual

        def _raise(bad_val, exp_val, act_val):
            if bad_val:
                raise RuntimeError(
                    f"guard failed: {label} count={int(exp_val)} canon_ids={int(act_val)}"
                )

        jax.debug.callback(_raise, mismatch, expected, actual)
    if canon_ids.a.shape[0] == 0:
        return ids
    start = jnp.asarray(stratum.start, dtype=jnp.int32)
    count = expected
    in_range = (ids.a >= start) & (ids.a < start + count)
    idx = jnp.where(in_range, ids.a - start, jnp.int32(0))
    canon_len = jnp.asarray(canon_ids.a.shape[0], dtype=jnp.int32)
    values, ok_idx, corrupt = call_with_optional_kwargs(
        safe_gather_ok_fn,
        {"policy": oob_policy},
        canon_ids.a,
        idx,
        label,
    )
    ok = jnp.where(in_range, ok_idx, True)
    mapped = values
    out = provisional_ids_fn(jnp.where(in_range, mapped, ids.a))
    if not return_ok:
        return out
    if oob_policy is None:
        oob_policy = SafetyPolicy()
    corrupt = jnp.where(in_range, corrupt, False)
    return out, ok, corrupt


def commit_stratum(
    ledger,
    stratum: Stratum,
    prior_q: QMap | None = None,
    validate: bool = False,
    validate_mode: str = "strict",
    *,
    intern_fn: InternFn = intern_nodes,
    node_batch_fn: NodeBatchFn = _node_batch,
    identity_q_fn=_identity_q,
    apply_stratum_q_fn=_apply_stratum_q,
    validate_within_fn=validate_stratum_no_within_refs,
    validate_future_fn=validate_stratum_no_future_refs,
    guards_enabled_fn=_guards_enabled,
    host_int_value_fn=_host_int_value,
    host_raise_fn: HostRaiseFn = _host_raise_if_bad,
    provisional_ids_fn=_provisional_ids,
    committed_ids_fn=_committed_ids,
    safe_gather_fn=safe_gather_1d,
    safe_gather_ok_fn: SafeGatherOkFn = safe_gather_1d_ok,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    # Collapseʰ: homomorphic projection q at the stratum boundary.
    if validate:
        mode = (validate_mode or "strict").strip().lower()
        if mode == "strict":
            ok = validate_within_fn(ledger, stratum)
        elif mode == "hyper":
            ok = validate_future_fn(ledger, stratum)
        else:
            raise ValueError(f"Unknown validate_mode={validate_mode!r}")
        if not ok:
            if mode == "strict":
                raise ValueError("Stratum contains within-tier references")
            raise ValueError("Stratum contains future references")
    # BSP_t barrier + Collapse_h: project provisional ids via q-map.
    # See IMPLEMENTATION_PLAN.md (m2 q boundary).
    q_prev: QMap = prior_q or identity_q_fn
    if safe_gather_policy is None:
        safe_gather_policy = SafetyPolicy()
    safe_gather_fn = resolve_safe_gather_fn(
        safe_gather_fn=safe_gather_fn,
        policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    safe_gather_ok_fn = resolve_safe_gather_ok_fn(
        safe_gather_ok_fn=safe_gather_ok_fn,
        policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    # SYNC: host int() pulls device scalar for host-side control flow (m1).
    count = host_int_value_fn(jnp.maximum(stratum.count, 0))
    if count == 0:
        canon_ids = committed_ids_fn(jnp.zeros((0,), dtype=jnp.int32))
        return ledger, canon_ids, q_prev
    start = jnp.asarray(stratum.start, dtype=jnp.int32)
    ids = start + jnp.arange(count, dtype=jnp.int32)
    ops = ledger.opcode[ids]
    a1 = q_prev(provisional_ids_fn(ledger.arg1[ids])).a
    a2 = q_prev(provisional_ids_fn(ledger.arg2[ids])).a
    canon_ids_raw, ledger = intern_fn(ledger, node_batch_fn(ops, a1, a2))
    canon_ids = committed_ids_fn(canon_ids_raw)
    if (validate or guards_enabled_fn()) and canon_ids.a.shape[0] != count:
        raise ValueError("Stratum count mismatch in commit_stratum")

    def q_map(ids_in):
        mapped = apply_stratum_q_fn(
            ids_in,
            stratum,
            canon_ids,
            "commit_stratum.q",
            safe_gather_fn=safe_gather_fn,
            safe_gather_ok_fn=safe_gather_ok_fn,
            guards_enabled_fn=guards_enabled_fn,
            provisional_ids_fn=provisional_ids_fn,
            oob_policy=safe_gather_policy,
        )
        return q_prev(mapped)

    q_meta = QMapMeta(
        stratum=stratum,
        canon_len=jnp.asarray(canon_ids.a.shape[0], dtype=jnp.int32),
        safe_gather_policy=safe_gather_policy,
    )
    setattr(q_map, "_prism_meta", q_meta)

    host_raise_fn(ledger, "Ledger capacity exceeded during commit_stratum")
    return ledger, canon_ids, q_map


__all__ = [
    "apply_q",
    "commit_stratum",
    "validate_stratum_no_within_refs_jax",
    "validate_stratum_no_within_refs",
    "validate_stratum_no_future_refs_jax",
    "validate_stratum_no_future_refs",
]
