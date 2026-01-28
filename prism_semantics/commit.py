import jax
import jax.numpy as jnp
from jax import lax

from prism_core import jax_safe as _jax_safe
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

safe_gather_1d = _jax_safe.safe_gather_1d


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


def validate_stratum_no_within_refs(ledger, stratum):
    if _guards_enabled():
        start = max(0, _host_int_value(stratum.start))
        count = max(0, _host_int_value(stratum.count))
        if count == 0:
            return True
        ledger_count = _host_int_value(ledger.count)
        end = min(start + count, ledger_count)
        if end <= start:
            return True
        # SYNC: host reads only the stratum slice for validation (m2).
        a1 = jax.device_get(ledger.arg1[start:end])
        a2 = jax.device_get(ledger.arg2[start:end])
        ok_a1 = bool((a1 < start).all())
        ok_a2 = bool((a2 < start).all())
        return _host_bool(ok_a1 and ok_a2)
    # SYNC: host bool() reads device result for validation (m1).
    return _host_bool(validate_stratum_no_within_refs_jax(ledger, stratum))


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


def validate_stratum_no_future_refs(ledger, stratum):
    if _guards_enabled():
        start = max(0, _host_int_value(stratum.start))
        count = max(0, _host_int_value(stratum.count))
        if count == 0:
            return True
        ledger_count = _host_int_value(ledger.count)
        end = min(start + count, ledger_count)
        if end <= start:
            return True
        ids = jnp.arange(start, end, dtype=jnp.int32)
        a1 = jax.device_get(ledger.arg1[start:end])
        a2 = jax.device_get(ledger.arg2[start:end])
        ok_a1 = bool((a1 < ids).all())
        ok_a2 = bool((a2 < ids).all())
        return _host_bool(ok_a1 and ok_a2)
    return _host_bool(validate_stratum_no_future_refs_jax(ledger, stratum))


def _identity_q(ids):
    return _committed_ids(ids.a)


def apply_q(q: QMap, ids):
    # Collapseʰ: homomorphic projection q.
    return q(_provisional_ids(ids))


def _apply_stratum_q(
    ids,
    stratum: Stratum,
    canon_ids,
    label: str,
):
    expected = jnp.maximum(jnp.asarray(stratum.count, dtype=jnp.int32), 0)
    actual = jnp.asarray(canon_ids.a.shape[0], dtype=jnp.int32)
    if _guards_enabled():
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
    mapped = safe_gather_1d(canon_ids.a, idx, label)
    return _provisional_ids(jnp.where(in_range, mapped, ids.a))


def commit_stratum(
    ledger,
    stratum: Stratum,
    prior_q: QMap | None = None,
    validate: bool = False,
    validate_mode: str = "strict",
    intern_fn=intern_nodes,
):
    # Collapseʰ: homomorphic projection q at the stratum boundary.
    if validate:
        mode = (validate_mode or "strict").strip().lower()
        if mode == "strict":
            ok = validate_stratum_no_within_refs(ledger, stratum)
        elif mode == "hyper":
            ok = validate_stratum_no_future_refs(ledger, stratum)
        else:
            raise ValueError(f"Unknown validate_mode={validate_mode!r}")
        if not ok:
            if mode == "strict":
                raise ValueError("Stratum contains within-tier references")
            raise ValueError("Stratum contains future references")
    # BSP_t barrier + Collapse_h: project provisional ids via q-map.
    # See IMPLEMENTATION_PLAN.md (m2 q boundary).
    q_prev: QMap = prior_q or _identity_q
    # SYNC: host int() pulls device scalar for host-side control flow (m1).
    count = _host_int_value(jnp.maximum(stratum.count, 0))
    if count == 0:
        canon_ids = _committed_ids(jnp.zeros((0,), dtype=jnp.int32))
        return ledger, canon_ids, q_prev
    start = jnp.asarray(stratum.start, dtype=jnp.int32)
    ids = start + jnp.arange(count, dtype=jnp.int32)
    ops = ledger.opcode[ids]
    a1 = q_prev(_provisional_ids(ledger.arg1[ids])).a
    a2 = q_prev(_provisional_ids(ledger.arg2[ids])).a
    canon_ids_raw, ledger = intern_fn(ledger, _node_batch(ops, a1, a2))
    canon_ids = _committed_ids(canon_ids_raw)
    if (validate or _guards_enabled()) and canon_ids.a.shape[0] != count:
        raise ValueError("Stratum count mismatch in commit_stratum")

    def q_map(ids_in):
        mapped = _apply_stratum_q(ids_in, stratum, canon_ids, "commit_stratum.q")
        return q_prev(mapped)

    _host_raise_if_bad(ledger, "Ledger capacity exceeded during commit_stratum")
    return ledger, canon_ids, q_map


__all__ = [
    "apply_q",
    "commit_stratum",
    "validate_stratum_no_within_refs_jax",
    "validate_stratum_no_within_refs",
    "validate_stratum_no_future_refs_jax",
    "validate_stratum_no_future_refs",
]
