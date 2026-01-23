from jax import jit, lax
import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Callable, Tuple
import inspect
import os
import re
import time

# --- 1. Ontology (Opcodes) ---
# id=0/1 are semantic reserves (NULL/ZERO) per univalence hard-cap contract.
OP_NULL = 0
OP_ZERO = 1
OP_SUC  = 2
OP_ADD  = 10
OP_MUL  = 11
OP_SORT = 99
OP_COORD_ZERO = 20
OP_COORD_ONE = 21
OP_COORD_PAIR = 22
ZERO_PTR = 1  # Must stay aligned with OP_ZERO (identity semantics).

OP_NAMES = {
    0: "NULL", 1: "zero", 2: "suc",
    10: "add", 11: "mul", 99: "sort",
    20: "coord_zero", 21: "coord_one", 22: "coord_pair"
}

MAX_ROWS = 1024 * 32
MAX_KEY_NODES = 1 << 16
MAX_NODES = MAX_KEY_NODES - 1
MAX_ID = MAX_NODES - 1
# Hard-cap is semantic (univalence), not just capacity; see IMPLEMENTATION_PLAN.md.
if MAX_NODES >= MAX_KEY_NODES:
    raise ValueError("MAX_NODES exceeds 16-bit key packing")
MAX_COORD_STEPS = 8
_TEST_GUARDS = os.environ.get("PRISM_TEST_GUARDS", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Test-time guards favor correctness over performance (m1 gate in plan).
_SCATTER_GUARD = _TEST_GUARDS or os.environ.get(
    "PRISM_SCATTER_GUARD", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_GATHER_GUARD = _TEST_GUARDS or os.environ.get(
    "PRISM_GATHER_GUARD", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_HAS_DEBUG_CALLBACK = hasattr(jax, "debug") and hasattr(jax.debug, "callback")


def _scatter_guard(indices, max_index, label):
    if not _SCATTER_GUARD or not _HAS_DEBUG_CALLBACK:
        return
    if indices.size == 0:
        return
    min_idx = jnp.min(indices)
    max_idx = jnp.max(indices)
    # Allow sentinel index == max_index for intentional drop semantics.
    bad = (min_idx < 0) | (max_idx > max_index)

    def _raise(bad_val, min_val, max_val, max_allowed):
        if bad_val:
            raise RuntimeError(
                f"scatter index out of bounds in {label} "
                f"(min={int(min_val)}, max={int(max_val)}, max={int(max_allowed)})"
            )

    jax.debug.callback(_raise, bad, min_idx, max_idx, max_index)


def _scatter_drop(target, indices, values, label):
    max_index = jnp.asarray(target.shape[0], dtype=jnp.int32)
    _scatter_guard(indices, max_index, label)
    return target.at[indices].set(values, mode="drop")


def _guard_gather_index(idx, size, label):
    if not _GATHER_GUARD or not _HAS_DEBUG_CALLBACK:
        return
    if idx.size == 0:
        return
    min_idx = jnp.min(idx)
    max_idx = jnp.max(idx)
    bad = (min_idx < 0) | (max_idx >= size)

    def _raise(bad_val, min_val, max_val, size_val):
        if bad_val:
            raise RuntimeError(
                "gather index out of bounds in "
                f"{label} (min={int(min_val)}, max={int(max_val)}, size={int(size_val)})"
            )

    jax.debug.callback(_raise, bad, min_idx, max_idx, size)


def safe_gather_1d(arr, idx, label="safe_gather_1d"):
    # Guarded gather: raise on invalid indices, clamp to avoid backend-specific OOB.
    if _GATHER_GUARD and _HAS_DEBUG_CALLBACK:
        size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
        idx_i = jnp.asarray(idx, dtype=jnp.int32)
        _guard_gather_index(idx_i, size, label)
        idx_safe = jnp.clip(idx_i, 0, size - 1)
        return arr[idx_safe]
    return arr[idx]


_BINCOUT_HAS_LENGTH = "length" in inspect.signature(jnp.bincount).parameters


def _bincount_256(x, weights):
    # Fixed-size bincount keeps JIT shapes static across JAX versions.
    if _BINCOUT_HAS_LENGTH:
        return jnp.bincount(x, weights=weights, minlength=256, length=256)
    out = jnp.zeros(256, dtype=weights.dtype)
    return out.at[x].add(weights)


def _parse_milestone_value(value):
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("m"):
        value = value[1:]
    if value.isdigit():
        return int(value)
    return None


def _read_pytest_milestone():
    if not _TEST_GUARDS:
        return None
    path = os.path.join(os.path.dirname(__file__), ".pytest-milestone")
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "PRISM_MILESTONE":
                        return _parse_milestone_value(value)
                else:
                    return _parse_milestone_value(line)
    except FileNotFoundError:
        return None
    return None


def _cnf2_enabled():
    # CNF-2 pipeline is staged for m2+; guard uses env/milestone in tests.
    value = os.environ.get("PRISM_ENABLE_CNF2", "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    milestone = _parse_milestone_value(os.environ.get("PRISM_MILESTONE", ""))
    if milestone is None:
        milestone = _read_pytest_milestone()
    return milestone is not None and milestone >= 2


def _default_bsp_mode():
    # CNF-2 becomes the default at m2; intrinsic remains the oracle path.
    return "cnf2" if _cnf2_enabled() else "intrinsic"


def _normalize_bsp_mode(bsp_mode):
    if bsp_mode in (None, "", "auto"):
        return _default_bsp_mode()
    return bsp_mode


def _guards_enabled():
    return _TEST_GUARDS and _HAS_DEBUG_CALLBACK


def _guard_max(value, max_value, label):
    if not _guards_enabled():
        return
    bad = value > max_value

    def _raise(bad_val, val, max_val):
        if bad_val:
            raise RuntimeError(
                f"guard failed: {label} {int(val)} > {int(max_val)}"
            )

    jax.debug.callback(_raise, bad, value, max_value)


def _pop_token(tokens):
    if not tokens:
        raise ValueError("Unexpected end of input")
    return tokens.pop(0)


def _expect_token(tokens, expected):
    token = _pop_token(tokens)
    if token != expected:
        raise ValueError(f"Expected {expected!r}, got {token!r}")
    return token


def _guard_slot0_perm(perm, inv_perm, label):
    if not _guards_enabled():
        return
    p0 = perm[0]
    i0 = inv_perm[0]
    ok = (p0 == 0) & (i0 == 0)

    def _raise(ok_val, p0_val, i0_val):
        if not ok_val:
            raise RuntimeError(
                f"guard failed: {label} perm[0]={int(p0_val)} inv_perm[0]={int(i0_val)}"
            )

    jax.debug.callback(_raise, ok, p0, i0)


def _guard_null_row(opcode, arg1, arg2, label):
    if not _guards_enabled():
        return
    op0 = opcode[0]
    a10 = arg1[0]
    a20 = arg2[0]
    ok = (op0 == OP_NULL) & (a10 == 0) & (a20 == 0)

    def _raise(ok_val, op0_val, a10_val, a20_val):
        if not ok_val:
            raise RuntimeError(
                f"guard failed: {label} op0={int(op0_val)} a1={int(a10_val)} a2={int(a20_val)}"
            )

    jax.debug.callback(_raise, ok, op0, a10, a20)


_coord_norm_probe_count = 0


def coord_norm_probe_reset():
    # Debug-only probe used by m4 tests to detect coord normalization scope.
    global _coord_norm_probe_count
    _coord_norm_probe_count = 0


def coord_norm_probe_get():
    return int(_coord_norm_probe_count)


def _coord_norm_probe_enabled():
    value = os.environ.get("PRISM_COORD_NORM_PROBE", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _coord_norm_probe_tick(n):
    # Only increments when PRISM_COORD_NORM_PROBE is enabled.
    if not _coord_norm_probe_enabled():
        return
    global _coord_norm_probe_count
    _coord_norm_probe_count += int(n)

# --- Rank (2-bit Scheduler) ---
RANK_HOT = 0
RANK_WARM = 1  # Reserved for future policies.
RANK_COLD = 2
RANK_FREE = 3

# --- 2. Manifest (Heap) ---
class Manifest(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    active_count: jnp.ndarray
    oom: jnp.ndarray

class Arena(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    rank:   jnp.ndarray
    count:  jnp.ndarray
    oom: jnp.ndarray

class Ledger(NamedTuple):
    opcode: jnp.ndarray
    arg1:   jnp.ndarray
    arg2:   jnp.ndarray
    keys_b0_sorted: jnp.ndarray
    keys_b1_sorted: jnp.ndarray
    keys_b2_sorted: jnp.ndarray
    keys_b3_sorted: jnp.ndarray
    keys_b4_sorted: jnp.ndarray
    ids_sorted: jnp.ndarray
    count:  jnp.ndarray
    oom: jnp.ndarray
    corrupt: jnp.ndarray

class CandidateBuffer(NamedTuple):
    enabled: jnp.ndarray
    opcode: jnp.ndarray
    arg1: jnp.ndarray
    arg2: jnp.ndarray

class Stratum(NamedTuple):
    start: jnp.ndarray
    count: jnp.ndarray

def init_manifest():
    return Manifest(
        opcode=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_ROWS, dtype=jnp.int32),
        active_count=jnp.array(1, dtype=jnp.int32),
        oom=jnp.array(False, dtype=jnp.bool_),
    )

def init_arena():
    arena = Arena(
        opcode=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        arg1=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        arg2=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        rank=jnp.full(MAX_NODES, RANK_FREE, dtype=jnp.int8),
        count=jnp.array(1, dtype=jnp.int32),
        oom=jnp.array(False, dtype=jnp.bool_),
    )
    arena = arena._replace(
        opcode=arena.opcode.at[1].set(OP_ZERO),
        arg1=arena.arg1.at[1].set(0),
        arg2=arena.arg2.at[1].set(0),
        count=jnp.array(2, dtype=jnp.int32),
    )
    return arena

def init_ledger():
    max_key = jnp.uint8(0xFF)

    opcode = jnp.zeros(MAX_NODES, dtype=jnp.int32)
    arg1 = jnp.zeros(MAX_NODES, dtype=jnp.int32)
    arg2 = jnp.zeros(MAX_NODES, dtype=jnp.int32)

    opcode = opcode.at[1].set(OP_ZERO)

    keys_b0_sorted = jnp.full(MAX_NODES, max_key, dtype=jnp.uint8)
    keys_b1_sorted = jnp.full(MAX_NODES, max_key, dtype=jnp.uint8)
    keys_b2_sorted = jnp.full(MAX_NODES, max_key, dtype=jnp.uint8)
    keys_b3_sorted = jnp.full(MAX_NODES, max_key, dtype=jnp.uint8)
    keys_b4_sorted = jnp.full(MAX_NODES, max_key, dtype=jnp.uint8)
    ids_sorted = jnp.zeros(MAX_NODES, dtype=jnp.int32)

    k0_b0, k0_b1, k0_b2, k0_b3, k0_b4 = _pack_key(
        jnp.uint8(OP_NULL), jnp.uint16(0), jnp.uint16(0)
    )
    k1_b0, k1_b1, k1_b2, k1_b3, k1_b4 = _pack_key(
        jnp.uint8(OP_ZERO), jnp.uint16(0), jnp.uint16(0)
    )
    keys_b0_sorted = keys_b0_sorted.at[0].set(k0_b0).at[1].set(k1_b0)
    keys_b1_sorted = keys_b1_sorted.at[0].set(k0_b1).at[1].set(k1_b1)
    keys_b2_sorted = keys_b2_sorted.at[0].set(k0_b2).at[1].set(k1_b2)
    keys_b3_sorted = keys_b3_sorted.at[0].set(k0_b3).at[1].set(k1_b3)
    keys_b4_sorted = keys_b4_sorted.at[0].set(k0_b4).at[1].set(k1_b4)
    ids_sorted = ids_sorted.at[0].set(0).at[1].set(1)

    return Ledger(
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
        keys_b0_sorted=keys_b0_sorted,
        keys_b1_sorted=keys_b1_sorted,
        keys_b2_sorted=keys_b2_sorted,
        keys_b3_sorted=keys_b3_sorted,
        keys_b4_sorted=keys_b4_sorted,
        ids_sorted=ids_sorted,
        count=jnp.array(2, dtype=jnp.int32),
        oom=jnp.array(False, dtype=jnp.bool_),
        corrupt=jnp.array(False, dtype=jnp.bool_),
    )

def emit_candidates(ledger, frontier_ids):
    num_frontier = frontier_ids.shape[0]
    size = num_frontier * 2
    enabled = jnp.zeros(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = jnp.zeros(size, dtype=jnp.int32)
    arg2 = jnp.zeros(size, dtype=jnp.int32)

    if num_frontier == 0:
        return CandidateBuffer(enabled=enabled, opcode=opcode, arg1=arg1, arg2=arg2)

    f_ops = ledger.opcode[frontier_ids]
    f_a1 = ledger.arg1[frontier_ids]
    f_a2 = ledger.arg2[frontier_ids]
    op_a1 = ledger.opcode[f_a1]
    op_a2 = ledger.opcode[f_a2]

    is_add = f_ops == OP_ADD
    is_mul = f_ops == OP_MUL
    is_zero_a1 = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    is_suc_a1 = op_a1 == OP_SUC
    is_suc_a2 = op_a2 == OP_SUC

    is_add_zero = is_add & (is_zero_a1 | is_zero_a2)
    is_mul_zero = is_mul & (is_zero_a1 | is_zero_a2)
    is_add_suc = is_add & (is_suc_a1 | is_suc_a2) & (~is_add_zero)
    is_mul_suc = is_mul & (is_suc_a1 | is_suc_a2) & (~is_mul_zero)
    enable0 = is_add_zero | is_mul_zero | is_add_suc | is_mul_suc
    zero_on_a1 = is_zero_a1
    zero_on_a2 = (~is_zero_a1) & is_zero_a2
    zero_other = jnp.where(zero_on_a1, f_a2, f_a1)
    y_id = jnp.where(is_add_zero, zero_other, jnp.int32(ZERO_PTR))

    suc_on_a1 = is_suc_a1
    suc_on_a2 = (~is_suc_a1) & is_suc_a2
    suc_node = jnp.where(suc_on_a1, f_a1, f_a2)
    other_node = jnp.where(suc_on_a1, f_a2, f_a1)
    val_x = ledger.arg1[suc_node]
    val_y = other_node

    cand0_op = jnp.zeros_like(f_ops)
    cand0_a1 = jnp.zeros_like(f_a1)
    cand0_a2 = jnp.zeros_like(f_a2)

    id_mask = is_add_zero | is_mul_zero
    cand0_op = jnp.where(id_mask, ledger.opcode[y_id], cand0_op)
    cand0_a1 = jnp.where(id_mask, ledger.arg1[y_id], cand0_a1)
    cand0_a2 = jnp.where(id_mask, ledger.arg2[y_id], cand0_a2)

    cand0_op = jnp.where(is_add_suc, jnp.int32(OP_ADD), cand0_op)
    cand0_a1 = jnp.where(is_add_suc, val_x, cand0_a1)
    cand0_a2 = jnp.where(is_add_suc, val_y, cand0_a2)

    cand0_op = jnp.where(is_mul_suc, jnp.int32(OP_MUL), cand0_op)
    cand0_a1 = jnp.where(is_mul_suc, val_x, cand0_a1)
    cand0_a2 = jnp.where(is_mul_suc, val_y, cand0_a2)

    cand0_op = jnp.where(enable0, cand0_op, jnp.int32(0))
    cand0_a1 = jnp.where(enable0, cand0_a1, jnp.int32(0))
    cand0_a2 = jnp.where(enable0, cand0_a2, jnp.int32(0))

    idx0 = jnp.arange(num_frontier, dtype=jnp.int32) * 2
    enabled = enabled.at[idx0].set(enable0.astype(jnp.int32))
    opcode = opcode.at[idx0].set(cand0_op)
    arg1 = arg1.at[idx0].set(cand0_a1)
    arg2 = arg2.at[idx0].set(cand0_a2)

    return CandidateBuffer(enabled=enabled, opcode=opcode, arg1=arg1, arg2=arg2)

def _candidate_perm(enabled):
    enabled = enabled.astype(jnp.int32)
    size = enabled.shape[0]
    idx = jnp.arange(size, dtype=jnp.int32)
    sort_key = (1 - enabled) * (size + 1) + idx
    return jnp.argsort(sort_key).astype(jnp.int32)

def _candidate_indices(enabled):
    size = enabled.shape[0]
    count = jnp.sum(enabled).astype(jnp.int32)
    idx = jnp.nonzero(enabled, size=size, fill_value=0)[0].astype(jnp.int32)
    valid = jnp.arange(size, dtype=jnp.int32) < count
    return idx, valid, count

def compact_candidates(candidates):
    enabled = candidates.enabled.astype(jnp.int32)
    idx, valid, count = _candidate_indices(enabled)
    safe_idx = jnp.where(valid, idx, 0)

    compacted = CandidateBuffer(
        enabled=valid.astype(jnp.int32),
        opcode=candidates.opcode[safe_idx],
        arg1=candidates.arg1[safe_idx],
        arg2=candidates.arg2[safe_idx],
    )
    return compacted, count

def compact_candidates_with_index(candidates):
    enabled = candidates.enabled.astype(jnp.int32)
    idx, valid, count = _candidate_indices(enabled)
    safe_idx = jnp.where(valid, idx, 0)
    compacted = CandidateBuffer(
        enabled=valid.astype(jnp.int32),
        opcode=candidates.opcode[safe_idx],
        arg1=candidates.arg1[safe_idx],
        arg2=candidates.arg2[safe_idx],
    )
    return compacted, count, idx

def _scatter_compacted_ids(comp_idx, ids_compact, count, size):
    valid = jnp.arange(size, dtype=jnp.int32) < count
    scatter_idx = jnp.where(valid, comp_idx, jnp.int32(size))
    scatter_ids = jnp.where(valid, ids_compact, jnp.int32(0))
    ids_full = jnp.zeros(size, dtype=ids_compact.dtype)
    return _scatter_drop(
        ids_full, scatter_idx, scatter_ids, "scatter_compacted_ids"
    )

def intern_candidates(ledger, candidates):
    compacted, count = compact_candidates(candidates)
    enabled = compacted.enabled.astype(jnp.int32)
    ops = jnp.where(enabled, compacted.opcode, jnp.int32(0))
    a1 = jnp.where(enabled, compacted.arg1, jnp.int32(0))
    a2 = jnp.where(enabled, compacted.arg2, jnp.int32(0))
    ids, new_ledger = intern_nodes(ledger, ops, a1, a2)
    return ids, new_ledger, count


@jit
def validate_stratum_no_within_refs_jax(ledger, stratum):
    # Strict strata rule: new nodes may only reference ids < stratum.start (m2 gate).
    start = stratum.start
    count = jnp.maximum(stratum.count, 0)
    ids = jnp.arange(ledger.arg1.shape[0], dtype=jnp.int32)
    # Mask to live region only; static shape keeps JIT happy while ignoring junk.
    live = ids < ledger.count.astype(jnp.int32)
    mask = live & (ids >= start) & (ids < start + count)
    a1 = ledger.arg1[ids]
    a2 = ledger.arg2[ids]
    ok_a1 = jnp.all(jnp.where(mask, a1 < start, True))
    ok_a2 = jnp.all(jnp.where(mask, a2 < start, True))
    return ok_a1 & ok_a2

def validate_stratum_no_within_refs(ledger, stratum):
    return bool(validate_stratum_no_within_refs_jax(ledger, stratum))

def cycle_candidates(ledger, frontier_ids, validate_stratum=False):
    if not _cnf2_enabled():
        # CNF-2 candidate pipeline is staged for m2+ (plan); guard at entry.
        raise RuntimeError("cycle_candidates disabled until m2 (set PRISM_ENABLE_CNF2=1)")
    num_frontier = frontier_ids.shape[0]
    if num_frontier == 0:
        empty = Stratum(start=ledger.count.astype(jnp.int32), count=jnp.int32(0))
        return ledger, frontier_ids, (empty, empty, empty)

    f_ops = ledger.opcode[frontier_ids]
    f_a1 = ledger.arg1[frontier_ids]
    child_ops = ledger.opcode[f_a1]
    rewrite_child = (f_ops == OP_SUC) & (
        (child_ops == OP_ADD) | (child_ops == OP_MUL)
    )
    rewrite_ids = jnp.where(rewrite_child, f_a1, frontier_ids)

    candidates = emit_candidates(ledger, rewrite_ids)
    start0 = ledger.count.astype(jnp.int32)
    compacted0, count0, comp_idx0 = compact_candidates_with_index(candidates)
    enabled0 = compacted0.enabled.astype(jnp.int32)
    ops0 = jnp.where(enabled0, compacted0.opcode, jnp.int32(0))
    a1_0 = jnp.where(enabled0, compacted0.arg1, jnp.int32(0))
    a2_0 = jnp.where(enabled0, compacted0.arg2, jnp.int32(0))
    ids_compact, ledger0 = intern_nodes(ledger, ops0, a1_0, a2_0)
    size0 = candidates.enabled.shape[0]
    ids_full0 = _scatter_compacted_ids(comp_idx0, ids_compact, count0, size0)
    idx0 = jnp.arange(num_frontier, dtype=jnp.int32) * 2
    slot0_ids = ids_full0[idx0]

    r_ops = ledger.opcode[rewrite_ids]
    r_a1 = ledger.arg1[rewrite_ids]
    r_a2 = ledger.arg2[rewrite_ids]
    op_a1 = ledger.opcode[r_a1]
    op_a2 = ledger.opcode[r_a2]
    is_add = r_ops == OP_ADD
    is_mul = r_ops == OP_MUL
    is_suc_a1 = op_a1 == OP_SUC
    is_suc_a2 = op_a2 == OP_SUC
    is_add_suc = is_add & (is_suc_a1 | is_suc_a2)
    is_mul_suc = is_mul & (is_suc_a1 | is_suc_a2)
    is_zero_a1 = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    is_add_zero = is_add & (is_zero_a1 | is_zero_a2)
    is_mul_zero = is_mul & (is_zero_a1 | is_zero_a2)
    is_add_suc = is_add_suc & (~is_add_zero)
    is_mul_suc = is_mul_suc & (~is_mul_zero)
    suc_on_a1 = is_suc_a1
    suc_on_a2 = (~is_suc_a1) & is_suc_a2
    suc_node = jnp.where(suc_on_a1, r_a1, r_a2)
    val_x = ledger.arg1[suc_node]
    val_y = jnp.where(suc_on_a1, r_a2, r_a1)

    # Slot1 is the continuation slot in CNF-2; enabled only in m2+ staging.
    slot1_enabled = is_add_suc | is_mul_suc
    slot1_ops = jnp.zeros_like(r_ops)
    slot1_a1 = jnp.zeros_like(r_a1)
    slot1_a2 = jnp.zeros_like(r_a2)
    slot1_ops = jnp.where(is_add_suc, jnp.int32(OP_SUC), slot1_ops)
    slot1_a1 = jnp.where(is_add_suc, slot0_ids, slot1_a1)
    slot1_ops = jnp.where(is_mul_suc, jnp.int32(OP_ADD), slot1_ops)
    slot1_a1 = jnp.where(is_mul_suc, val_y, slot1_a1)
    slot1_a2 = jnp.where(is_mul_suc, slot0_ids, slot1_a2)

    slot1_ops = jnp.where(slot1_enabled, slot1_ops, jnp.int32(0))
    slot1_a1 = jnp.where(slot1_enabled, slot1_a1, jnp.int32(0))
    slot1_a2 = jnp.where(slot1_enabled, slot1_a2, jnp.int32(0))
    slot1_ids, ledger1 = intern_nodes(ledger0, slot1_ops, slot1_a1, slot1_a2)
    zero_on_a1 = is_zero_a1
    zero_on_a2 = (~is_zero_a1) & is_zero_a2
    zero_other = jnp.where(zero_on_a1, r_a2, r_a1)
    base_next = rewrite_ids
    base_next = jnp.where(is_add_zero, zero_other, base_next)
    base_next = jnp.where(is_mul_zero, jnp.int32(ZERO_PTR), base_next)
    base_next = jnp.where(is_add_suc, slot1_ids, base_next)
    base_next = jnp.where(is_mul_suc, slot1_ids, base_next)

    wrap_emit = rewrite_child & (base_next != rewrite_ids)
    wrap_ops = jnp.where(wrap_emit, jnp.int32(OP_SUC), jnp.int32(0))
    wrap_a1 = jnp.where(wrap_emit, base_next, jnp.int32(0))
    wrap_a2 = jnp.zeros_like(wrap_a1)
    wrap_ids, ledger2 = intern_nodes(ledger1, wrap_ops, wrap_a1, wrap_a2)

    next_frontier = jnp.where(rewrite_child, frontier_ids, base_next)
    next_frontier = jnp.where(wrap_emit, wrap_ids, next_frontier)

    stratum0 = Stratum(
        start=start0, count=(ledger0.count - start0).astype(jnp.int32)
    )
    start1 = ledger0.count.astype(jnp.int32)
    stratum1 = Stratum(
        start=start1, count=(ledger1.count - start1).astype(jnp.int32)
    )
    start2 = ledger1.count.astype(jnp.int32)
    stratum2 = Stratum(
        start=start2, count=(ledger2.count - start2).astype(jnp.int32)
    )
    if validate_stratum:
        if not validate_stratum_no_within_refs(ledger0, stratum0):
            raise ValueError("Stratum contains within-tier references")
        if not validate_stratum_no_within_refs(ledger1, stratum1):
            raise ValueError("Stratum contains within-tier references")
        if not validate_stratum_no_within_refs(ledger2, stratum2):
            raise ValueError("Stratum contains within-tier references")
    return ledger2, next_frontier, (stratum0, stratum1, stratum2)

@jit
def op_rank(arena):
    ops = arena.opcode
    is_free = ops == OP_NULL
    is_inst = ops >= 10
    new_rank = jnp.where(is_free, RANK_FREE, jnp.where(is_inst, RANK_HOT, RANK_COLD))
    return arena._replace(rank=new_rank.astype(jnp.int8))

def _invert_perm(perm):
    inv = jnp.empty_like(perm)
    return inv.at[perm].set(jnp.arange(perm.shape[0], dtype=perm.dtype))

def _canonicalize_commutative_host(op, a1, a2):
    if op in (OP_ADD, OP_MUL) and a2 < a1:
        return a2, a1
    return a1, a2

def _canonicalize_nodes(ops, a1, a2):
    is_coord_leaf = (ops == OP_COORD_ZERO) | (ops == OP_COORD_ONE)
    a1 = jnp.where(is_coord_leaf, jnp.int32(0), a1)
    a2 = jnp.where(is_coord_leaf, jnp.int32(0), a2)
    swap = (ops == OP_MUL) | (ops == OP_ADD)
    swap = swap & (a2 < a1)
    a1_swapped = jnp.where(swap, a2, a1)
    a2_swapped = jnp.where(swap, a1, a2)
    return ops, a1_swapped, a2_swapped


def _coord_norm_id_jax(ledger, coord_id):
    # Debug-only probe to detect vmap scope; see tests/test_coord_norm_probe.py.
    if _guards_enabled():
        jax.debug.callback(_coord_norm_probe_tick, jnp.int32(1))
    leaf_zero_id, leaf_zero_found = _lookup_node_id(
        ledger,
        jnp.int32(OP_COORD_ZERO),
        jnp.int32(0),
        jnp.int32(0),
    )
    leaf_one_id, leaf_one_found = _lookup_node_id(
        ledger,
        jnp.int32(OP_COORD_ONE),
        jnp.int32(0),
        jnp.int32(0),
    )

    def body(_, cid):
        op = ledger.opcode[cid]
        is_zero = op == OP_COORD_ZERO
        is_one = op == OP_COORD_ONE
        is_pair = op == OP_COORD_PAIR
        cid = jnp.where(is_zero & leaf_zero_found, leaf_zero_id, cid)
        cid = jnp.where(is_one & leaf_one_found, leaf_one_id, cid)

        left = ledger.arg1[cid]
        right = ledger.arg2[cid]
        left_op = ledger.opcode[left]
        right_op = ledger.opcode[right]
        left = jnp.where(
            (left_op == OP_COORD_ZERO) & leaf_zero_found, leaf_zero_id, left
        )
        left = jnp.where(
            (left_op == OP_COORD_ONE) & leaf_one_found, leaf_one_id, left
        )
        right = jnp.where(
            (right_op == OP_COORD_ZERO) & leaf_zero_found, leaf_zero_id, right
        )
        right = jnp.where(
            (right_op == OP_COORD_ONE) & leaf_one_found, leaf_one_id, right
        )

        pair_id, pair_found = _lookup_node_id(
            ledger, jnp.int32(OP_COORD_PAIR), left, right
        )
        cid = jnp.where(is_pair & pair_found, pair_id, cid)
        return cid

    return lax.fori_loop(0, MAX_COORD_STEPS, body, coord_id)


def _coord_leaf_id(ledger, op):
    ids, ledger = intern_nodes(
        ledger,
        jnp.array([op], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    return int(ids[0]), ledger


def _coord_promote_leaf(ledger, leaf_id):
    zero_id, ledger = _coord_leaf_id(ledger, OP_COORD_ZERO)
    ids, ledger = intern_nodes(
        ledger,
        jnp.array([OP_COORD_PAIR], dtype=jnp.int32),
        jnp.array([leaf_id], dtype=jnp.int32),
        jnp.array([zero_id], dtype=jnp.int32),
    )
    return int(ids[0]), ledger


def coord_norm(ledger, coord_id):
    # Host-only reference path; device hot paths should use jitted batch ops.
    coord_id = int(coord_id)
    op = int(ledger.opcode[coord_id])
    if op in (OP_COORD_ZERO, OP_COORD_ONE):
        return coord_id, ledger
    if op != OP_COORD_PAIR:
        return coord_id, ledger
    left = int(ledger.arg1[coord_id])
    right = int(ledger.arg2[coord_id])
    left_norm, ledger = coord_norm(ledger, left)
    right_norm, ledger = coord_norm(ledger, right)
    ids, ledger = intern_nodes(
        ledger,
        jnp.array([OP_COORD_PAIR], dtype=jnp.int32),
        jnp.array([left_norm], dtype=jnp.int32),
        jnp.array([right_norm], dtype=jnp.int32),
    )
    return int(ids[0]), ledger


def coord_xor(ledger, left_id, right_id):
    # Host-only reference path; device hot paths should use jitted batch ops.
    left_id = int(left_id)
    right_id = int(right_id)
    if left_id == right_id:
        return _coord_leaf_id(ledger, OP_COORD_ZERO)

    left_op = int(ledger.opcode[left_id])
    right_op = int(ledger.opcode[right_id])

    if left_op == OP_COORD_ZERO:
        return right_id, ledger
    if right_op == OP_COORD_ZERO:
        return left_id, ledger

    if left_op in (OP_COORD_ZERO, OP_COORD_ONE) and right_op in (
        OP_COORD_ZERO,
        OP_COORD_ONE,
    ):
        if left_op == right_op:
            return _coord_leaf_id(ledger, OP_COORD_ZERO)
        return _coord_leaf_id(ledger, OP_COORD_ONE)

    if left_op != OP_COORD_PAIR:
        left_id, ledger = _coord_promote_leaf(ledger, left_id)
    if right_op != OP_COORD_PAIR:
        right_id, ledger = _coord_promote_leaf(ledger, right_id)

    left_a1 = int(ledger.arg1[left_id])
    left_a2 = int(ledger.arg2[left_id])
    right_a1 = int(ledger.arg1[right_id])
    right_a2 = int(ledger.arg2[right_id])

    new_left, ledger = coord_xor(ledger, left_a1, right_a1)
    new_right, ledger = coord_xor(ledger, left_a2, right_a2)
    ids, ledger = intern_nodes(
        ledger,
        jnp.array([OP_COORD_PAIR], dtype=jnp.int32),
        jnp.array([new_left], dtype=jnp.int32),
        jnp.array([new_right], dtype=jnp.int32),
    )
    return int(ids[0]), ledger

def _pack_key(op, a1, a2):
    # Byte layout: op, a1_hi, a1_lo, a2_hi, a2_lo for lexicographic sort.
    op_u = op.astype(jnp.uint8)
    a1_u = (a1.astype(jnp.uint32) & jnp.uint32(0xFFFF)).astype(jnp.uint16)
    a2_u = (a2.astype(jnp.uint32) & jnp.uint32(0xFFFF)).astype(jnp.uint16)
    a1_hi = (a1_u >> jnp.uint16(8)).astype(jnp.uint8)
    a1_lo = (a1_u & jnp.uint16(0xFF)).astype(jnp.uint8)
    a2_hi = (a2_u >> jnp.uint16(8)).astype(jnp.uint8)
    a2_lo = (a2_u & jnp.uint16(0xFF)).astype(jnp.uint8)
    return op_u, a1_hi, a1_lo, a2_hi, a2_lo


def _lookup_node_id(ledger, op, a1, a2):
    k0, k1, k2, k3, k4 = _pack_key(op, a1, a2)
    L_b0 = ledger.keys_b0_sorted
    L_b1 = ledger.keys_b1_sorted
    L_b2 = ledger.keys_b2_sorted
    L_b3 = ledger.keys_b3_sorted
    L_b4 = ledger.keys_b4_sorted
    L_ids = ledger.ids_sorted
    count = ledger.count.astype(jnp.int32)

    def _lex_less(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4):
        return jnp.logical_or(
            a0 < b0,
            jnp.logical_and(
                a0 == b0,
                jnp.logical_or(
                    a1 < b1,
                    jnp.logical_and(
                        a1 == b1,
                        jnp.logical_or(
                            a2 < b2,
                            jnp.logical_and(
                                a2 == b2,
                                jnp.logical_or(
                                    a3 < b3, jnp.logical_and(a3 == b3, a4 < b4)
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def _do_search(_):
        lo = jnp.int32(0)
        hi = count

        def cond(state):
            lo_i, hi_i = state
            return lo_i < hi_i

        def body(state):
            lo_i, hi_i = state
            mid = (lo_i + hi_i) // 2
            mid_b0 = L_b0[mid]
            mid_b1 = L_b1[mid]
            mid_b2 = L_b2[mid]
            mid_b3 = L_b3[mid]
            mid_b4 = L_b4[mid]
            go_right = _lex_less(mid_b0, mid_b1, mid_b2, mid_b3, mid_b4, k0, k1, k2, k3, k4)
            lo_i = jnp.where(go_right, mid + 1, lo_i)
            hi_i = jnp.where(go_right, hi_i, mid)
            return (lo_i, hi_i)

        pos, _ = lax.while_loop(cond, body, (lo, hi))
        safe_pos = jnp.minimum(pos, count - 1)
        found = (
            (pos < count)
            & (L_b0[safe_pos] == k0)
            & (L_b1[safe_pos] == k1)
            & (L_b2[safe_pos] == k2)
            & (L_b3[safe_pos] == k3)
            & (L_b4[safe_pos] == k4)
        )
        out_id = jnp.where(found, L_ids[safe_pos], jnp.int32(0))
        return out_id, found

    return lax.cond(
        count > 0,
        _do_search,
        lambda _: (jnp.int32(0), jnp.bool_(False)),
        operand=None,
    )

def _intern_nodes_impl_core(ledger, proposed_ops, proposed_a1, proposed_a2):
    max_key = jnp.uint8(0xFF)
    # CORRUPT is semantic (alias risk); OOM is capacity.
    base_corrupt = ledger.corrupt
    base_oom = ledger.oom
    proposed_ops = jnp.where(base_corrupt, jnp.int32(0), proposed_ops)
    proposed_a1 = jnp.where(base_corrupt, jnp.int32(0), proposed_a1)
    proposed_a2 = jnp.where(base_corrupt, jnp.int32(0), proposed_a2)
    is_coord_pair = proposed_ops == OP_COORD_PAIR

    has_coord = jnp.any(is_coord_pair)

    def _norm(args):
        proposed_a1, proposed_a2 = args
        coord_enabled = is_coord_pair.astype(jnp.int32)
        coord_idx, coord_valid, _ = _candidate_indices(coord_enabled)
        coord_idx_safe = jnp.where(coord_valid, coord_idx, 0)
        coord_a1 = proposed_a1[coord_idx_safe]
        coord_a2 = proposed_a2[coord_idx_safe]

        def _maybe_norm(cid, do_norm):
            def _do(_):
                return _coord_norm_id_jax(ledger, cid)

            def _no(_):
                return cid

            return lax.cond(do_norm, _do, _no, operand=None)

        # Normalize only the coord-pair subset to avoid global vmap overhead (m4).
        norm_a1 = jax.vmap(_maybe_norm)(coord_a1, coord_valid)
        norm_a2 = jax.vmap(_maybe_norm)(coord_a2, coord_valid)
        scatter_idx = jnp.where(
            coord_valid, coord_idx_safe, jnp.int32(proposed_a1.shape[0])
        )
        proposed_a1 = _scatter_drop(
            proposed_a1, scatter_idx, norm_a1, "intern_nodes.coord_a1"
        )
        proposed_a2 = _scatter_drop(
            proposed_a2, scatter_idx, norm_a2, "intern_nodes.coord_a2"
        )
        return proposed_a1, proposed_a2

    def _no_norm(args):
        return args

    proposed_a1, proposed_a2 = lax.cond(
        has_coord, _norm, _no_norm, (proposed_a1, proposed_a2)
    )

    P_b0, P_b1, P_b2, P_b3, P_b4 = _pack_key(
        proposed_ops, proposed_a1, proposed_a2
    )
    perm = jnp.lexsort((P_b4, P_b3, P_b2, P_b1, P_b0)).astype(jnp.int32)

    s_b0 = P_b0[perm]
    s_b1 = P_b1[perm]
    s_b2 = P_b2[perm]
    s_b3 = P_b3[perm]
    s_b4 = P_b4[perm]
    s_ops = proposed_ops[perm]
    s_a1 = proposed_a1[perm]
    s_a2 = proposed_a2[perm]
    new_entry_len = s_b0.shape[0]

    is_diff = jnp.concatenate([
        jnp.array([True]),
        (s_b0[1:] != s_b0[:-1])
        | (s_b1[1:] != s_b1[:-1])
        | (s_b2[1:] != s_b2[:-1])
        | (s_b3[1:] != s_b3[:-1])
        | (s_b4[1:] != s_b4[:-1]),
    ])

    idx = jnp.arange(s_b0.shape[0], dtype=jnp.int32)

    def scan_fn(carry, x):
        is_leader, i = x
        new_carry = jnp.where(is_leader, i, carry)
        return new_carry, new_carry

    _, leader_idx = lax.scan(scan_fn, jnp.int32(0), (is_diff, idx))

    L_b0 = ledger.keys_b0_sorted
    L_b1 = ledger.keys_b1_sorted
    L_b2 = ledger.keys_b2_sorted
    L_b3 = ledger.keys_b3_sorted
    L_b4 = ledger.keys_b4_sorted
    L_ids = ledger.ids_sorted

    count = ledger.count.astype(jnp.int32)
    max_nodes = jnp.int32(MAX_NODES)
    available = jnp.maximum(max_nodes - count, 0)
    available = jnp.where(base_oom | base_corrupt, jnp.int32(0), available)
    idx_all = jnp.arange(L_b0.shape[0], dtype=jnp.int32)
    valid_all = idx_all < count
    op_counts = _bincount_256(
        L_b0.astype(jnp.int32),
        valid_all.astype(jnp.int32),
    )
    op_start = jnp.cumsum(op_counts) - op_counts
    op_end = op_start + op_counts

    def _lex_less(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4):
        return jnp.logical_or(
            a0 < b0,
            jnp.logical_and(
                a0 == b0,
                jnp.logical_or(
                    a1 < b1,
                    jnp.logical_and(
                        a1 == b1,
                        jnp.logical_or(
                            a2 < b2,
                            jnp.logical_and(
                                a2 == b2,
                                jnp.logical_or(
                                    a3 < b3, jnp.logical_and(a3 == b3, a4 < b4)
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def _search_one(t_b0, t_b1, t_b2, t_b3, t_b4, start, end):
        lo = start
        hi = end

        def cond(state):
            lo_i, hi_i = state
            return lo_i < hi_i

        def body(state):
            lo_i, hi_i = state
            mid = (lo_i + hi_i) // 2
            mid_b0 = L_b0[mid]
            mid_b1 = L_b1[mid]
            mid_b2 = L_b2[mid]
            mid_b3 = L_b3[mid]
            mid_b4 = L_b4[mid]
            go_right = _lex_less(
                mid_b0,
                mid_b1,
                mid_b2,
                mid_b3,
                mid_b4,
                t_b0,
                t_b1,
                t_b2,
                t_b3,
                t_b4,
            )
            lo_i = jnp.where(go_right, mid + 1, lo_i)
            hi_i = jnp.where(go_right, hi_i, mid)
            return (lo_i, hi_i)

        lo, _ = lax.while_loop(cond, body, (lo, hi))
        return lo

    op_idx = s_b0.astype(jnp.int32)
    op_lo = op_start[op_idx]
    op_hi = op_end[op_idx]
    insert_pos = jax.vmap(_search_one)(s_b0, s_b1, s_b2, s_b3, s_b4, op_lo, op_hi)
    safe_pos = jnp.minimum(insert_pos, count - 1)

    found_match = (
        (insert_pos < count)
        & (L_b0[safe_pos] == s_b0)
        & (L_b1[safe_pos] == s_b1)
        & (L_b2[safe_pos] == s_b2)
        & (L_b3[safe_pos] == s_b3)
        & (L_b4[safe_pos] == s_b4)
    )
    matched_ids = L_ids[safe_pos].astype(jnp.int32)

    is_new = is_diff & (~found_match) & (~(base_oom | base_corrupt))
    requested_new = jnp.sum(is_new.astype(jnp.int32))
    overflow = requested_new > available

    spawn = is_new.astype(jnp.int32)
    prefix = jnp.cumsum(spawn)
    spawn = spawn * (prefix <= available).astype(jnp.int32)
    is_new = spawn.astype(jnp.bool_)
    offsets = jnp.cumsum(spawn) - spawn
    num_new = jnp.sum(spawn).astype(jnp.int32)

    write_start = ledger.count.astype(jnp.int32)
    new_ids_for_sorted = jnp.where(
        found_match,
        matched_ids,
        jnp.where(is_new, write_start + offsets, jnp.int32(0)),
    )

    leader_ids = jnp.where(is_diff, new_ids_for_sorted, jnp.int32(0))
    ids_sorted_order = leader_ids[leader_idx]

    inv_perm = _invert_perm(perm)
    final_ids = ids_sorted_order[inv_perm]

    new_opcode = ledger.opcode
    new_arg1 = ledger.arg1
    new_arg2 = ledger.arg2

    write_idx = jnp.where(is_new, write_start + offsets, jnp.int32(-1))
    valid_w = write_idx >= 0
    safe_w = jnp.where(valid_w, write_idx, jnp.int32(new_opcode.shape[0]))

    new_opcode = _scatter_drop(
        new_opcode,
        safe_w,
        jnp.where(valid_w, s_ops, new_opcode[0]),
        "intern_nodes.new_opcode",
    )
    new_arg1 = _scatter_drop(
        new_arg1,
        safe_w,
        jnp.where(valid_w, s_a1, new_arg1[0]),
        "intern_nodes.new_arg1",
    )
    new_arg2 = _scatter_drop(
        new_arg2,
        safe_w,
        jnp.where(valid_w, s_a2, new_arg2[0]),
        "intern_nodes.new_arg2",
    )

    new_count = ledger.count + num_new
    new_oom = base_oom | overflow
    new_corrupt = base_corrupt
    _guard_max(new_count, max_nodes, "ledger.count")

    new_pos = jnp.where(is_new, offsets, jnp.int32(-1))
    valid_new = new_pos >= 0
    safe_new = jnp.where(valid_new, new_pos, jnp.int32(new_entry_len))

    new_entry_b0_sorted = jnp.full_like(s_b0, max_key)
    new_entry_b1_sorted = jnp.full_like(s_b1, max_key)
    new_entry_b2_sorted = jnp.full_like(s_b2, max_key)
    new_entry_b3_sorted = jnp.full_like(s_b3, max_key)
    new_entry_b4_sorted = jnp.full_like(s_b4, max_key)
    new_entry_ids_sorted = jnp.zeros(new_entry_len, dtype=jnp.int32)

    new_entry_b0_sorted = _scatter_drop(
        new_entry_b0_sorted,
        safe_new,
        jnp.where(valid_new, s_b0, new_entry_b0_sorted[0]),
        "intern_nodes.new_entry_b0",
    )
    new_entry_b1_sorted = _scatter_drop(
        new_entry_b1_sorted,
        safe_new,
        jnp.where(valid_new, s_b1, new_entry_b1_sorted[0]),
        "intern_nodes.new_entry_b1",
    )
    new_entry_b2_sorted = _scatter_drop(
        new_entry_b2_sorted,
        safe_new,
        jnp.where(valid_new, s_b2, new_entry_b2_sorted[0]),
        "intern_nodes.new_entry_b2",
    )
    new_entry_b3_sorted = _scatter_drop(
        new_entry_b3_sorted,
        safe_new,
        jnp.where(valid_new, s_b3, new_entry_b3_sorted[0]),
        "intern_nodes.new_entry_b3",
    )
    new_entry_b4_sorted = _scatter_drop(
        new_entry_b4_sorted,
        safe_new,
        jnp.where(valid_new, s_b4, new_entry_b4_sorted[0]),
        "intern_nodes.new_entry_b4",
    )
    new_entry_ids_sorted = _scatter_drop(
        new_entry_ids_sorted,
        safe_new,
        jnp.where(valid_new, new_ids_for_sorted, new_entry_ids_sorted[0]),
        "intern_nodes.new_entry_ids",
    )

    def _merge_sorted_keys(
        old_b0,
        old_b1,
        old_b2,
        old_b3,
        old_b4,
        old_ids,
        old_count,
        new_b0,
        new_b1,
        new_b2,
        new_b3,
        new_b4,
        new_ids,
        new_items,
    ):
        out_b0 = jnp.full_like(old_b0, max_key)
        out_b1 = jnp.full_like(old_b1, max_key)
        out_b2 = jnp.full_like(old_b2, max_key)
        out_b3 = jnp.full_like(old_b3, max_key)
        out_b4 = jnp.full_like(old_b4, max_key)
        out_ids = jnp.zeros_like(old_ids)
        total = old_count + new_items

        def body(k, state):
            i, j, b0, b1, b2, b3, b4, ids = state
            old_valid = i < old_count
            new_valid = j < new_items
            safe_i = jnp.where(old_valid, i, 0)
            safe_j = jnp.where(new_valid, j, 0)

            old0 = jnp.where(old_valid, old_b0[safe_i], max_key)
            old1 = jnp.where(old_valid, old_b1[safe_i], max_key)
            old2 = jnp.where(old_valid, old_b2[safe_i], max_key)
            old3 = jnp.where(old_valid, old_b3[safe_i], max_key)
            old4 = jnp.where(old_valid, old_b4[safe_i], max_key)

            new0 = jnp.where(new_valid, new_b0[safe_j], max_key)
            new1 = jnp.where(new_valid, new_b1[safe_j], max_key)
            new2 = jnp.where(new_valid, new_b2[safe_j], max_key)
            new3 = jnp.where(new_valid, new_b3[safe_j], max_key)
            new4 = jnp.where(new_valid, new_b4[safe_j], max_key)

            new_less = _lex_less(
                new0,
                new1,
                new2,
                new3,
                new4,
                old0,
                old1,
                old2,
                old3,
                old4,
            )
            take_new = jnp.where(old_valid & new_valid, new_less, new_valid)

            picked0 = jnp.where(take_new, new0, old0)
            picked1 = jnp.where(take_new, new1, old1)
            picked2 = jnp.where(take_new, new2, old2)
            picked3 = jnp.where(take_new, new3, old3)
            picked4 = jnp.where(take_new, new4, old4)

            old_id = jnp.where(old_valid, old_ids[safe_i], jnp.int32(0))
            new_id = jnp.where(new_valid, new_ids[safe_j], jnp.int32(0))
            picked_id = jnp.where(take_new, new_id, old_id)

            b0 = b0.at[k].set(picked0)
            b1 = b1.at[k].set(picked1)
            b2 = b2.at[k].set(picked2)
            b3 = b3.at[k].set(picked3)
            b4 = b4.at[k].set(picked4)
            ids = ids.at[k].set(picked_id)

            i = jnp.where(take_new, i, i + 1)
            j = jnp.where(take_new, j + 1, j)
            return (i, j, b0, b1, b2, b3, b4, ids)

        init_state = (
            jnp.int32(0),
            jnp.int32(0),
            out_b0,
            out_b1,
            out_b2,
            out_b3,
            out_b4,
            out_ids,
        )
        _, _, out_b0, out_b1, out_b2, out_b3, out_b4, out_ids = lax.fori_loop(
            0, total, body, init_state
        )
        return out_b0, out_b1, out_b2, out_b3, out_b4, out_ids

    (
        new_keys_b0_sorted,
        new_keys_b1_sorted,
        new_keys_b2_sorted,
        new_keys_b3_sorted,
        new_keys_b4_sorted,
        new_ids_sorted,
    ) = _merge_sorted_keys(
        L_b0,
        L_b1,
        L_b2,
        L_b3,
        L_b4,
        L_ids,
        count,
        new_entry_b0_sorted,
        new_entry_b1_sorted,
        new_entry_b2_sorted,
        new_entry_b3_sorted,
        new_entry_b4_sorted,
        new_entry_ids_sorted,
        num_new,
    )

    new_ledger = Ledger(
        opcode=new_opcode,
        arg1=new_arg1,
        arg2=new_arg2,
        keys_b0_sorted=new_keys_b0_sorted,
        keys_b1_sorted=new_keys_b1_sorted,
        keys_b2_sorted=new_keys_b2_sorted,
        keys_b3_sorted=new_keys_b3_sorted,
        keys_b4_sorted=new_keys_b4_sorted,
        ids_sorted=new_ids_sorted,
        count=new_count,
        oom=new_oom,
        corrupt=new_corrupt,
    )
    return final_ids, new_ledger


def _intern_nodes_impl(ledger, proposed_ops, proposed_a1, proposed_a2):
    proposed_ops, proposed_a1, proposed_a2 = _canonicalize_nodes(
        proposed_ops, proposed_a1, proposed_a2
    )
    max_id = jnp.int32(MAX_ID)
    # Reject negative ids/opcodes to avoid key aliasing from fixed-width packing.
    op_oob = jnp.any((proposed_ops < 0) | (proposed_ops > jnp.int32(255)))
    bounds_corrupt = (
        (ledger.count > max_id)
        | (ledger.count < 0)
        | jnp.any(proposed_a1 > max_id)
        | jnp.any(proposed_a2 > max_id)
        | jnp.any(proposed_a1 < 0)
        | jnp.any(proposed_a2 < 0)
        | op_oob
    )

    def _corrupt_return(_):
        # Bounds violations are semantic CORRUPT; return zero ids without mutation.
        zero_ids = jnp.zeros_like(proposed_ops)
        new_ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))
        return zero_ids, new_ledger

    def _do(_):
        return _intern_nodes_impl_core(ledger, proposed_ops, proposed_a1, proposed_a2)

    return lax.cond(bounds_corrupt, _corrupt_return, _do, operand=None)


@jit
def intern_nodes(ledger, proposed_ops, proposed_a1, proposed_a2):
    """
    Batch-intern a list of proposed (op,a1,a2) nodes into the canonical Ledger.

    Args:
      ledger: Ledger
      proposed_ops/a1/a2: int32 arrays, shape [N]

    Returns:
      final_ids: int32 array, shape [N], canonical ids for each proposal
      new_ledger: Ledger, updated
    """
    if proposed_ops.shape[0] == 0:
        return jnp.zeros_like(proposed_ops), ledger
    stop = ledger.oom | ledger.corrupt

    # Once invalid, interning must not allocate or mutate state (m1 guardrail).
    def _skip(_):
        return jnp.zeros_like(proposed_ops), ledger

    def _do(_):
        return _intern_nodes_impl(ledger, proposed_ops, proposed_a1, proposed_a2)

    return lax.cond(stop, _skip, _do, operand=None)

def _active_prefix_count(arena):
    size = arena.rank.shape[0]
    count = int(arena.count)
    return size if count > size else count

def _apply_perm_and_swizzle(arena, perm):
    inv_perm = _invert_perm(perm)
    new_ops = arena.opcode[perm]
    new_arg1 = arena.arg1[perm]
    new_arg2 = arena.arg2[perm]
    new_rank = arena.rank[perm]
    # Guard pointer swizzles in test mode; mask to live region; NULL stays pinned at 0.
    ids = jnp.arange(new_arg1.shape[0], dtype=jnp.int32)
    live = ids < arena.count.astype(jnp.int32)
    idx1 = jnp.where(live, new_arg1, jnp.int32(0))
    idx2 = jnp.where(live, new_arg2, jnp.int32(0))
    g1 = safe_gather_1d(inv_perm, idx1, "swizzle.arg1")
    g2 = safe_gather_1d(inv_perm, idx2, "swizzle.arg2")
    swizzled_arg1 = jnp.where(new_arg1 != 0, g1, 0)
    swizzled_arg2 = jnp.where(new_arg2 != 0, g2, 0)
    # Swizzle is renormalization only; denotation must not change (plan).
    _guard_slot0_perm(perm, inv_perm, "swizzle.perm")
    _guard_null_row(new_ops, swizzled_arg1, swizzled_arg2, "swizzle.row0")
    return (
        Arena(new_ops, swizzled_arg1, swizzled_arg2, new_rank, arena.count, arena.oom),
        inv_perm,
    )

@jit
def _op_sort_and_swizzle_with_perm_full(arena):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.int32)
    sort_key = arena.rank.astype(jnp.int32) * (size + 1) + idx
    sort_key = sort_key.at[0].set(jnp.int32(-1))
    perm = jnp.argsort(sort_key)
    return _apply_perm_and_swizzle(arena, perm)

def _op_sort_and_swizzle_with_perm_prefix(arena, active_count):
    size = arena.rank.shape[0]
    if active_count <= 1:
        perm = jnp.arange(size, dtype=jnp.int32)
        return _apply_perm_and_swizzle(arena, perm)
    idx = jnp.arange(active_count, dtype=jnp.int32)
    sort_key = arena.rank[:active_count].astype(jnp.int32) * (active_count + 1) + idx
    sort_key = sort_key.at[0].set(jnp.int32(-1))
    perm_active = jnp.argsort(sort_key)
    tail = jnp.arange(active_count, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return _apply_perm_and_swizzle(arena, perm)

def op_sort_and_swizzle_with_perm(arena):
    active_count = _active_prefix_count(arena)
    size = arena.rank.shape[0]
    if active_count >= size:
        return _op_sort_and_swizzle_with_perm_full(arena)
    return _op_sort_and_swizzle_with_perm_prefix(arena, active_count)

def op_sort_and_swizzle(arena):
    sorted_arena, _ = op_sort_and_swizzle_with_perm(arena)
    return sorted_arena

def _blocked_perm(arena, block_size, morton=None, active_count=None):
    size = int(arena.rank.shape[0])
    if block_size <= 0 or size % block_size != 0:
        raise ValueError("block_size must evenly divide arena size")
    num_blocks = size // block_size
    if active_count is None or active_count >= size:
        active_blocks = num_blocks
    else:
        active_blocks = (active_count + block_size - 1) // block_size
        if active_blocks < 0:
            active_blocks = 0
        if active_blocks > num_blocks:
            active_blocks = num_blocks

    ranks = arena.rank.reshape((num_blocks, block_size)).astype(jnp.uint32)
    idx = jnp.arange(block_size, dtype=jnp.uint32)
    idx_u = idx & jnp.uint32(0xFFFF)
    if morton is None:
        morton_u = jnp.zeros_like(ranks, dtype=jnp.uint32)
    else:
        morton_u = morton.reshape((num_blocks, block_size)).astype(jnp.uint32) & jnp.uint32(0x3FFF)
    # Keep NULL row pinned to preserve slot-0 invariants across permutations.
    ranks = ranks.at[0, 0].set(jnp.uint32(0))
    morton_u = morton_u.at[0, 0].set(jnp.uint32(0))

    if active_blocks <= 0:
        return jnp.arange(size, dtype=jnp.int32)

    if active_blocks == num_blocks:
        sort_key = (ranks << 30) | (morton_u << 16) | idx_u
        perm_local = jnp.argsort(sort_key, axis=1)
        base = (jnp.arange(num_blocks, dtype=jnp.uint32) * block_size)[:, None]
        perm = (base + perm_local).reshape((size,)).astype(jnp.int32)
        return perm

    ranks_active = ranks[:active_blocks]
    morton_active = morton_u[:active_blocks]
    if active_count is not None and active_count < active_blocks * block_size:
        base = (jnp.arange(active_blocks, dtype=jnp.uint32) * block_size)[:, None]
        block_idx = base + idx_u[None, :]
        tail_mask = block_idx >= active_count
        ranks_active = jnp.where(tail_mask, jnp.uint32(RANK_FREE), ranks_active)
        morton_active = jnp.where(tail_mask, jnp.uint32(0), morton_active)

    sort_key = (ranks_active << 30) | (morton_active << 16) | idx_u
    perm_local = jnp.argsort(sort_key, axis=1)
    base = (jnp.arange(active_blocks, dtype=jnp.uint32) * block_size)[:, None]
    perm_active = (base + perm_local).reshape((active_blocks * block_size,)).astype(jnp.int32)
    tail = jnp.arange(active_blocks * block_size, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return perm

def op_sort_and_swizzle_blocked_with_perm(arena, block_size, morton=None):
    active_count = _active_prefix_count(arena)
    perm = _blocked_perm(arena, block_size, morton=morton, active_count=active_count)
    return _apply_perm_and_swizzle(arena, perm)

def op_sort_and_swizzle_blocked(arena, block_size, morton=None):
    sorted_arena, _ = op_sort_and_swizzle_blocked_with_perm(
        arena, block_size, morton=morton
    )
    return sorted_arena

def _apply_perm_to_morton(morton, inv_perm):
    if morton is None:
        return None
    perm = _invert_perm(inv_perm)
    return morton[perm]

def _walk_block_sizes(start_block_size, size):
    sizes = []
    if start_block_size <= 0 or start_block_size >= size:
        return sizes
    block_size = start_block_size
    while block_size < size:
        next_block = block_size * 2
        if next_block >= size:
            sizes.append(size)
            break
        if size % next_block != 0:
            sizes.append(size)
            break
        sizes.append(next_block)
        block_size = next_block
    return sizes

def op_sort_and_swizzle_hierarchical_with_perm(
    arena, l2_block_size, l1_block_size, morton=None, do_global=False
):
    size = int(arena.rank.shape[0])
    if l2_block_size <= 0 or l1_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if size % l2_block_size != 0 or size % l1_block_size != 0:
        raise ValueError("block sizes must evenly divide arena size")
    if l1_block_size % l2_block_size != 0:
        raise ValueError("l1_block_size must be a multiple of l2_block_size")

    arena, inv_perm = op_sort_and_swizzle_blocked_with_perm(
        arena, l2_block_size, morton=morton
    )
    morton = _apply_perm_to_morton(morton, inv_perm)
    inv_perm_total = inv_perm

    if l1_block_size > l2_block_size:
        arena, inv_perm_l1 = op_sort_and_swizzle_blocked_with_perm(
            arena, l1_block_size, morton=morton
        )
        morton = _apply_perm_to_morton(morton, inv_perm_l1)
        inv_perm_total = inv_perm_l1[inv_perm_total]

    if do_global and l1_block_size < size:
        for block_size in _walk_block_sizes(l1_block_size, size):
            arena, inv_perm_global = op_sort_and_swizzle_blocked_with_perm(
                arena, block_size, morton=morton
            )
            morton = _apply_perm_to_morton(morton, inv_perm_global)
            inv_perm_total = inv_perm_global[inv_perm_total]

    return arena, inv_perm_total

def op_sort_and_swizzle_hierarchical(
    arena, l2_block_size, l1_block_size, morton=None, do_global=False
):
    sorted_arena, _ = op_sort_and_swizzle_hierarchical_with_perm(
        arena,
        l2_block_size,
        l1_block_size,
        morton=morton,
        do_global=do_global,
    )
    return sorted_arena

def swizzle_2to1_host(x, y):
    z = 0
    for i in range(10):
        x0 = (x >> (2 * i)) & 1
        x1 = (x >> (2 * i + 1)) & 1
        y0 = (y >> i) & 1
        z |= (x0 << (3 * i))
        z |= (x1 << (3 * i + 1))
        z |= (y0 << (3 * i + 2))
    return z

@jit
def swizzle_2to1_dev(x, y):
    x = x.astype(jnp.uint32)
    y = y.astype(jnp.uint32)
    z = jnp.zeros_like(x, dtype=jnp.uint32)

    def body(i, val):
        z_acc, x_in, y_in = val
        x_bits = x_in & jnp.uint32(3)
        y_bit = y_in & jnp.uint32(1)
        chunk = (y_bit << 2) | x_bits
        z_acc = z_acc | (chunk << (3 * i))
        return (z_acc, x_in >> 2, y_in >> 1)

    res, _, _ = lax.fori_loop(0, 10, body, (z, x, y))
    return res

def _build_pallas_swizzle(backend):
    try:
        import jax as jax_module
        import jax.experimental.pallas as pl
        if backend == "triton":
            import jax.experimental.pallas.triton  # noqa: F401
    except Exception:
        return None

    if jax_module.default_backend() == "cpu":
        return None
    if backend == "triton" and jax_module.default_backend() != "gpu":
        return None

    def kernel(x_ref, y_ref, out_ref):
        x_val = x_ref[0].astype(jnp.uint32)
        y_val = y_ref[0].astype(jnp.uint32)
        z = jnp.uint32(0)
        for i in range(10):
            x_bits = x_val & jnp.uint32(3)
            y_bit = y_val & jnp.uint32(1)
            chunk = (y_bit << 2) | x_bits
            z = z | (chunk << (3 * i))
            x_val = x_val >> 2
            y_val = y_val >> 1
        out_ref[0] = z

    def swizzle(x, y):
        out_shape = jax_module.ShapeDtypeStruct(x.shape, jnp.uint32)
        in_specs = [
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
        ]
        out_specs = pl.BlockSpec((1,), lambda i: (i,))
        grid = (x.shape[0],)
        return pl.pallas_call(
            kernel,
            out_shape=out_shape,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            backend="triton" if backend == "triton" else None,
        )(x, y)

    return swizzle

_SWIZZLE_BACKEND = os.environ.get("PRISM_SWIZZLE_BACKEND", "jax").strip().lower()
_SWIZZLE_ACCEL = None
if _SWIZZLE_BACKEND in ("pallas", "triton"):
    _SWIZZLE_ACCEL = _build_pallas_swizzle(_SWIZZLE_BACKEND)
    if _SWIZZLE_ACCEL is None:
        _SWIZZLE_BACKEND = "jax"

def swizzle_2to1(x, y):
    if _SWIZZLE_ACCEL is not None:
        return _SWIZZLE_ACCEL(x, y)
    return swizzle_2to1_dev(x, y)

@jit
def op_morton(arena):
    size = arena.opcode.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    x = idx
    y = jnp.zeros_like(idx)
    return swizzle_2to1(x, y)

@jit
def _op_sort_and_swizzle_morton_with_perm_full(arena, morton):
    size = arena.rank.shape[0]
    idx = jnp.arange(size, dtype=jnp.uint32)
    rank_u = arena.rank.astype(jnp.uint32)
    morton_u = morton.astype(jnp.uint32) & jnp.uint32(0x3FFF)
    idx_u = idx & jnp.uint32(0xFFFF)
    sort_key = (rank_u << 30) | (morton_u << 16) | idx_u
    sort_key = sort_key.at[0].set(jnp.uint32(0))
    perm = jnp.argsort(sort_key).astype(jnp.int32)
    return _apply_perm_and_swizzle(arena, perm)

def _op_sort_and_swizzle_morton_with_perm_prefix(arena, morton, active_count):
    size = arena.rank.shape[0]
    if active_count <= 1:
        perm = jnp.arange(size, dtype=jnp.int32)
        return _apply_perm_and_swizzle(arena, perm)
    idx = jnp.arange(active_count, dtype=jnp.uint32)
    rank_u = arena.rank[:active_count].astype(jnp.uint32)
    morton_u = morton[:active_count].astype(jnp.uint32) & jnp.uint32(0x3FFF)
    idx_u = idx & jnp.uint32(0xFFFF)
    sort_key = (rank_u << 30) | (morton_u << 16) | idx_u
    sort_key = sort_key.at[0].set(jnp.uint32(0))
    perm_active = jnp.argsort(sort_key).astype(jnp.int32)
    tail = jnp.arange(active_count, size, dtype=jnp.int32)
    perm = jnp.concatenate([perm_active, tail], axis=0)
    return _apply_perm_and_swizzle(arena, perm)

def op_sort_and_swizzle_morton_with_perm(arena, morton):
    active_count = _active_prefix_count(arena)
    size = arena.rank.shape[0]
    if active_count >= size:
        return _op_sort_and_swizzle_morton_with_perm_full(arena, morton)
    return _op_sort_and_swizzle_morton_with_perm_prefix(arena, morton, active_count)

def op_sort_and_swizzle_morton(arena, morton):
    sorted_arena, _ = op_sort_and_swizzle_morton_with_perm(arena, morton)
    return sorted_arena

@jit
def op_interact(arena):
    ops = arena.opcode
    a1 = arena.arg1
    a2 = arena.arg2
    cap = jnp.int32(ops.shape[0])
    is_hot = arena.rank == RANK_HOT
    is_add = ops == OP_ADD
    # Guard pointer gathers in test mode; avoid touching inactive garbage rows.
    a1_for_op = jnp.where(is_hot & is_add, a1, jnp.int32(0))
    op_x = safe_gather_1d(ops, a1_for_op, "op_interact.op_x")
    mask_zero = is_hot & is_add & (op_x == OP_ZERO)
    mask_suc = is_hot & is_add & (op_x == OP_SUC) & (~arena.oom)

    # First: local rewrites that don't allocate.
    a2_for_zero = jnp.where(mask_zero, a2, jnp.int32(0))
    y_op = safe_gather_1d(ops, a2_for_zero, "op_interact.y_op")
    y_a1 = safe_gather_1d(a1, a2_for_zero, "op_interact.y_a1")
    y_a2 = safe_gather_1d(a2, a2_for_zero, "op_interact.y_a2")
    new_ops = jnp.where(mask_zero, y_op, ops)
    new_a1 = jnp.where(mask_zero, y_a1, a1)
    new_a2 = jnp.where(mask_zero, y_a2, a2)

    # Second: allocation for suc-case.
    available = jnp.maximum(cap - arena.count, 0)
    spawn = mask_suc.astype(jnp.int32)
    prefix = jnp.cumsum(spawn)
    spawn = spawn * (prefix <= available).astype(jnp.int32)
    offsets = jnp.cumsum(spawn) - spawn
    total_spawn = jnp.sum(spawn).astype(jnp.int32)
    base_free = arena.count
    new_add_idx = base_free + offsets

    spawn_mask = spawn.astype(jnp.bool_)
    new_ops = jnp.where(spawn_mask, OP_SUC, new_ops)
    new_a1 = jnp.where(spawn_mask, new_add_idx, new_a1)
    new_a2 = jnp.where(spawn_mask, 0, new_a2)

    # Scatter-create the spawned add nodes only where mask_suc is true.
    idxs = jnp.where(spawn_mask, new_add_idx, jnp.int32(-1))
    a1_for_spawn = jnp.where(spawn_mask, a1, jnp.int32(0))
    grandchild_x = safe_gather_1d(a1, a1_for_spawn, "op_interact.grandchild_x")
    payload_op = jnp.full_like(idxs, OP_ADD)
    payload_a1_raw = jnp.where(spawn_mask, grandchild_x, jnp.int32(0))
    payload_a2_raw = jnp.where(spawn_mask, a2, jnp.int32(0))
    payload_swap = payload_a2_raw < payload_a1_raw
    payload_a1 = jnp.where(payload_swap, payload_a2_raw, payload_a1_raw)
    payload_a2 = jnp.where(payload_swap, payload_a1_raw, payload_a2_raw)

    valid = idxs >= 0
    idxs2 = jnp.where(valid, idxs, cap)

    final_ops = _scatter_drop(
        new_ops,
        idxs2,
        jnp.where(valid, payload_op, new_ops[0]),
        "op_interact.final_ops",
    )
    final_a1 = _scatter_drop(
        new_a1,
        idxs2,
        jnp.where(valid, payload_a1, new_a1[0]),
        "op_interact.final_a1",
    )
    final_a2 = _scatter_drop(
        new_a2,
        idxs2,
        jnp.where(valid, payload_a2, new_a2[0]),
        "op_interact.final_a2",
    )

    overflow = jnp.sum(mask_suc.astype(jnp.int32)) > available
    new_oom = arena.oom | overflow
    new_count = arena.count + total_spawn
    _guard_max(new_count, cap, "arena.count")
    return Arena(
        final_ops, final_a1, final_a2, arena.rank, new_count, new_oom
    )

@jit
def cycle_intrinsic(ledger, frontier_ids):
    # m1 evaluator: intrinsic rewrite steps on Ledger (CNF-2 gated off).
    def _peel_one(ptr):
        def cond(state):
            curr, _ = state
            return ledger.opcode[curr] == OP_SUC

        def body(state):
            curr, depth = state
            return ledger.arg1[curr], depth + 1

        return lax.while_loop(cond, body, (ptr, jnp.int32(0)))

    base_ids, depths = jax.vmap(_peel_one)(frontier_ids)

    t_ops = ledger.opcode[base_ids]
    t_a1 = ledger.arg1[base_ids]
    t_a2 = ledger.arg2[base_ids]
    op_a1 = ledger.opcode[t_a1]
    op_a2 = ledger.opcode[t_a2]
    is_add = t_ops == OP_ADD
    is_mul = t_ops == OP_MUL
    is_zero_a1 = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    is_suc_a1 = op_a1 == OP_SUC
    is_suc_a2 = op_a2 == OP_SUC
    is_add_suc = is_add & (is_suc_a1 | is_suc_a2)
    is_mul_suc = is_mul & (is_suc_a1 | is_suc_a2)
    is_add_zero = is_add & (is_zero_a1 | is_zero_a2)
    is_mul_zero = is_mul & (is_zero_a1 | is_zero_a2)
    is_add_suc = is_add_suc & (~is_add_zero)
    is_mul_suc = is_mul_suc & (~is_mul_zero)
    zero_on_a1 = is_zero_a1
    zero_on_a2 = (~is_zero_a1) & is_zero_a2
    zero_other = jnp.where(zero_on_a1, t_a2, t_a1)

    suc_on_a1 = is_suc_a1
    suc_on_a2 = (~is_suc_a1) & is_suc_a2
    suc_node = jnp.where(suc_on_a1, t_a1, t_a2)
    val_x = ledger.arg1[suc_node]
    val_y = jnp.where(suc_on_a1, t_a2, t_a1)

    l1_ops = jnp.zeros_like(t_ops)
    l1_a1 = jnp.zeros_like(t_a1)
    l1_a2 = jnp.zeros_like(t_a2)
    l1_ops = jnp.where(is_add_suc, OP_ADD, l1_ops)
    l1_a1 = jnp.where(is_add_suc, val_x, l1_a1)
    l1_a2 = jnp.where(is_add_suc, val_y, l1_a2)
    l1_ops = jnp.where(is_mul_suc, OP_MUL, l1_ops)
    l1_a1 = jnp.where(is_mul_suc, val_x, l1_a1)
    l1_a2 = jnp.where(is_mul_suc, val_y, l1_a2)

    l1_ids, ledger = intern_nodes(ledger, l1_ops, l1_a1, l1_a2)

    l2_ops = jnp.zeros_like(t_ops)
    l2_a1 = jnp.zeros_like(t_a1)
    l2_a2 = jnp.zeros_like(t_a2)
    l2_ops = jnp.where(is_add_suc, OP_SUC, l2_ops)
    l2_a1 = jnp.where(is_add_suc, l1_ids, l2_a1)
    l2_ops = jnp.where(is_mul_suc, OP_ADD, l2_ops)
    l2_a1 = jnp.where(is_mul_suc, val_y, l2_a1)
    l2_a2 = jnp.where(is_mul_suc, l1_ids, l2_a2)

    l2_ids, ledger = intern_nodes(ledger, l2_ops, l2_a1, l2_a2)

    base_next = base_ids
    base_next = jnp.where(is_add_zero, zero_other, base_next)
    base_next = jnp.where(is_mul_zero, jnp.int32(ZERO_PTR), base_next)
    base_next = jnp.where(is_add_suc, l2_ids, base_next)
    base_next = jnp.where(is_mul_suc, l2_ids, base_next)
    changed = base_next != base_ids
    wrap_depth = jnp.where(changed, depths, jnp.int32(0))
    wrap_child = jnp.where(changed, base_next, frontier_ids)

    def wrap_cond(state):
        depth, _, led = state
        return jnp.any((depth > 0) & (~led.oom))

    def wrap_body(state):
        depth, child, led = state
        to_wrap = (depth > 0) & (~led.oom)
        ops = jnp.where(to_wrap, jnp.int32(OP_SUC), jnp.int32(0))
        a1 = jnp.where(to_wrap, child, jnp.int32(0))
        a2 = jnp.zeros_like(a1)
        new_ids, led = intern_nodes(led, ops, a1, a2)
        child = jnp.where(to_wrap, new_ids, child)
        depth = depth - to_wrap.astype(jnp.int32)
        return depth, child, led

    _, wrap_child, ledger = lax.while_loop(
        wrap_cond, wrap_body, (wrap_depth, wrap_child, ledger)
    )
    return ledger, wrap_child

def cycle(
    arena,
    root_ptr,
    do_sort=True,
    use_morton=False,
    block_size=None,
    morton=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
):
    """Run one BSP cycle; sorting/scheduling is renormalization only."""
    arena = op_rank(arena)
    root_arr = jnp.asarray(root_ptr, dtype=jnp.int32)
    if do_sort:
        morton_arr = None
        if use_morton or morton is not None:
            morton_arr = morton if morton is not None else op_morton(arena)
        if l2_block_size is not None or l1_block_size is not None:
            if l2_block_size is None:
                l2_block_size = l1_block_size
            if l1_block_size is None:
                l1_block_size = l2_block_size
            arena, inv_perm = op_sort_and_swizzle_hierarchical_with_perm(
                arena,
                l2_block_size,
                l1_block_size,
                morton=morton_arr,
                do_global=do_global,
            )
        elif block_size is not None:
            arena, inv_perm = op_sort_and_swizzle_blocked_with_perm(
                arena, block_size, morton=morton_arr
            )
        elif morton_arr is not None:
            arena, inv_perm = op_sort_and_swizzle_morton_with_perm(arena, morton_arr)
        else:
            arena, inv_perm = op_sort_and_swizzle_with_perm(arena)
        # Root remap is a pointer gather; guard in test mode.
        root_idx = jnp.where(root_arr != 0, root_arr, jnp.int32(0))
        root_g = safe_gather_1d(inv_perm, root_idx, "cycle.root_remap")
        root_arr = jnp.where(root_arr != 0, root_g, 0)
    arena = op_interact(arena)
    return arena, root_arr

# --- 3. JAX Kernels (Static) ---
# --- 3. JAX Kernels (Static) ---
@jit
def kernel_add(manifest, ptr):
    ops, a1, a2, count, oom = (
        manifest.opcode,
        manifest.arg1,
        manifest.arg2,
        manifest.active_count,
        manifest.oom,
    )
    cap = ops.shape[0]
    init_x = a1[ptr]
    init_y = a2[ptr]
    init_val = (init_x, init_y, True, ops, a1, count, oom)

    def cond(v): return v[2]

    def body(v):
        curr_x, curr_y, active, b_ops, b_a1, b_count, b_oom = v
        op_x = b_ops[curr_x]
        is_suc = (op_x == OP_SUC) & (~b_oom)
        next_x = jnp.where(is_suc, b_a1[curr_x], curr_x)

        def do_spawn(state):
            ops, a1s, count, y_val, oom = state
            ok = (count < cap) & (~oom)
            w_idx = jnp.where(ok, count, cap)
            ops = _scatter_drop(ops, w_idx, OP_SUC, "kernel_add.ops")
            a1s = _scatter_drop(a1s, w_idx, y_val, "kernel_add.a1s")
            next_count = jnp.where(ok, count + 1, count)
            next_y = jnp.where(ok, w_idx, y_val)
            next_oom = oom | (~ok)
            return ops, a1s, next_count, next_y, next_oom

        def no_spawn(state):
            ops, a1s, count, y_val, oom = state
            return ops, a1s, count, y_val, oom

        b_ops, b_a1, next_count, next_y, next_oom = lax.cond(
            is_suc,
            do_spawn,
            no_spawn,
            (b_ops, b_a1, b_count, curr_y, b_oom),
        )
        return (next_x, next_y, is_suc, b_ops, b_a1, next_count, next_oom)

    _, final_y, _, f_ops, f_a1, f_count, f_oom = lax.while_loop(
        cond, body, init_val
    )
    return (
        manifest._replace(opcode=f_ops, arg1=f_a1, active_count=f_count, oom=f_oom),
        final_y,
    )

# Kernel stub for MUL
@jit
def kernel_mul(manifest, ptr):
    ops, a1, a2, count, oom = (
        manifest.opcode,
        manifest.arg1,
        manifest.arg2,
        manifest.active_count,
        manifest.oom,
    )
    cap = ops.shape[0]
    init_x = a1[ptr]
    y = a2[ptr]
    init_acc = jnp.array(ZERO_PTR, dtype=jnp.int32)
    init_val = (init_x, init_acc, ops, a1, a2, count, oom)

    def cond(v):
        curr_x, _, b_ops, _, _, _, b_oom = v
        return (b_ops[curr_x] == OP_SUC) & (~b_oom)

    def body(v):
        curr_x, acc, b_ops, b_a1, b_a2, b_count, b_oom = v
        next_x = b_a1[curr_x]
        ok = (b_count < cap) & (~b_oom)

        def do_add(state):
            b_ops, b_a1, b_a2, b_count, b_oom, acc = state
            add_idx = b_count
            b_ops = _scatter_drop(b_ops, add_idx, OP_ADD, "kernel_mul.b_ops")
            add_a1_raw = y
            add_a2_raw = acc
            add_swap = add_a2_raw < add_a1_raw
            add_a1 = jnp.where(add_swap, add_a2_raw, add_a1_raw)
            add_a2 = jnp.where(add_swap, add_a1_raw, add_a2_raw)
            b_a1 = _scatter_drop(b_a1, add_idx, add_a1, "kernel_mul.b_a1")
            b_a2 = _scatter_drop(b_a2, add_idx, add_a2, "kernel_mul.b_a2")
            b_count = b_count + 1
            add_manifest = Manifest(b_ops, b_a1, b_a2, b_count, b_oom)
            updated_manifest, next_acc = kernel_add(add_manifest, add_idx)
            return (
                updated_manifest.opcode,
                updated_manifest.arg1,
                updated_manifest.arg2,
                updated_manifest.active_count,
                updated_manifest.oom,
                next_acc,
            )

        def no_add(state):
            b_ops, b_a1, b_a2, b_count, b_oom, acc = state
            return b_ops, b_a1, b_a2, b_count, b_oom | (~ok), acc

        b_ops, b_a1, b_a2, b_count, b_oom, next_acc = lax.cond(
            ok,
            do_add,
            no_add,
            (b_ops, b_a1, b_a2, b_count, b_oom, acc),
        )
        return (next_x, next_acc, b_ops, b_a1, b_a2, b_count, b_oom)

    _, final_acc, f_ops, f_a1, f_a2, f_count, f_oom = lax.while_loop(
        cond, body, init_val
    )
    return (
        manifest._replace(
            opcode=f_ops, arg1=f_a1, arg2=f_a2, active_count=f_count, oom=f_oom
        ),
        final_acc,
    )

def _dispatch_identity(args):
    manifest, ptr = args
    return manifest, ptr

def _dispatch_add(args):
    manifest, ptr = args
    return kernel_add(manifest, ptr)

def _dispatch_mul(args):
    manifest, ptr = args
    return kernel_mul(manifest, ptr)

@jit
def optimize_ptr(manifest, ptr):
    ops, a1s, a2s = manifest.opcode, manifest.arg1, manifest.arg2
    op = ops[ptr]
    a1 = a1s[ptr]
    a2 = a2s[ptr]
    op_a1 = ops[a1]
    op_a2 = ops[a2]
    is_add = op == OP_ADD
    is_mul = op == OP_MUL
    is_zero = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    add_zero_left = is_add & is_zero
    add_zero_right = is_add & is_zero_a2
    mul_zero_left = is_mul & is_zero
    mul_zero_right = is_mul & is_zero_a2
    optimized = add_zero_left | add_zero_right | mul_zero_left | mul_zero_right
    out_ptr = jnp.where(add_zero_left, a2, ptr)
    out_ptr = jnp.where(add_zero_right, a1, out_ptr)
    out_ptr = jnp.where(mul_zero_left | mul_zero_right, jnp.int32(ZERO_PTR), out_ptr)
    return out_ptr, optimized

@jit
def dispatch_kernel(manifest, ptr):
    opt_ptr, opt_applied = optimize_ptr(manifest, ptr)
    op = manifest.opcode[opt_ptr]
    case_index = jnp.where(op == OP_ADD, 1, jnp.where(op == OP_MUL, 2, 0))
    new_manifest, res_ptr = lax.switch(
        case_index,
        (_dispatch_identity, _dispatch_add, _dispatch_mul),
        (manifest, opt_ptr),
    )
    return new_manifest, res_ptr, opt_applied


# --- 4. Prism VM (Host Logic) ---
class PrismVM:
    def __init__(self):
        print("⚡ Prism IR: Initializing Host Context...")
        # Baseline oracle (Manifest + host cache) kept for cross-engine checks.
        self.manifest = init_manifest()
        self.active_count_host = int(self.manifest.active_count)
        self.refresh_cache_on_eval = True
        # Trace Cache: (opcode, arg1, arg2) -> ptr
        self.trace_cache: Dict[Tuple[int, int, int], int] = {}
        self.canonical_ptrs: Dict[int, int] = {0: 0}
        # Initialize Universe (Seed with ZERO)
        self._cons_raw(OP_ZERO, 0, 0)
        self.trace_cache[(OP_ZERO, 0, 0)] = 1
        self.canonical_ptrs[1] = 1
        self.cache_filled_to = self.active_count_host

        self.kernels = {OP_ADD: kernel_add, OP_MUL: kernel_mul}

    def _cons_raw(self, op, a1, a2):
        """Physical allocation (Device Write)"""
        cap = int(self.manifest.opcode.shape[0])
        if self.active_count_host >= cap:
            self.manifest = self.manifest._replace(
                oom=jnp.array(True, dtype=jnp.bool_)
            )
            raise ValueError("Manifest capacity exceeded")
        idx = self.active_count_host
        self.active_count_host += 1
        self.manifest = self.manifest._replace(
            opcode=self.manifest.opcode.at[idx].set(op),
            arg1=self.manifest.arg1.at[idx].set(a1),
            arg2=self.manifest.arg2.at[idx].set(a2),
            active_count=jnp.array(self.active_count_host, dtype=jnp.int32)
        )
        return idx

    def _refresh_trace_cache(self, start_idx, end_idx):
        if end_idx <= start_idx:
            return
        ops = jax.device_get(self.manifest.opcode[start_idx:end_idx])
        a1s = jax.device_get(self.manifest.arg1[start_idx:end_idx])
        a2s = jax.device_get(self.manifest.arg2[start_idx:end_idx])
        for offset, (op, a1, a2) in enumerate(zip(ops, a1s, a2s)):
            ptr = start_idx + offset
            op_i = int(op)
            a1_i = self._canonical_ptr(int(a1))
            a2_i = self._canonical_ptr(int(a2))
            a1_i, a2_i = _canonicalize_commutative_host(op_i, a1_i, a2_i)
            signature = (op_i, a1_i, a2_i)
            canonical = self.trace_cache.get(signature, ptr)
            self.trace_cache[signature] = canonical
            self.canonical_ptrs[ptr] = canonical
            if canonical == ptr:
                self.canonical_ptrs[canonical] = canonical

    def _canonical_ptr(self, ptr):
        return self.canonical_ptrs.get(ptr, ptr)

    def cons(self, op, a1=0, a2=0):
        """
        The Smart Allocator.
        1. Checks Cache (Deduplication).
        2. Allocates if new.
        """
        a1 = self._canonical_ptr(a1)
        a2 = self._canonical_ptr(a2)
        a1, a2 = _canonicalize_commutative_host(op, a1, a2)
        signature = (op, a1, a2)
        if signature in self.trace_cache:
            return self.trace_cache[signature]
        ptr = self._cons_raw(op, a1, a2)
        self.trace_cache[signature] = ptr
        self.canonical_ptrs[ptr] = ptr
        return ptr

    # --- STATIC ANALYSIS ENGINE ---
    def analyze_and_optimize(self, ptr):
        """
        Examines the IR at `ptr` BEFORE execution.
        Performs trivial reductions (Constant Folding / Identity Elimination).
        """
        ptr = self._canonical_ptr(ptr)
        ptr_arr = jnp.array(ptr, dtype=jnp.int32)
        opt_ptr, opt_applied = optimize_ptr(self.manifest, ptr_arr)
        if bool(opt_applied):
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        return int(self._canonical_ptr(int(opt_ptr)))

    def eval(self, ptr):
        """
        The Hybrid Interpreter.
        1. Static Analysis (Host)
        2. Dispatch (Device)
        """
        ptr = self._canonical_ptr(ptr)
        ptr_arr = jnp.array(ptr, dtype=jnp.int32)
        prev_count = self.active_count_host
        new_manifest, res_ptr, opt_applied = dispatch_kernel(self.manifest, ptr_arr)
        res_ptr.block_until_ready()
        self.manifest = new_manifest
        self.active_count_host = int(self.manifest.active_count)
        if self.refresh_cache_on_eval and self.active_count_host > prev_count:
            self._refresh_trace_cache(prev_count, self.active_count_host)
            self.cache_filled_to = self.active_count_host
        if bool(self.manifest.oom):
            raise RuntimeError("Manifest capacity exceeded during kernel execution")
        if bool(opt_applied):
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        return int(self._canonical_ptr(int(res_ptr)))

    # --- PARSING & DISPLAY ---
    def parse(self, tokens):
        # Explicit token pops keep parse errors readable for malformed input.
        token = _pop_token(tokens)
        if token == 'zero': return self.cons(OP_ZERO)
        if token == 'suc':  return self.cons(OP_SUC, self.parse(tokens))
        if token in ['add', 'mul']:
            op = OP_ADD if token == 'add' else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self.cons(op, a1, a2)
        if token == '(': 
            val = self.parse(tokens)
            _expect_token(tokens, ')')
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr):
        op = int(self.manifest.opcode[ptr])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:  return f"(suc {self.decode(int(self.manifest.arg1[ptr]))})"
        if op == OP_ADD:
            return (
                f"(add {self.decode(int(self.manifest.arg1[ptr]))} "
                f"{self.decode(int(self.manifest.arg2[ptr]))})"
            )
        if op == OP_MUL:
            return (
                f"(mul {self.decode(int(self.manifest.arg1[ptr]))} "
                f"{self.decode(int(self.manifest.arg2[ptr]))})"
            )
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


class PrismVM_BSP_Legacy:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Arena (Legacy)...")
        self.arena = init_arena()

    def _alloc(self, op, a1=0, a2=0):
        cap = int(self.arena.opcode.shape[0])
        idx = int(self.arena.count)
        if idx >= cap:
            self.arena = self.arena._replace(oom=jnp.array(True, dtype=jnp.bool_))
            raise ValueError("Arena capacity exceeded")
        a1, a2 = _canonicalize_commutative_host(op, a1, a2)
        self.arena = self.arena._replace(
            opcode=self.arena.opcode.at[idx].set(op),
            arg1=self.arena.arg1.at[idx].set(a1),
            arg2=self.arena.arg2.at[idx].set(a2),
            count=jnp.array(idx + 1, dtype=jnp.int32),
        )
        return idx

    def parse(self, tokens):
        token = _pop_token(tokens)
        if token == "zero": return self._alloc(OP_ZERO, 0, 0)
        if token == "suc":  return self._alloc(OP_SUC, self.parse(tokens), 0)
        if token in ["add", "mul"]:
            op = OP_ADD if token == "add" else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self._alloc(op, a1, a2)
        if token == "(":
            val = self.parse(tokens)
            _expect_token(tokens, ")")
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr, show_ids=False):
        op = int(self.arena.opcode[ptr])
        if op == OP_ZERO:
            return "zero"
        if op == OP_SUC:
            return f"(suc {self.decode(int(self.arena.arg1[ptr]), show_ids=show_ids)})"
        name = OP_NAMES.get(op, "?")
        if show_ids:
            return f"<{name}:{ptr}>"
        return f"<{name}>"

class PrismVM_BSP:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Ledger...")
        # Canonical Ledger path (univalence + deterministic interning).
        self.ledger = init_ledger()

    def _intern(self, op, a1=0, a2=0):
        a1, a2 = _canonicalize_commutative_host(op, a1, a2)
        ids, self.ledger = intern_nodes(
            self.ledger,
            jnp.array([op], dtype=jnp.int32),
            jnp.array([a1], dtype=jnp.int32),
            jnp.array([a2], dtype=jnp.int32),
        )
        self.ledger.count.block_until_ready()
        if bool(self.ledger.corrupt):
            raise RuntimeError("CORRUPT: key encoding alias risk (id width exceeded)")
        if bool(self.ledger.oom):
            raise ValueError("Ledger capacity exceeded during interning")
        return int(ids[0])

    def parse(self, tokens):
        token = _pop_token(tokens)
        if token == "zero": return self._intern(OP_ZERO, 0, 0)
        if token == "suc":  return self._intern(OP_SUC, self.parse(tokens), 0)
        if token in ["add", "mul"]:
            op = OP_ADD if token == "add" else OP_MUL
            a1 = self.parse(tokens)
            a2 = self.parse(tokens)
            return self._intern(op, a1, a2)
        if token == "(":
            val = self.parse(tokens)
            _expect_token(tokens, ")")
            return val
        raise ValueError(f"Unknown: {token}")

    def decode(self, ptr):
        op = int(self.ledger.opcode[ptr])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:  return f"(suc {self.decode(int(self.ledger.arg1[ptr]))})"
        if op == OP_ADD:
            return (
                f"(add {self.decode(int(self.ledger.arg1[ptr]))} "
                f"{self.decode(int(self.ledger.arg2[ptr]))})"
            )
        if op == OP_MUL:
            return (
                f"(mul {self.decode(int(self.ledger.arg1[ptr]))} "
                f"{self.decode(int(self.ledger.arg2[ptr]))})"
            )
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


def make_vm(mode="baseline"):
    if mode == "bsp":
        return PrismVM_BSP()
    return PrismVM()

def _rank_counts(arena):
    hot = int(jnp.sum(arena.rank == RANK_HOT))
    warm = int(jnp.sum(arena.rank == RANK_WARM))
    cold = int(jnp.sum(arena.rank == RANK_COLD))
    free = int(jnp.sum(arena.rank == RANK_FREE))
    return hot, warm, cold, free

# --- 5. Telemetric REPL ---
def run_program_lines(lines, vm=None):
    if vm is None:
        vm = PrismVM()
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith('#'):
            continue
        start_rows = int(vm.manifest.active_count)
        t0 = time.perf_counter()
        tokens = re.findall(r'\(|\)|[a-z]+', inp)
        ir_ptr = vm.parse(tokens)
        parse_ms = (time.perf_counter() - t0) * 1000
        mid_rows = int(vm.manifest.active_count)
        ir_allocs = mid_rows - start_rows
        ir_op = OP_NAMES.get(int(vm.manifest.opcode[ir_ptr]), "?")
        print(f"   ├─ IR Build: {ir_op} @ {ir_ptr}")
        if ir_allocs == 0:
            print(f"   ├─ Cache   : \033[96mHIT (No new IR rows)\033[0m")
        else:
            print(f"   ├─ Cache   : MISS (+{ir_allocs} IR rows)")
        t1 = time.perf_counter()
        res_ptr = vm.eval(ir_ptr)
        eval_ms = (time.perf_counter() - t1) * 1000
        end_rows = int(vm.manifest.active_count)
        exec_allocs = end_rows - mid_rows
        print(f"   ├─ Execute : {eval_ms:.2f}ms")
        if exec_allocs > 0:
            print(f"   ├─ Kernel  : +{exec_allocs} rows allocated")
        else:
            print(f"   ├─ Kernel  : \033[96mSKIPPED (Static Optimization)\033[0m")
        print(f"   └─ Result  : \033[92m{vm.decode(res_ptr)}\033[0m")
    return vm

def run_program_lines_bsp(
    lines,
    vm=None,
    cycles=1,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
    bsp_mode="auto",
    validate_stratum=False,
):
    if vm is None:
        vm = PrismVM_BSP()
    bsp_mode = _normalize_bsp_mode(bsp_mode)
    if bsp_mode == "cnf2" and not _cnf2_enabled():
        # Keep intrinsic as the m1 evaluator until the m2 gate is active.
        raise ValueError("bsp_mode='cnf2' disabled until m2")
    if bsp_mode not in ("intrinsic", "cnf2"):
        raise ValueError(f"Unknown bsp_mode={bsp_mode!r}")
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith("#"):
            continue
        tokens = re.findall(r"\(|\)|[a-z]+", inp)
        root_ptr = vm.parse(tokens)
        frontier = jnp.array([root_ptr], dtype=jnp.int32)
        for _ in range(max(1, cycles)):
            if bsp_mode == "intrinsic":
                vm.ledger, frontier = cycle_intrinsic(vm.ledger, frontier)
            else:
                vm.ledger, frontier, _ = cycle_candidates(
                    vm.ledger, frontier, validate_stratum=validate_stratum
                )
            vm.ledger.count.block_until_ready()
            if bool(vm.ledger.corrupt):
                raise RuntimeError(
                    "CORRUPT: key encoding alias risk (id width exceeded)"
                )
            if bool(vm.ledger.oom):
                raise RuntimeError("Ledger capacity exceeded during cycle")
        root_ptr = frontier[0]
        root_ptr_int = int(root_ptr)
        print(f"   ├─ Ledger   : {int(vm.ledger.count)} nodes")
        print(f"   └─ Result  : \033[92m{vm.decode(root_ptr_int)}\033[0m")
    return vm

def repl(
    mode="baseline",
    use_morton=False,
    block_size=None,
    bsp_mode="auto",
    validate_stratum=False,
):
    if mode == "bsp":
        vm = PrismVM_BSP()
        bsp_mode = _normalize_bsp_mode(bsp_mode)
        mode_label = "CNF-2" if bsp_mode == "cnf2" else "Intrinsic"
        print(f"\n🔮 Prism IR Shell (BSP Ledger, {mode_label})")
        print("   Try: (add (suc zero) (suc zero))")
    else:
        vm = PrismVM()
        print("\n🔮 Prism IR Shell (Static Analysis + Deduplication)")
        print("   Try: (add (suc zero) (suc zero))")
        print("   Try: (add zero (suc (suc zero))) <- Triggers Optimizer")
    while True:
        try:
            inp = input("\nλ> ").strip()
            if inp == "exit": break
            if not inp: continue
            if mode == "bsp":
                run_program_lines_bsp(
                    [inp],
                    vm,
                    use_morton=use_morton,
                    block_size=block_size,
                    bsp_mode=bsp_mode,
                    validate_stratum=validate_stratum,
                )
            else:
                run_program_lines([inp], vm)
        except Exception as e:
            print(f"   ERROR: {e}")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    mode = "baseline"
    bsp_mode = "auto"
    validate_stratum = False
    cycles = 1
    do_sort = True
    use_morton = False
    block_size = None
    path = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--mode", "-m") and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
            continue
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--cycles" and i + 1 < len(args):
            cycles = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("--cycles="):
            cycles = int(arg.split("=", 1)[1])
            i += 1
            continue
        if arg == "--bsp-mode" and i + 1 < len(args):
            bsp_mode = args[i + 1]
            i += 2
            continue
        if arg.startswith("--bsp-mode="):
            bsp_mode = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--validate-stratum":
            validate_stratum = True
            i += 1
            continue
        if arg.startswith("--validate-stratum="):
            value = arg.split("=", 1)[1].strip().lower()
            validate_stratum = value in ("1", "true", "yes", "on")
            i += 1
            continue
        if arg == "--no-sort":
            do_sort = False
            i += 1
            continue
        if arg == "--morton":
            use_morton = True
            i += 1
            continue
        if arg == "--block-size" and i + 1 < len(args):
            block_size = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("--block-size="):
            block_size = int(arg.split("=", 1)[1])
            i += 1
            continue
        if path is None:
            path = arg
            i += 1
            continue
        i += 1
    if path:
        with open(path) as f:
            lines = f.readlines()
        if mode == "bsp":
            run_program_lines_bsp(
                lines,
                cycles=cycles,
                do_sort=do_sort,
                use_morton=use_morton,
                block_size=block_size,
                bsp_mode=bsp_mode,
                validate_stratum=validate_stratum,
            )
        else:
            run_program_lines(lines)
    else:
        repl(
            mode=mode,
            use_morton=use_morton,
            block_size=block_size,
            bsp_mode=bsp_mode,
            validate_stratum=validate_stratum,
        )
