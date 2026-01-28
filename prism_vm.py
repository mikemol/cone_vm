from jax import jit, lax
import jax
import jax.numpy as jnp
from typing import Dict, Callable, Tuple
import inspect
import os
import re
import time
from prism_core import jax_safe as _jax_safe
from prism_core.permutation import _invert_perm
from prism_ledger import intern as _ledger_intern
from prism_ledger.intern import _lookup_node_id
from prism_coord.coord import (
    cd_lift_binary,
    coord_norm,
    coord_norm_batch,
    coord_xor,
    coord_xor_batch,
)
from prism_vm_core.structures import (
    Arena,
    CandidateBuffer,
    Ledger,
    Manifest,
    NodeBatch,
    StagingContext,
    Stratum,
    hyperstrata_precedes,
    staging_context_forgets_detail,
)
from prism_vm_core.constants import (
    LEDGER_CAPACITY,
    MAX_COORD_STEPS,
    MAX_COUNT,
    MAX_ID,
    MAX_KEY_NODES,
    MAX_ROWS,
)
from prism_vm_core.graphs import (
    init_arena as _init_arena,
    init_ledger as _init_ledger,
    init_manifest as _init_manifest,
)
from prism_vm_core.candidates import _candidate_indices
from prism_vm_core.ledger_keys import _pack_key
from prism_vm_core.ontology import (
    OP_ADD,
    OP_COORD_ONE,
    OP_COORD_PAIR,
    OP_COORD_ZERO,
    OP_MUL,
    OP_NAMES,
    OP_NULL,
    OP_SORT,
    OP_SUC,
    OP_ZERO,
    ArenaPtr,
    CommittedIds,
    HostBool,
    HostInt,
    LedgerId,
    ManifestPtr,
    ProvisionalIds,
    ZERO_PTR,
)
from prism_vm_core.domains import (
    QMap,
    _arena_ptr,
    _committed_ids,
    _host_bool,
    _host_bool_value,
    _host_int,
    _host_int_value,
    _host_raise_if_bad,
    _ledger_id,
    _manifest_ptr,
    _provisional_ids,
    _require_arena_ptr,
    _require_ledger_id,
    _require_manifest_ptr,
)
from prism_vm_core.gating import (
    _cnf2_enabled,
    _cnf2_slot1_enabled,
    _default_bsp_mode,
    _gpu_metrics_device_index,
    _gpu_metrics_enabled,
    _normalize_bsp_mode,
    _parse_milestone_value,
    _read_pytest_milestone,
    _servo_enabled,
)
from prism_vm_core.guards import (
    _expect_token,
    _pop_token,
)
from prism_metrics.gpu import GPUWatchdog, _gpu_watchdog_create
from prism_metrics.metrics import (
    _damage_metrics_enabled,
    _damage_metrics_update,
    _damage_tile_size,
    cnf2_metrics_get,
    cnf2_metrics_reset,
    damage_metrics_get,
    damage_metrics_reset,
)
from prism_metrics.probes import coord_norm_probe_get, coord_norm_probe_reset
from prism_vm_core.hashes import _ledger_root_hash_host, _ledger_roots_hash_host
from prism_bsp.arena_step import cycle, op_interact
from prism_bsp.cnf2 import (
    compact_candidates,
    compact_candidates_with_index,
    cycle_candidates as _cycle_candidates_impl,
    emit_candidates,
    intern_candidates,
    _scatter_compacted_ids,
)
from prism_bsp.intrinsic import cycle_intrinsic
from prism_baseline.kernels import (
    dispatch_kernel,
    kernel_add,
    kernel_mul,
    optimize_ptr,
)
from prism_bsp.space import (
    RANK_COLD,
    RANK_FREE,
    RANK_HOT,
    RANK_WARM,
    _blocked_perm,
    op_morton,
    op_rank,
    op_sort_and_swizzle,
    op_sort_and_swizzle_blocked,
    op_sort_and_swizzle_blocked_with_perm,
    op_sort_and_swizzle_hierarchical,
    op_sort_and_swizzle_hierarchical_with_perm,
    op_sort_and_swizzle_morton,
    op_sort_and_swizzle_morton_with_perm,
    op_sort_and_swizzle_servo,
    op_sort_and_swizzle_servo_with_perm,
    op_sort_and_swizzle_with_perm,
    swizzle_2to1,
    swizzle_2to1_dev,
    swizzle_2to1_host,
)
from prism_semantics.commit import (
    _identity_q,
    apply_q,
    commit_stratum as _commit_stratum,
    validate_stratum_no_future_refs,
    validate_stratum_no_future_refs_jax,
    validate_stratum_no_within_refs,
    validate_stratum_no_within_refs_jax,
)
from prism_semantics.project import project_arena_to_ledger, project_manifest_to_ledger
import numpy as np

# NOTE: JAX op dtype normalization (int32) is assumed; tighten if drift appears
# (see IMPLEMENTATION_PLAN.md).
_TEST_GUARDS = _jax_safe.TEST_GUARDS
# Test-time guards favor correctness over performance (m1 gate).
# See IMPLEMENTATION_PLAN.md (m1 acceptance gate).
_SCATTER_GUARD = _jax_safe.SCATTER_GUARD
_GATHER_GUARD = _jax_safe.GATHER_GUARD
_HAS_DEBUG_CALLBACK = _jax_safe.HAS_DEBUG_CALLBACK
_scatter_guard = _jax_safe.scatter_guard
_scatter_guard_strict = _jax_safe.scatter_guard_strict


def safe_gather_1d(arr, idx, label="safe_gather_1d"):
    return _jax_safe.safe_gather_1d(arr, idx, label, guard=_GATHER_GUARD)


_OP_BUCKETS_FULL_RANGE = os.environ.get(
    "PRISM_OP_BUCKETS_FULL_RANGE", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_FORCE_SPAWN_CLIP = os.environ.get(
    "PRISM_FORCE_SPAWN_CLIP", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# --- 2. Manifest (Heap) ---


def node_batch(op, a1, a2) -> NodeBatch:
    if _TEST_GUARDS:
        if op.shape != a1.shape or op.shape != a2.shape:
            raise ValueError("node_batch expects aligned 1d arrays")
        if op.ndim != 1 or a1.ndim != 1 or a2.ndim != 1:
            raise ValueError("node_batch expects aligned 1d arrays")
    return NodeBatch(op=op, a1=a1, a2=a2)

def init_manifest():
    return _init_manifest()

def init_arena():
    return _init_arena(LEDGER_CAPACITY, RANK_FREE, OP_ZERO)

def init_ledger():
    return _init_ledger(_pack_key, LEDGER_CAPACITY, OP_NULL, OP_ZERO)


def ledger_has_corrupt(ledger) -> HostBool:
    # Host helper for deterministic corrupt checks in tests/debug.
    flag = ledger.corrupt if hasattr(ledger, "corrupt") else ledger.oom
    # SYNC: device_get forces host sync for deterministic checks (m1).
    return _host_bool(jax.device_get(flag))


def commit_stratum(
    ledger,
    stratum,
    prior_q: QMap | None = None,
    validate: bool = False,
    validate_mode: str = "strict",
    intern_fn=None,
):
    if intern_fn is None:
        intern_fn = intern_nodes
    return _commit_stratum(
        ledger,
        stratum,
        prior_q=prior_q,
        validate=validate,
        validate_mode=validate_mode,
        intern_fn=intern_fn,
    )


def cycle_candidates(
    ledger,
    frontier_ids: CommittedIds,
    validate_stratum: bool = False,
    validate_mode: str = "strict",
):
    if not _cnf2_enabled():
        raise RuntimeError("cycle_candidates disabled until m2 (set PRISM_ENABLE_CNF2=1)")
    ledger, frontier_ids, strata, q_map = _cycle_candidates_impl(
        ledger,
        frontier_ids,
        validate_stratum=validate_stratum,
        validate_mode=validate_mode,
        intern_fn=intern_nodes,
        cnf2_enabled_fn=_cnf2_enabled,
        cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    )
    if not _host_bool_value(ledger.corrupt):
        _host_raise_if_bad(ledger, "Ledger capacity exceeded during cycle")
    return ledger, frontier_ids, strata, q_map


__all__ = [
    "OP_NULL",
    "OP_ZERO",
    "OP_SUC",
    "OP_ADD",
    "OP_MUL",
    "OP_SORT",
    "OP_COORD_ZERO",
    "OP_COORD_ONE",
    "OP_COORD_PAIR",
    "ZERO_PTR",
    "OP_NAMES",
    "ManifestPtr",
    "LedgerId",
    "ArenaPtr",
    "HostInt",
    "HostBool",
    "ProvisionalIds",
    "CommittedIds",
    "QMap",
    "MAX_ROWS",
    "MAX_KEY_NODES",
    "LEDGER_CAPACITY",
    "MAX_ID",
    "MAX_COUNT",
    "MAX_COORD_STEPS",
    "safe_gather_1d",
    "coord_norm_probe_reset",
    "coord_norm_probe_get",
    "damage_metrics_reset",
    "cnf2_metrics_reset",
    "damage_metrics_get",
    "cnf2_metrics_get",
    "GPUWatchdog",
    "RANK_HOT",
    "RANK_WARM",
    "RANK_COLD",
    "RANK_FREE",
    "Manifest",
    "Arena",
    "Ledger",
    "CandidateBuffer",
    "NodeBatch",
    "_candidate_indices",
    "_scatter_compacted_ids",
    "Stratum",
    "StagingContext",
    "hyperstrata_precedes",
    "staging_context_forgets_detail",
    "node_batch",
    "init_manifest",
    "init_arena",
    "init_ledger",
    "ledger_has_corrupt",
    "project_manifest_to_ledger",
    "project_arena_to_ledger",
    "_ledger_root_hash_host",
    "_ledger_roots_hash_host",
    "emit_candidates",
    "compact_candidates",
    "compact_candidates_with_index",
    "intern_candidates",
    "validate_stratum_no_within_refs_jax",
    "validate_stratum_no_within_refs",
    "validate_stratum_no_future_refs_jax",
    "validate_stratum_no_future_refs",
    "apply_q",
    "commit_stratum",
    "cycle_candidates",
    "op_rank",
    "coord_norm",
    "coord_xor",
    "cd_lift_binary",
    "coord_norm_batch",
    "coord_xor_batch",
    "intern_nodes",
    "op_sort_and_swizzle_with_perm",
    "op_sort_and_swizzle",
    "op_sort_and_swizzle_blocked_with_perm",
    "op_sort_and_swizzle_blocked",
    "_blocked_perm",
    "op_sort_and_swizzle_hierarchical_with_perm",
    "op_sort_and_swizzle_hierarchical",
    "swizzle_2to1_host",
    "swizzle_2to1_dev",
    "swizzle_2to1",
    "op_morton",
    "op_sort_and_swizzle_morton_with_perm",
    "op_sort_and_swizzle_morton",
    "op_sort_and_swizzle_servo_with_perm",
    "op_sort_and_swizzle_servo",
    "op_interact",
    "cycle_intrinsic",
    "cycle",
    "kernel_add",
    "kernel_mul",
    "optimize_ptr",
    "dispatch_kernel",
    "PrismVM",
    "PrismVM_BSP_Legacy",
    "PrismVM_BSP",
    "make_vm",
    "run_program_lines",
    "run_program_lines_arena",
    "run_program_lines_bsp",
    "repl",
]




def _key_order_commutative_host(op, a1, a2):
    if op in (OP_ADD, OP_MUL) and a2 < a1:
        return a2, a1
    return a1, a2

@jit
def intern_nodes(ledger, batch_or_ops, a1=None, a2=None):
    _ledger_intern.OP_BUCKETS_FULL_RANGE = _OP_BUCKETS_FULL_RANGE
    _ledger_intern.FORCE_SPAWN_CLIP = _FORCE_SPAWN_CLIP
    return _ledger_intern.intern_nodes(ledger, batch_or_ops, a1, a2)


# --- 4. Prism VM (Host Logic) ---
class PrismVM:
    def __init__(self):
        print("⚡ Prism IR: Initializing Host Context...")
        # Baseline oracle (Manifest + host cache) kept for cross-engine checks.
        self.manifest = init_manifest()
        # SYNC: host reads device scalar for manifest count (m1).
        self.active_count_host = _host_int_value(self.manifest.active_count)
        self.refresh_cache_on_eval = True
        # Trace Cache: (opcode, arg1, arg2) -> ptr
        self.trace_cache: Dict[Tuple[int, ManifestPtr, ManifestPtr], ManifestPtr] = {}
        self.canonical_ptrs: Dict[ManifestPtr, ManifestPtr] = {
            _manifest_ptr(0): _manifest_ptr(0)
        }
        # Initialize Universe (Seed with ZERO)
        zero_ptr = self._cons_raw(OP_ZERO, _manifest_ptr(0), _manifest_ptr(0))
        self.trace_cache[(OP_ZERO, _manifest_ptr(0), _manifest_ptr(0))] = zero_ptr
        self.canonical_ptrs[zero_ptr] = zero_ptr
        self.cache_filled_to = self.active_count_host

        self.kernels = {OP_ADD: kernel_add, OP_MUL: kernel_mul}

    def _cons_raw(self, op: int, a1: ManifestPtr, a2: ManifestPtr) -> ManifestPtr:
        """Physical allocation (Device Write)"""
        _require_manifest_ptr(a1, "PrismVM._cons_raw a1")
        _require_manifest_ptr(a2, "PrismVM._cons_raw a2")
        a1_i, a2_i = _key_order_commutative_host(op, int(a1), int(a2))
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
            arg1=self.manifest.arg1.at[idx].set(a1_i),
            arg2=self.manifest.arg2.at[idx].set(a2_i),
            active_count=jnp.array(self.active_count_host, dtype=jnp.int32)
        )
        return _manifest_ptr(idx)

    def _refresh_trace_cache(self, start_idx: int, end_idx: int) -> None:
        if end_idx <= start_idx:
            return
        # SYNC: device_get pulls device buffers for host cache refresh (m1).
        # NOTE: trace_cache is a hint-only memo; avoid pointer rewrites.
        ops = jax.device_get(self.manifest.opcode[start_idx:end_idx])
        a1s = jax.device_get(self.manifest.arg1[start_idx:end_idx])
        a2s = jax.device_get(self.manifest.arg2[start_idx:end_idx])
        for offset, (op, a1, a2) in enumerate(zip(ops, a1s, a2s)):
            ptr = _manifest_ptr(start_idx + offset)
            op_i = int(op)
            a1_i = int(self._canonical_ptr(_manifest_ptr(a1)))
            a2_i = int(self._canonical_ptr(_manifest_ptr(a2)))
            a1_i, a2_i = _key_order_commutative_host(op_i, a1_i, a2_i)
            signature = (op_i, _manifest_ptr(a1_i), _manifest_ptr(a2_i))
            if signature not in self.trace_cache:
                self.trace_cache[signature] = ptr
            self.canonical_ptrs[ptr] = ptr

    def _canonical_ptr(self, ptr: ManifestPtr) -> ManifestPtr:
        return self.canonical_ptrs.get(ptr, ptr)

    def cons(
        self,
        op: int,
        a1: ManifestPtr = ManifestPtr(0),
        a2: ManifestPtr = ManifestPtr(0),
    ) -> ManifestPtr:
        """
        The Smart Allocator.
        1. Checks Cache (Deduplication).
        2. Allocates if new.
        """
        _require_manifest_ptr(a1, "PrismVM.cons a1")
        _require_manifest_ptr(a2, "PrismVM.cons a2")
        a1_i = int(self._canonical_ptr(a1))
        a2_i = int(self._canonical_ptr(a2))
        a1_i, a2_i = _key_order_commutative_host(op, a1_i, a2_i)
        signature = (op, _manifest_ptr(a1_i), _manifest_ptr(a2_i))
        if signature in self.trace_cache:
            return self.trace_cache[signature]
        ptr = self._cons_raw(op, _manifest_ptr(a1_i), _manifest_ptr(a2_i))
        self.trace_cache[signature] = ptr
        self.canonical_ptrs[ptr] = ptr
        return ptr

    # --- STATIC ANALYSIS ENGINE ---
    def analyze_and_optimize(self, ptr: ManifestPtr) -> ManifestPtr:
        """
        Examines the IR at `ptr` BEFORE execution.
        Performs trivial reductions (Constant Folding / Identity Elimination).
        """
        _require_manifest_ptr(ptr, "PrismVM.analyze_and_optimize ptr")
        ptr = self._canonical_ptr(ptr)
        ptr_arr = jnp.array(int(ptr), dtype=jnp.int32)
        opt_ptr, opt_reason = optimize_ptr(self.manifest, ptr_arr)
        # SYNC: host reads device flag for optimization signal (m1).
        opt_reason_i = _host_int_value(opt_reason)
        if opt_reason_i == 1:
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        elif opt_reason_i == 2:
            print("   [!] Static Analysis: Optimized (mul zero x) -> zero")
        # SYNC: host reads device scalar for optimized ptr (m1).
        return self._canonical_ptr(_manifest_ptr(_host_int_value(opt_ptr)))

    def eval(self, ptr: ManifestPtr) -> ManifestPtr:
        """
        The Hybrid Interpreter.
        1. Static Analysis (Host)
        2. Dispatch (Device)
        """
        _require_manifest_ptr(ptr, "PrismVM.eval ptr")
        ptr = self._canonical_ptr(ptr)
        ptr_arr = jnp.array(int(ptr), dtype=jnp.int32)
        prev_count = self.active_count_host
        new_manifest, res_ptr, opt_reason = dispatch_kernel(self.manifest, ptr_arr)
        # SYNC: wait for device result before host state update (m1).
        res_ptr.block_until_ready()
        self.manifest = new_manifest
        # SYNC: host reads device scalar for manifest count (m1).
        self.active_count_host = _host_int_value(self.manifest.active_count)
        if self.refresh_cache_on_eval and self.active_count_host > prev_count:
            self._refresh_trace_cache(prev_count, self.active_count_host)
            self.cache_filled_to = self.active_count_host
        # SYNC: host reads device flags for error/opt reporting (m1).
        if _host_bool_value(self.manifest.oom):
            raise RuntimeError("Manifest capacity exceeded during kernel execution")
        opt_reason_i = _host_int_value(opt_reason)
        if opt_reason_i == 1:
            print("   [!] Static Analysis: Optimized (add zero x) -> x")
        elif opt_reason_i == 2:
            print("   [!] Static Analysis: Optimized (mul zero x) -> zero")
        # SYNC: host reads device scalar for result ptr (m1).
        return self._canonical_ptr(_manifest_ptr(_host_int_value(res_ptr)))

    # --- PARSING & DISPLAY ---
    def parse(self, tokens) -> ManifestPtr:
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

    def decode(self, ptr: ManifestPtr) -> str:
        # SYNC: host reads device opcode/args for decoding (m1).
        _require_manifest_ptr(ptr, "PrismVM.decode ptr")
        ptr_i = int(ptr)
        op = _host_int_value(self.manifest.opcode[ptr_i])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:
            return f"(suc {self.decode(_manifest_ptr(self.manifest.arg1[ptr_i]))})"
        if op == OP_ADD:
            return (
                f"(add {self.decode(_manifest_ptr(self.manifest.arg1[ptr_i]))} "
                f"{self.decode(_manifest_ptr(self.manifest.arg2[ptr_i]))})"
            )
        if op == OP_MUL:
            return (
                f"(mul {self.decode(_manifest_ptr(self.manifest.arg1[ptr_i]))} "
                f"{self.decode(_manifest_ptr(self.manifest.arg2[ptr_i]))})"
            )
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


class PrismVM_BSP_Legacy:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Arena (Legacy)...")
        self.arena = init_arena()

    def _alloc(
        self, op: int, a1: ArenaPtr = ArenaPtr(0), a2: ArenaPtr = ArenaPtr(0)
    ) -> ArenaPtr:
        cap = int(self.arena.opcode.shape[0])
        # SYNC: host reads device scalar for arena count (m1).
        idx = _host_int_value(self.arena.count)
        if idx >= cap:
            self.arena = self.arena._replace(oom=jnp.array(True, dtype=jnp.bool_))
            raise ValueError("Arena capacity exceeded")
        _require_arena_ptr(a1, "PrismVM_BSP_Legacy._alloc a1")
        _require_arena_ptr(a2, "PrismVM_BSP_Legacy._alloc a2")
        a1_i, a2_i = _key_order_commutative_host(op, int(a1), int(a2))
        self.arena = self.arena._replace(
            opcode=self.arena.opcode.at[idx].set(op),
            arg1=self.arena.arg1.at[idx].set(a1_i),
            arg2=self.arena.arg2.at[idx].set(a2_i),
            count=jnp.array(idx + 1, dtype=jnp.int32),
        )
        return _arena_ptr(idx)

    def parse(self, tokens) -> ArenaPtr:
        token = _pop_token(tokens)
        if token == "zero": return self._alloc(OP_ZERO)
        if token == "suc":  return self._alloc(OP_SUC, self.parse(tokens))
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

    def decode(self, ptr: ArenaPtr, show_ids: bool = False) -> str:
        # SYNC: host reads device opcode/args for decoding (m1).
        _require_arena_ptr(ptr, "PrismVM_BSP_Legacy.decode ptr")
        ptr_i = int(ptr)
        op = _host_int_value(self.arena.opcode[ptr_i])
        if op == OP_ZERO:
            return "zero"
        if op == OP_SUC:
            return (
                f"(suc "
                f"{self.decode(_arena_ptr(self.arena.arg1[ptr_i]), show_ids=show_ids)})"
            )
        name = OP_NAMES.get(op, "?")
        if show_ids:
            return f"<{name}:{ptr}>"
        return f"<{name}>"

class PrismVM_BSP:
    def __init__(self):
        print("⚡ Prism IR: Initializing BSP Ledger...")
        # Canonical Ledger path (univalence + deterministic interning).
        self.ledger = init_ledger()

    def _intern(
        self, op: int, a1: LedgerId = LedgerId(0), a2: LedgerId = LedgerId(0)
    ) -> LedgerId:
        _require_ledger_id(a1, "PrismVM_BSP._intern a1")
        _require_ledger_id(a2, "PrismVM_BSP._intern a2")
        a1_i, a2_i = _key_order_commutative_host(op, int(a1), int(a2))
        ids, self.ledger = intern_nodes(
            self.ledger,
            node_batch(
                jnp.array([op], dtype=jnp.int32),
                jnp.array([a1_i], dtype=jnp.int32),
                jnp.array([a2_i], dtype=jnp.int32),
            ),
        )
        # SYNC: host wait/flag checks for BSP interning (m1).
        _host_raise_if_bad(
            self.ledger,
            "Ledger capacity exceeded during interning",
            oom_exc=ValueError,
        )
        # SYNC: host reads device id for interned node (m1).
        return _ledger_id(ids[0])

    def parse(self, tokens) -> LedgerId:
        token = _pop_token(tokens)
        if token == "zero": return self._intern(OP_ZERO)
        if token == "suc":  return self._intern(OP_SUC, self.parse(tokens))
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

    def decode(self, ptr: LedgerId) -> str:
        # SYNC: host reads device opcode/args for decoding (m1).
        _require_ledger_id(ptr, "PrismVM_BSP.decode ptr")
        ptr_i = int(ptr)
        op = _host_int_value(self.ledger.opcode[ptr_i])
        if op == OP_ZERO: return "zero"
        if op == OP_SUC:
            return f"(suc {self.decode(_ledger_id(self.ledger.arg1[ptr_i]))})"
        if op == OP_ADD:
            return (
                f"(add {self.decode(_ledger_id(self.ledger.arg1[ptr_i]))} "
                f"{self.decode(_ledger_id(self.ledger.arg2[ptr_i]))})"
            )
        if op == OP_MUL:
            return (
                f"(mul {self.decode(_ledger_id(self.ledger.arg1[ptr_i]))} "
                f"{self.decode(_ledger_id(self.ledger.arg2[ptr_i]))})"
            )
        return f"<{OP_NAMES.get(op, '?')}:{ptr}>"


def make_vm(mode="baseline"):
    if mode == "arena":
        return PrismVM_BSP_Legacy()
    if mode == "bsp":
        return PrismVM_BSP()
    return PrismVM()

def _rank_counts(arena) -> Tuple[HostInt, HostInt, HostInt, HostInt]:
    # SYNC: host reads device counters for rank summary (m1).
    hot = _host_int(jnp.sum(arena.rank == RANK_HOT))
    warm = _host_int(jnp.sum(arena.rank == RANK_WARM))
    cold = _host_int(jnp.sum(arena.rank == RANK_COLD))
    free = _host_int(jnp.sum(arena.rank == RANK_FREE))
    return hot, warm, cold, free

# --- 5. Telemetric REPL ---
def run_program_lines(lines, vm=None):
    if vm is None:
        vm = PrismVM()
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith('#'):
            continue
        # SYNC: host reads of device counters for telemetry (m1).
        start_rows = _host_int_value(vm.manifest.active_count)
        t0 = time.perf_counter()
        tokens = re.findall(r'\(|\)|[a-z]+', inp)
        ir_ptr = vm.parse(tokens)
        ir_ptr_i = int(ir_ptr)
        parse_ms = (time.perf_counter() - t0) * 1000
        # SYNC: host reads device counters for telemetry (m1).
        mid_rows = _host_int_value(vm.manifest.active_count)
        ir_allocs = mid_rows - start_rows
        # SYNC: host reads device opcode for telemetry (m1).
        ir_op = OP_NAMES.get(_host_int_value(vm.manifest.opcode[ir_ptr_i]), "?")
        print(f"   ├─ IR Build: {ir_op} @ {ir_ptr}")
        if ir_allocs == 0:
            print(f"   ├─ Cache   : \033[96mHIT (No new IR rows)\033[0m")
        else:
            print(f"   ├─ Cache   : MISS (+{ir_allocs} IR rows)")
        t1 = time.perf_counter()
        res_ptr = vm.eval(ir_ptr)
        eval_ms = (time.perf_counter() - t1) * 1000
        # SYNC: host reads device counters for telemetry (m1).
        end_rows = _host_int_value(vm.manifest.active_count)
        exec_allocs = end_rows - mid_rows
        print(f"   ├─ Execute : {eval_ms:.2f}ms")
        if exec_allocs > 0:
            print(f"   ├─ Kernel  : +{exec_allocs} rows allocated")
        else:
            print(f"   ├─ Kernel  : \033[96mSKIPPED (Static Optimization)\033[0m")
        print(f"   └─ Result  : \033[92m{vm.decode(res_ptr)}\033[0m")
    return vm

def run_program_lines_arena(
    lines,
    vm=None,
    cycles=1,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
):
    if vm is None:
        vm = PrismVM_BSP_Legacy()
    tile_size = _damage_tile_size(block_size, l2_block_size, l1_block_size)
    watchdog = _gpu_watchdog_create()
    try:
        for inp in lines:
            inp = inp.strip()
            if not inp or inp.startswith("#"):
                continue
            if _damage_metrics_enabled():
                damage_metrics_reset()
            tokens = re.findall(r"\(|\)|[a-z]+", inp)
            root_ptr = vm.parse(tokens)
            for _ in range(max(1, cycles)):
                vm.arena, root_ptr = cycle(
                    vm.arena,
                    root_ptr,
                    do_sort=do_sort,
                    use_morton=use_morton,
                    block_size=block_size,
                    l2_block_size=l2_block_size,
                    l1_block_size=l1_block_size,
                    do_global=do_global,
                )
                root_ptr = _arena_ptr(_host_int_value(root_ptr))
                if _host_bool_value(vm.arena.oom):
                    raise RuntimeError("Arena capacity exceeded during cycle")
            print(f"   ├─ Arena    : {_host_int_value(vm.arena.count)} nodes")
            if _damage_metrics_enabled():
                metrics = damage_metrics_get()
                print(
                    "   ├─ Damage   : "
                    f"cycles={metrics['cycles']} "
                    f"hot={metrics['hot_nodes']} "
                    f"edges={metrics['edge_cross']}/{metrics['edge_total']} "
                    f"rate={metrics['damage_rate']:.4f} "
                    f"tile={int(tile_size)}"
                )
            if watchdog is not None:
                stats = watchdog.poll()
                if stats is not None:
                    power = (
                        f"{stats['power_w']:.1f}W"
                        if stats["power_w"] is not None
                        else "na"
                    )
                    clock = (
                        f"{stats['sm_clock']}MHz"
                        if stats["sm_clock"] is not None
                        else "na"
                    )
                    print(
                        "   ├─ GPU      : "
                        f"util={stats['gpu_util']}% "
                        f"memio={stats['mem_io']}% "
                        f"vram={stats['vram_used_mb']}/{stats['vram_total_mb']}MB "
                        f"power={power} "
                        f"clock={clock}"
                    )
            print(f"   └─ Result   : \033[92m{vm.decode(root_ptr)}\033[0m")
    finally:
        if watchdog is not None:
            watchdog.close()
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
        # See IMPLEMENTATION_PLAN.md (m1/m2 BSP gating).
        raise ValueError("bsp_mode='cnf2' disabled until m2")
    if bsp_mode not in ("intrinsic", "cnf2"):
        raise ValueError(f"Unknown bsp_mode={bsp_mode!r}")
    for inp in lines:
        inp = inp.strip()
        if not inp or inp.startswith("#"):
            continue
        tokens = re.findall(r"\(|\)|[a-z]+", inp)
        root_ptr = vm.parse(tokens)
        frontier = _committed_ids(jnp.array([int(root_ptr)], dtype=jnp.int32))
        for _ in range(max(1, cycles)):
            if bsp_mode == "intrinsic":
                vm.ledger, frontier_arr = cycle_intrinsic(vm.ledger, frontier.a)
                frontier = _committed_ids(frontier_arr)
            else:
                vm.ledger, frontier_prov, _, q_map = cycle_candidates(
                    vm.ledger, frontier, validate_stratum=validate_stratum
                )
                frontier = apply_q(q_map, frontier_prov)
            # SYNC: host wait/flag checks for BSP loop (m1).
            _host_raise_if_bad(vm.ledger, "Ledger capacity exceeded during cycle")
        root_ptr = frontier.a[0]
        # SYNC: host reads for reporting in BSP loop (m1).
        root_ptr_int = _host_int_value(root_ptr)
        # SYNC: host reads device counter for reporting (m1).
        print(f"   ├─ Ledger   : {_host_int_value(vm.ledger.count)} nodes")
        print(f"   └─ Result  : \033[92m{vm.decode(_ledger_id(root_ptr_int))}\033[0m")
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
    elif mode == "arena":
        vm = PrismVM_BSP_Legacy()
        print("\n🔮 Prism IR Shell (Arena BSP, Legacy)")
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
            elif mode == "arena":
                run_program_lines_arena(
                    [inp],
                    vm,
                    use_morton=use_morton,
                    block_size=block_size,
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
        elif mode == "arena":
            run_program_lines_arena(
                lines,
                cycles=cycles,
                do_sort=do_sort,
                use_morton=use_morton,
                block_size=block_size,
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
