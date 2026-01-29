from __future__ import annotations

"""Facade wrappers with explicit DI and glossary contracts.

Axis: Interface/Control (host-visible); wrappers must commute with q and be
erased by q. This module centralizes wrapper behavior to avoid accidental
shadowing or monkeypatching drift.
"""

from typing import Optional
from dataclasses import replace

import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_core.di import wrap_policy, wrap_index_policy
from prism_core.safety import SafetyPolicy, DEFAULT_SAFETY_POLICY, oob_mask
from prism_ledger import intern as _ledger_intern
from prism_ledger.config import InternConfig, DEFAULT_INTERN_CONFIG
from prism_bsp.config import (
    Cnf2Config,
    Cnf2Flags,
    DEFAULT_CNF2_CONFIG,
    DEFAULT_CNF2_FLAGS,
    ArenaInteractConfig,
    DEFAULT_ARENA_INTERACT_CONFIG,
    ArenaCycleConfig,
    DEFAULT_ARENA_CYCLE_CONFIG,
    IntrinsicConfig,
    DEFAULT_INTRINSIC_CONFIG,
)
from prism_coord.config import CoordConfig, DEFAULT_COORD_CONFIG
from prism_vm_core.protocols import EmitCandidatesFn, HostRaiseFn, InternFn

from prism_vm_core.constants import LEDGER_CAPACITY
from prism_vm_core.graphs import (
    init_arena as _init_arena,
    init_ledger as _init_ledger,
    init_manifest as _init_manifest,
)
from prism_vm_core.ledger_keys import _pack_key
from prism_core.permutation import _invert_perm
from prism_ledger.intern import _lookup_node_id
from prism_vm_core.hashes import _ledger_root_hash_host, _ledger_roots_hash_host
from prism_vm_core.candidates import _candidate_indices, candidate_indices_cfg
from prism_bsp.cnf2 import _scatter_compacted_ids
from prism_vm_core.ontology import OP_ADD, OP_MUL, OP_NULL, OP_ZERO, HostBool
from prism_vm_core.domains import _host_bool, _host_raise_if_bad
from prism_vm_core.structures import NodeBatch
from prism_bsp.space import RANK_FREE
from prism_bsp.cnf2 import (
    emit_candidates as _emit_candidates_default,
    compact_candidates as _compact_candidates,
    compact_candidates_with_index as _compact_candidates_with_index,
    intern_candidates as _intern_candidates,
    emit_candidates_cfg as _emit_candidates_cfg,
    compact_candidates_cfg as _compact_candidates_cfg,
    compact_candidates_with_index_cfg as _compact_candidates_with_index_cfg,
    scatter_compacted_ids_cfg as _scatter_compacted_ids_cfg,
    intern_candidates_cfg as _intern_candidates_cfg,
    emit_candidates as _emit_candidates,
)
from prism_bsp.arena_step import (
    cycle as _cycle,
    cycle_core as _cycle_core,
    op_interact as _op_interact,
    op_interact_cfg,
    cycle_cfg,
)
from prism_bsp.intrinsic import cycle_intrinsic as _cycle_intrinsic, cycle_intrinsic_cfg
from prism_bsp.space import (
    RANK_COLD,
    RANK_FREE as _RANK_FREE_EXPORT,
    RANK_HOT,
    RANK_WARM,
    _blind_spectral_probe,
    _servo_update,
    _servo_mask_from_k,
    _servo_mask_to_k,
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
from prism_coord.coord import (
    _coord_leaf_id,
    _coord_promote_leaf,
    cd_lift_binary,
    cd_lift_binary_cfg,
    coord_norm,
    coord_norm_batch,
    coord_xor,
    coord_xor_cfg,
    coord_xor_batch,
    coord_xor_batch_cfg,
)
from prism_semantics.commit import (
    apply_q,
    validate_stratum_no_future_refs,
    validate_stratum_no_future_refs_jax,
    validate_stratum_no_within_refs,
    validate_stratum_no_within_refs_jax,
)
from prism_semantics.project import project_arena_to_ledger, project_manifest_to_ledger
from prism_baseline.kernels import dispatch_kernel, kernel_add, kernel_mul, optimize_ptr
from prism_metrics.probes import coord_norm_probe_get, coord_norm_probe_reset
from prism_metrics.metrics import (
    _damage_metrics_enabled,
    _damage_metrics_update,
    _damage_tile_size,
    cnf2_metrics_get,
    cnf2_metrics_reset,
    damage_metrics_get,
    damage_metrics_reset,
)
from prism_metrics.gpu import GPUWatchdog, _gpu_watchdog_create


def arena_interact_config_with_policy(
    safety_policy: SafetyPolicy | None,
    *,
    cfg: ArenaInteractConfig = DEFAULT_ARENA_INTERACT_CONFIG,
) -> ArenaInteractConfig:
    """Return an ArenaInteractConfig with safety_policy set."""
    return replace(cfg, safe_gather_policy=safety_policy)


def arena_cycle_config_with_policy(
    safety_policy: SafetyPolicy | None,
    *,
    cfg: ArenaCycleConfig = DEFAULT_ARENA_CYCLE_CONFIG,
    include_interact: bool = True,
) -> ArenaCycleConfig:
    """Return an ArenaCycleConfig with safety_policy set (and optionally its interact_cfg)."""
    interact_cfg = cfg.interact_cfg
    if include_interact:
        if interact_cfg is None:
            interact_cfg = DEFAULT_ARENA_INTERACT_CONFIG
        interact_cfg = replace(interact_cfg, safe_gather_policy=safety_policy)
    return replace(cfg, safe_gather_policy=safety_policy, interact_cfg=interact_cfg)


def cnf2_config_with_policy(
    safety_policy: SafetyPolicy | None,
    *,
    cfg: Cnf2Config = DEFAULT_CNF2_CONFIG,
) -> Cnf2Config:
    """Return a Cnf2Config with safe_gather_policy set."""
    return replace(cfg, safe_gather_policy=safety_policy)
from prism_vm_core.gating import (
    _cnf2_enabled as _cnf2_enabled_default,
    _cnf2_slot1_enabled as _cnf2_slot1_enabled_default,
    _default_bsp_mode,
    _gpu_metrics_device_index,
    _gpu_metrics_enabled,
    _normalize_bsp_mode,
    _parse_milestone_value,
    _read_pytest_milestone,
    _servo_enabled,
)
from prism_vm_core.guards import (
    GuardConfig,
    DEFAULT_GUARD_CONFIG,
    guards_enabled_cfg,
    guard_max_cfg,
    guard_gather_index_cfg,
    guard_slot0_perm_cfg,
    guard_null_row_cfg,
    guard_zero_row_cfg,
    guard_zero_args_cfg,
    guard_swizzle_args_cfg,
)
from prism_semantics.commit import commit_stratum as _commit_stratum_impl
from prism_bsp.cnf2 import cycle_candidates as _cycle_candidates_impl
from prism_vm_core.jit_entrypoints import (
    coord_norm_batch_jit,
    cycle_candidates_jit,
    cycle_intrinsic_jit,
    cycle_intrinsic_jit_cfg,
    cycle_jit as _cycle_jit_factory,
    cycle_jit_cfg,
    emit_candidates_jit,
    emit_candidates_jit_cfg,
    compact_candidates_jit,
    compact_candidates_jit_cfg,
    compact_candidates_with_index_jit,
    compact_candidates_with_index_jit_cfg,
    intern_candidates_jit,
    intern_candidates_jit_cfg,
    intern_nodes_jit,
    op_interact_jit,
    op_interact_jit_cfg,
)


_TEST_GUARDS = _jax_safe.TEST_GUARDS
_GATHER_GUARD = _jax_safe.GATHER_GUARD
_SCATTER_GUARD = _jax_safe.SCATTER_GUARD
_HAS_DEBUG_CALLBACK = _jax_safe.HAS_DEBUG_CALLBACK
_scatter_guard = _jax_safe.scatter_guard
_scatter_guard_strict = _jax_safe.scatter_guard_strict
_scatter_drop = _jax_safe.scatter_drop
_scatter_strict = _jax_safe.scatter_strict
SafetyPolicy = SafetyPolicy
DEFAULT_SAFETY_POLICY = DEFAULT_SAFETY_POLICY
oob_mask = oob_mask
GuardConfig = GuardConfig
DEFAULT_GUARD_CONFIG = DEFAULT_GUARD_CONFIG
guards_enabled_cfg = guards_enabled_cfg
guard_max_cfg = guard_max_cfg
guard_gather_index_cfg = guard_gather_index_cfg
guard_slot0_perm_cfg = guard_slot0_perm_cfg
guard_null_row_cfg = guard_null_row_cfg
guard_zero_row_cfg = guard_zero_row_cfg
guard_zero_args_cfg = guard_zero_args_cfg
guard_swizzle_args_cfg = guard_swizzle_args_cfg

def safe_gather_1d(
    arr,
    idx,
    label="safe_gather_1d",
    *,
    guard=None,
    policy: SafetyPolicy | None = None,
    return_ok: bool = False,
):
    """Interface/Control wrapper for gather guards.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_gather_guard.py
    """
    if guard is None:
        guard = _GATHER_GUARD
    return _jax_safe.safe_gather_1d(
        arr, idx, label, guard=guard, policy=policy, return_ok=return_ok
    )


def safe_gather_1d_ok(
    arr,
    idx,
    label="safe_gather_1d_ok",
    *,
    guard=None,
    policy: SafetyPolicy | None = None,
):
    """Interface/Control wrapper for safe_gather_1d_ok.

    Axis: Interface/Control. Commutes with q. Erased by q.
    """
    if guard is None:
        guard = _GATHER_GUARD
    return _jax_safe.safe_gather_1d_ok(
        arr, idx, label, guard=guard, policy=policy
    )


def safe_index_1d(
    idx,
    size,
    label="safe_index_1d",
    *,
    guard=None,
    policy: SafetyPolicy | None = None,
):
    """Interface/Control wrapper for safe_index_1d.

    Axis: Interface/Control. Commutes with q. Erased by q.
    """
    if guard is None:
        guard = _GATHER_GUARD
    safe_index_fn = wrap_index_policy(_jax_safe.safe_index_1d, policy)
    return safe_index_fn(idx, size, label, guard=guard)


def safe_index_1d_cfg(
    idx,
    size,
    label="safe_index_1d",
    *,
    guard=None,
    policy: SafetyPolicy | None = None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    """Interface/Control wrapper for safe_index_1d with guard config."""
    guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)
    safe_index_fn = wrap_index_policy(_jax_safe.safe_index_1d, policy)
    return safe_index_fn(idx, size, label, guard=False)

def node_batch(op, a1, a2) -> NodeBatch:
    """Interface/Control wrapper for batch shape checks.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_invariants.py
    """
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
    """Interface/Control wrapper for host-visible corrupt checks.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_m1_gate.py
    """
    flag = ledger.corrupt if hasattr(ledger, "corrupt") else ledger.oom
    return _host_bool(jax.device_get(flag))


# Facade re-exports for tests that reach into prism_vm.
_pack_key = _pack_key
_invert_perm = _invert_perm
_lookup_node_id = _lookup_node_id
_ledger_root_hash_host = _ledger_root_hash_host
_ledger_roots_hash_host = _ledger_roots_hash_host
_candidate_indices = _candidate_indices
candidate_indices_cfg = candidate_indices_cfg
_scatter_compacted_ids = _scatter_compacted_ids
scatter_compacted_ids_cfg = _scatter_compacted_ids_cfg
_cnf2_enabled = _cnf2_enabled_default
_cnf2_slot1_enabled = _cnf2_slot1_enabled_default
emit_candidates = _emit_candidates
emit_candidates_cfg = _emit_candidates_cfg
compact_candidates = _compact_candidates
compact_candidates_cfg = _compact_candidates_cfg
compact_candidates_with_index = _compact_candidates_with_index
compact_candidates_with_index_cfg = _compact_candidates_with_index_cfg
intern_candidates = _intern_candidates
intern_candidates_cfg = _intern_candidates_cfg
cycle = _cycle
cycle_core = _cycle_core
op_interact = _op_interact
cycle_intrinsic = _cycle_intrinsic
RANK_COLD = RANK_COLD
RANK_FREE = _RANK_FREE_EXPORT
RANK_HOT = RANK_HOT
RANK_WARM = RANK_WARM
_blocked_perm = _blocked_perm
op_morton = op_morton
op_rank = op_rank
op_sort_and_swizzle = op_sort_and_swizzle
op_sort_and_swizzle_blocked = op_sort_and_swizzle_blocked
op_sort_and_swizzle_blocked_with_perm = op_sort_and_swizzle_blocked_with_perm
op_sort_and_swizzle_hierarchical = op_sort_and_swizzle_hierarchical
op_sort_and_swizzle_hierarchical_with_perm = op_sort_and_swizzle_hierarchical_with_perm
op_sort_and_swizzle_morton = op_sort_and_swizzle_morton
op_sort_and_swizzle_morton_with_perm = op_sort_and_swizzle_morton_with_perm
op_sort_and_swizzle_servo = op_sort_and_swizzle_servo
op_sort_and_swizzle_servo_with_perm = op_sort_and_swizzle_servo_with_perm
op_sort_and_swizzle_with_perm = op_sort_and_swizzle_with_perm
swizzle_2to1 = swizzle_2to1
swizzle_2to1_dev = swizzle_2to1_dev
swizzle_2to1_host = swizzle_2to1_host
cd_lift_binary = cd_lift_binary
coord_norm = coord_norm
coord_norm_batch = coord_norm_batch
coord_xor = coord_xor
coord_xor_batch = coord_xor_batch
apply_q = apply_q
validate_stratum_no_future_refs = validate_stratum_no_future_refs
validate_stratum_no_future_refs_jax = validate_stratum_no_future_refs_jax
validate_stratum_no_within_refs = validate_stratum_no_within_refs
validate_stratum_no_within_refs_jax = validate_stratum_no_within_refs_jax
project_arena_to_ledger = project_arena_to_ledger
project_manifest_to_ledger = project_manifest_to_ledger
dispatch_kernel = dispatch_kernel
kernel_add = kernel_add
kernel_mul = kernel_mul
optimize_ptr = optimize_ptr
coord_norm_probe_get = coord_norm_probe_get
coord_norm_probe_reset = coord_norm_probe_reset
_damage_metrics_enabled = _damage_metrics_enabled
_damage_metrics_update = _damage_metrics_update
_damage_tile_size = _damage_tile_size
damage_metrics_get = damage_metrics_get
damage_metrics_reset = damage_metrics_reset
cnf2_metrics_get = cnf2_metrics_get
cnf2_metrics_reset = cnf2_metrics_reset
GPUWatchdog = GPUWatchdog
_gpu_watchdog_create = _gpu_watchdog_create
InternConfig = InternConfig
DEFAULT_INTERN_CONFIG = DEFAULT_INTERN_CONFIG
Cnf2Config = Cnf2Config
Cnf2Flags = Cnf2Flags
DEFAULT_CNF2_CONFIG = DEFAULT_CNF2_CONFIG
DEFAULT_CNF2_FLAGS = DEFAULT_CNF2_FLAGS
CoordConfig = CoordConfig
DEFAULT_COORD_CONFIG = DEFAULT_COORD_CONFIG
dispatch_kernel = dispatch_kernel
kernel_add = kernel_add
kernel_mul = kernel_mul
optimize_ptr = optimize_ptr


def _key_order_commutative_host(op, a1, a2):
    """Host helper for commutative op argument normalization."""
    if op in (OP_ADD, OP_MUL) and a2 < a1:
        return a2, a1
    return a1, a2


def intern_nodes(
    ledger,
    batch_or_ops,
    a1=None,
    a2=None,
    *,
    cfg: InternConfig | None = None,
    op_buckets_full_range: Optional[bool] = None,
    force_spawn_clip: Optional[bool] = None,
):
    """Interface/Control wrapper for intern_nodes behavior flags.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_m1_gate.py
    """
    if cfg is not None:
        if op_buckets_full_range is not None or force_spawn_clip is not None:
            raise ValueError("Pass either cfg or individual flags, not both.")
    if cfg is None and op_buckets_full_range is None and force_spawn_clip is None:
        cfg = DEFAULT_INTERN_CONFIG
    elif cfg is None:
        cfg = InternConfig(
            op_buckets_full_range=bool(op_buckets_full_range or False),
            force_spawn_clip=bool(force_spawn_clip or False),
        )
    if a1 is None and a2 is None:
        if not isinstance(batch_or_ops, NodeBatch):
            raise TypeError("intern_nodes expects a NodeBatch or (ops, a1, a2)")
        batch = batch_or_ops
    else:
        if a1 is None or a2 is None:
            raise TypeError("intern_nodes expects both a1 and a2 arrays")
        batch = NodeBatch(batch_or_ops, a1, a2)
    return intern_nodes_jit(cfg)(ledger, batch)


def cycle_jit(
    *,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    op_sort_and_swizzle_with_perm_fn=op_sort_and_swizzle_with_perm,
    op_sort_and_swizzle_morton_with_perm_fn=op_sort_and_swizzle_morton_with_perm,
    op_sort_and_swizzle_blocked_with_perm_fn=op_sort_and_swizzle_blocked_with_perm,
    op_sort_and_swizzle_hierarchical_with_perm_fn=op_sort_and_swizzle_hierarchical_with_perm,
    op_sort_and_swizzle_servo_with_perm_fn=op_sort_and_swizzle_servo_with_perm,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    op_interact_fn=_op_interact,
):
    """Return a jitted cycle entrypoint for fixed DI."""
    return _cycle_jit_factory(
        do_sort=do_sort,
        use_morton=use_morton,
        block_size=block_size,
        l2_block_size=l2_block_size,
        l1_block_size=l1_block_size,
        do_global=do_global,
        op_rank_fn=op_rank_fn,
        servo_enabled_fn=servo_enabled_fn,
        servo_update_fn=servo_update_fn,
        op_morton_fn=op_morton_fn,
        op_sort_and_swizzle_with_perm_fn=op_sort_and_swizzle_with_perm_fn,
        op_sort_and_swizzle_morton_with_perm_fn=op_sort_and_swizzle_morton_with_perm_fn,
        op_sort_and_swizzle_blocked_with_perm_fn=op_sort_and_swizzle_blocked_with_perm_fn,
        op_sort_and_swizzle_hierarchical_with_perm_fn=op_sort_and_swizzle_hierarchical_with_perm_fn,
        op_sort_and_swizzle_servo_with_perm_fn=op_sort_and_swizzle_servo_with_perm_fn,
        safe_gather_fn=safe_gather_fn,
        op_interact_fn=op_interact_fn,
        test_guards=_TEST_GUARDS,
    )


def commit_stratum(
    ledger,
    stratum,
    prior_q=None,
    validate: bool = False,
    validate_mode: str = "strict",
    intern_fn: InternFn | None = None,
    *,
    safe_gather_policy: SafetyPolicy | None = None,
):
    """Interface/Control wrapper for commit_stratum injection.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_commit_stratum.py
    """
    if intern_fn is None:
        intern_fn = intern_nodes
    safe_gather_fn = wrap_policy(safe_gather_1d, safe_gather_policy)
    return _commit_stratum_impl(
        ledger,
        stratum,
        prior_q=prior_q,
        validate=validate,
        validate_mode=validate_mode,
        intern_fn=intern_fn,
        safe_gather_fn=safe_gather_fn,
    )


def cycle_candidates(
    ledger,
    frontier_ids,
    validate_stratum: bool = False,
    validate_mode: str = "strict",
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    cnf2_flags: Cnf2Flags | None = None,
    cnf2_enabled_fn=None,
    cnf2_slot1_enabled_fn=None,
):
    """Interface/Control wrapper for CNF-2 evaluation with DI hooks.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_candidate_cycle.py
    """
    if intern_fn is None:
        intern_fn = intern_nodes
    if cnf2_cfg is not None and cnf2_flags is not None:
        cnf2_cfg = replace(cnf2_cfg, flags=cnf2_flags)
    elif cnf2_cfg is None and cnf2_flags is not None:
        cnf2_cfg = Cnf2Config(flags=cnf2_flags)
    if cnf2_cfg is not None:
        if intern_cfg is None:
            intern_cfg = cnf2_cfg.intern_cfg
        if safe_gather_policy is None and cnf2_cfg.safe_gather_policy is not None:
            safe_gather_policy = cnf2_cfg.safe_gather_policy
        if intern_fn is None and cnf2_cfg.intern_fn is not None:
            intern_fn = cnf2_cfg.intern_fn
        if emit_candidates_fn is None and cnf2_cfg.emit_candidates_fn is not None:
            emit_candidates_fn = cnf2_cfg.emit_candidates_fn
        if cnf2_enabled_fn is None and cnf2_cfg.cnf2_enabled_fn is not None:
            cnf2_enabled_fn = cnf2_cfg.cnf2_enabled_fn
        if cnf2_slot1_enabled_fn is None and cnf2_cfg.cnf2_slot1_enabled_fn is not None:
            cnf2_slot1_enabled_fn = cnf2_cfg.cnf2_slot1_enabled_fn
        cnf2_flags = cnf2_cfg.flags if cnf2_flags is None else cnf2_flags
    def _resolve_gate(flag_value, fn_value, default_fn):
        if flag_value is not None:
            return bool(flag_value)
        if fn_value is not None:
            return bool(fn_value())
        return bool(default_fn())

    if cnf2_flags is not None:
        if cnf2_enabled_fn is not None or cnf2_slot1_enabled_fn is not None:
            raise ValueError("Pass either cnf2_flags or cnf2_*_enabled_fn, not both.")
        enabled_value = _resolve_gate(
            cnf2_flags.enabled, None, _cnf2_enabled_default
        )
        slot1_value = _resolve_gate(
            cnf2_flags.slot1_enabled, None, _cnf2_slot1_enabled_default
        )
        cnf2_enabled_fn = lambda: enabled_value
        cnf2_slot1_enabled_fn = lambda: slot1_value
    if emit_candidates_fn is None:
        emit_candidates_fn = _emit_candidates_default
    if cnf2_flags is None:
        enabled_value = _resolve_gate(None, cnf2_enabled_fn, _cnf2_enabled_default)
        slot1_value = _resolve_gate(
            None, cnf2_slot1_enabled_fn, _cnf2_slot1_enabled_default
        )
        cnf2_enabled_fn = lambda: enabled_value
        cnf2_slot1_enabled_fn = lambda: slot1_value
    if host_raise_if_bad_fn is None:
        host_raise_if_bad_fn = _host_raise_if_bad
    if not cnf2_enabled_fn():
        raise RuntimeError("cycle_candidates disabled until m2 (set PRISM_ENABLE_CNF2=1)")
    ledger, frontier_ids, strata, q_map = _cycle_candidates_impl(
        ledger,
        frontier_ids,
        validate_stratum=validate_stratum,
        validate_mode=validate_mode,
        cfg=cnf2_cfg,
        safe_gather_policy=safe_gather_policy,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        cnf2_enabled_fn=cnf2_enabled_fn,
        cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
    )
    if not bool(jax.device_get(ledger.corrupt)):
        host_raise_if_bad_fn(ledger, "Ledger capacity exceeded during cycle")
    return ledger, frontier_ids, strata, q_map
