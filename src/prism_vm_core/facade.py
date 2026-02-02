from __future__ import annotations

"""Facade wrappers with explicit DI and glossary contracts.

Axis: Interface/Control (host-visible); wrappers must commute with q and be
erased by q. This module centralizes wrapper behavior to avoid accidental
shadowing or monkeypatching drift.
"""

# dataflow-bundle: cfg, guard, policy
# dataflow-bundle: cfg, guard, policy, return_ok
# dataflow-bundle: emit_candidates_fn, intern_fn
# dataflow-bundle: guard_cfg, safe_gather_value_fn

from typing import Optional
from functools import partial
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_core.di import call_with_optional_kwargs
from prism_core.safety import (
    PolicyMode,
    coerce_policy_mode,
    PolicyBinding,
    SafetyMode,
    coerce_safety_mode,
    SafetyPolicy,
    DEFAULT_SAFETY_POLICY,
    PolicyValue,
    POLICY_VALUE_DEFAULT,
    POLICY_VALUE_CLAMP,
    POLICY_VALUE_CORRUPT,
    POLICY_VALUE_DROP,
    policy_to_value,
    oob_any_value,
    oob_mask,
    oob_mask_value,
    resolve_policy_binding,
    require_static_policy,
    require_value_policy,
)
from prism_core.modes import ValidateMode, require_validate_mode, BspMode, coerce_bsp_mode
from prism_core.errors import PrismPolicyModeError, PrismPolicyBindingError
from prism_ledger import intern as _ledger_intern
from prism_ledger.config import InternConfig, DEFAULT_INTERN_CONFIG
from prism_ledger.index import (
    LedgerIndex,
    LedgerState,
    derive_ledger_index,
    derive_ledger_state,
)
from prism_bsp.config import (
    Cnf2Config,
    Cnf2BoundConfig,
    Cnf2StaticBoundConfig,
    Cnf2ValueBoundConfig,
    DEFAULT_CNF2_CONFIG,
    Cnf2RuntimeFns,
    DEFAULT_CNF2_RUNTIME_FNS,
    Cnf2CandidateFns,
    Cnf2PolicyFns,
    ArenaInteractConfig,
    DEFAULT_ARENA_INTERACT_CONFIG,
    SwizzleWithPermFns,
    SwizzleWithPermFnsBound,
    ArenaCycleConfig,
    DEFAULT_ARENA_CYCLE_CONFIG,
    ArenaSortConfig,
    DEFAULT_ARENA_SORT_CONFIG,
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
from prism_vm_core.domains import QMap, _host_bool, _host_raise_if_bad
from prism_vm_core.structures import Ledger, NodeBatch, Stratum
from prism_bsp.space import RANK_FREE
from prism_bsp.cnf2 import (
    emit_candidates as _emit_candidates_default,
    compact_candidates as _compact_candidates,
    compact_candidates_result as _compact_candidates_result,
    compact_candidates_with_index as _compact_candidates_with_index,
    compact_candidates_with_index_result as _compact_candidates_with_index_result,
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
    cycle_core_value as _cycle_core_value,
    cycle_value as _cycle_value,
    op_interact as _op_interact,
    op_interact_bound_cfg,
    op_interact_cfg,
    op_interact_value as _op_interact_value,
    cycle_bound_cfg,
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
    op_sort_and_swizzle_value,
    op_sort_and_swizzle_blocked,
    op_sort_and_swizzle_blocked_value,
    op_sort_and_swizzle_blocked_with_perm,
    op_sort_and_swizzle_blocked_with_perm_value,
    op_sort_and_swizzle_hierarchical,
    op_sort_and_swizzle_hierarchical_value,
    op_sort_and_swizzle_hierarchical_with_perm,
    op_sort_and_swizzle_hierarchical_with_perm_value,
    op_sort_and_swizzle_morton,
    op_sort_and_swizzle_morton_value,
    op_sort_and_swizzle_morton_with_perm,
    op_sort_and_swizzle_morton_with_perm_value,
    op_sort_and_swizzle_servo,
    op_sort_and_swizzle_servo_value,
    op_sort_and_swizzle_servo_with_perm,
    op_sort_and_swizzle_servo_with_perm_value,
    op_sort_and_swizzle_with_perm,
    op_sort_and_swizzle_with_perm_value,
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
    binding = resolve_policy_binding(
        policy=safety_policy,
        policy_value=None,
        context="arena_interact_config_with_policy",
    )
    return replace(
        cfg,
        policy_binding=binding,
        safe_gather_policy=None,
        safe_gather_policy_value=None,
    )


def arena_interact_config_with_policy_value(
    policy_value: PolicyValue | None,
    *,
    cfg: ArenaInteractConfig = DEFAULT_ARENA_INTERACT_CONFIG,
) -> ArenaInteractConfig:
    """Return an ArenaInteractConfig with safe_gather_policy_value set."""
    binding = resolve_policy_binding(
        policy=None,
        policy_value=policy_value,
        context="arena_interact_config_with_policy_value",
    )
    return replace(
        cfg,
        policy_binding=binding,
        safe_gather_policy=None,
        safe_gather_policy_value=None,
    )


def arena_interact_config_with_guard(
    guard_cfg: GuardConfig | None,
    *,
    cfg: ArenaInteractConfig = DEFAULT_ARENA_INTERACT_CONFIG,
) -> ArenaInteractConfig:
    """Return an ArenaInteractConfig with guard_cfg set."""
    return replace(cfg, guard_cfg=guard_cfg)


def arena_cycle_config_with_policy(
    safety_policy: SafetyPolicy | None,
    *,
    cfg: ArenaCycleConfig = DEFAULT_ARENA_CYCLE_CONFIG,
    include_interact: bool = True,
) -> ArenaCycleConfig:
    """Return an ArenaCycleConfig with safety_policy set (and optionally its interact_cfg)."""
    binding = resolve_policy_binding(
        policy=safety_policy,
        policy_value=None,
        context="arena_cycle_config_with_policy",
    )
    interact_cfg = cfg.interact_cfg
    if include_interact:
        if interact_cfg is None:
            interact_cfg = DEFAULT_ARENA_INTERACT_CONFIG
        interact_cfg = replace(
            interact_cfg,
            policy_binding=binding,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )
    return replace(
        cfg,
        policy_binding=binding,
        safe_gather_policy=None,
        safe_gather_policy_value=None,
        interact_cfg=interact_cfg,
    )


def arena_cycle_config_with_policy_value(
    policy_value: PolicyValue | None,
    *,
    cfg: ArenaCycleConfig = DEFAULT_ARENA_CYCLE_CONFIG,
    include_interact: bool = True,
) -> ArenaCycleConfig:
    """Return an ArenaCycleConfig with safe_gather_policy_value set."""
    binding = resolve_policy_binding(
        policy=None,
        policy_value=policy_value,
        context="arena_cycle_config_with_policy_value",
    )
    interact_cfg = cfg.interact_cfg
    if include_interact:
        if interact_cfg is None:
            interact_cfg = DEFAULT_ARENA_INTERACT_CONFIG
        interact_cfg = replace(
            interact_cfg,
            policy_binding=binding,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )
    return replace(
        cfg,
        policy_binding=binding,
        safe_gather_policy=None,
        safe_gather_policy_value=None,
        interact_cfg=interact_cfg,
    )


def arena_cycle_config_with_guard(
    guard_cfg: GuardConfig | None,
    *,
    cfg: ArenaCycleConfig = DEFAULT_ARENA_CYCLE_CONFIG,
    include_interact: bool = True,
) -> ArenaCycleConfig:
    """Return an ArenaCycleConfig with guard_cfg set (and optionally its interact_cfg)."""
    interact_cfg = cfg.interact_cfg
    if include_interact:
        if interact_cfg is None:
            interact_cfg = DEFAULT_ARENA_INTERACT_CONFIG
        interact_cfg = replace(interact_cfg, guard_cfg=guard_cfg)
    return replace(cfg, guard_cfg=guard_cfg, interact_cfg=interact_cfg)


def op_sort_with_perm_cfg(
    arena,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Interface/Control wrapper for op_sort_and_swizzle_with_perm with guard cfg."""
    bundle = _GuardSafeGatherValueBundle(
        guard_cfg=guard_cfg, safe_gather_value_fn=safe_gather_value_fn
    )
    if safe_gather_policy_value is not None:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_with_perm_value,
            {
                "guard_cfg": bundle.guard_cfg,
                "safe_gather_value_fn": bundle.safe_gather_value_fn,
            },
            arena,
            safe_gather_policy_value,
        )
    policy_bound = _safe_gather_is_bound(safe_gather_fn)
    if safe_gather_policy is None and policy_bound:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_with_perm,
            {
                "safe_gather_policy": None,
                "guard_cfg": guard_cfg,
            },
            arena,
            safe_gather_fn=safe_gather_fn,
        )
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    return call_with_optional_kwargs(
        op_sort_and_swizzle_with_perm,
        {
            "safe_gather_policy": safe_gather_policy,
            "guard_cfg": guard_cfg,
        },
        arena,
        safe_gather_fn=safe_gather_fn,
    )


def op_sort_blocked_with_perm_cfg(
    arena,
    block_size,
    morton=None,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Interface/Control wrapper for op_sort_and_swizzle_blocked_with_perm with guard cfg."""
    bundle = _GuardSafeGatherValueBundle(
        guard_cfg=guard_cfg, safe_gather_value_fn=safe_gather_value_fn
    )
    if safe_gather_policy_value is not None:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_blocked_with_perm_value,
            {
                "guard_cfg": bundle.guard_cfg,
                "safe_gather_value_fn": bundle.safe_gather_value_fn,
            },
            arena,
            block_size,
            safe_gather_policy_value,
            morton=morton,
        )
    policy_bound = _safe_gather_is_bound(safe_gather_fn)
    if safe_gather_policy is None and policy_bound:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_blocked_with_perm,
            {
                "safe_gather_policy": None,
                "guard_cfg": guard_cfg,
            },
            arena,
            block_size,
            morton=morton,
            safe_gather_fn=safe_gather_fn,
        )
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    return call_with_optional_kwargs(
        op_sort_and_swizzle_blocked_with_perm,
        {
            "safe_gather_policy": safe_gather_policy,
            "guard_cfg": guard_cfg,
        },
        arena,
        block_size,
        morton=morton,
        safe_gather_fn=safe_gather_fn,
    )


def op_sort_hierarchical_with_perm_cfg(
    arena,
    l2_block_size,
    l1_block_size,
    morton=None,
    do_global=False,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Interface/Control wrapper for op_sort_and_swizzle_hierarchical_with_perm with guard cfg."""
    bundle = _GuardSafeGatherValueBundle(
        guard_cfg=guard_cfg, safe_gather_value_fn=safe_gather_value_fn
    )
    if safe_gather_policy_value is not None:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_hierarchical_with_perm_value,
            {
                "guard_cfg": bundle.guard_cfg,
                "safe_gather_value_fn": bundle.safe_gather_value_fn,
            },
            arena,
            l2_block_size,
            l1_block_size,
            safe_gather_policy_value,
            morton=morton,
            do_global=do_global,
        )
    policy_bound = _safe_gather_is_bound(safe_gather_fn)
    if safe_gather_policy is None and policy_bound:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_hierarchical_with_perm,
            {
                "safe_gather_policy": None,
                "guard_cfg": guard_cfg,
            },
            arena,
            l2_block_size,
            l1_block_size,
            morton=morton,
            do_global=do_global,
            safe_gather_fn=safe_gather_fn,
        )
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    return call_with_optional_kwargs(
        op_sort_and_swizzle_hierarchical_with_perm,
        {
            "safe_gather_policy": safe_gather_policy,
            "guard_cfg": guard_cfg,
        },
        arena,
        l2_block_size,
        l1_block_size,
        morton=morton,
        do_global=do_global,
        safe_gather_fn=safe_gather_fn,
    )


def op_sort_morton_with_perm_cfg(
    arena,
    morton,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Interface/Control wrapper for op_sort_and_swizzle_morton_with_perm with guard cfg."""
    bundle = _GuardSafeGatherValueBundle(
        guard_cfg=guard_cfg, safe_gather_value_fn=safe_gather_value_fn
    )
    if safe_gather_policy_value is not None:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_morton_with_perm_value,
            {
                "guard_cfg": bundle.guard_cfg,
                "safe_gather_value_fn": bundle.safe_gather_value_fn,
            },
            arena,
            morton,
            safe_gather_policy_value,
        )
    policy_bound = _safe_gather_is_bound(safe_gather_fn)
    if safe_gather_policy is None and policy_bound:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_morton_with_perm,
            {
                "safe_gather_policy": None,
                "guard_cfg": guard_cfg,
            },
            arena,
            morton,
            safe_gather_fn=safe_gather_fn,
        )
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    return call_with_optional_kwargs(
        op_sort_and_swizzle_morton_with_perm,
        {
            "safe_gather_policy": safe_gather_policy,
            "guard_cfg": guard_cfg,
        },
        arena,
        morton,
        safe_gather_fn=safe_gather_fn,
    )


def op_sort_servo_with_perm_cfg(
    arena,
    morton,
    servo_mask,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Interface/Control wrapper for op_sort_and_swizzle_servo_with_perm with guard cfg."""
    bundle = _GuardSafeGatherValueBundle(
        guard_cfg=guard_cfg, safe_gather_value_fn=safe_gather_value_fn
    )
    if safe_gather_policy_value is not None:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_servo_with_perm_value,
            {
                "guard_cfg": bundle.guard_cfg,
                "safe_gather_value_fn": bundle.safe_gather_value_fn,
            },
            arena,
            morton,
            servo_mask,
            safe_gather_policy_value,
        )
    policy_bound = _safe_gather_is_bound(safe_gather_fn)
    if safe_gather_policy is None and policy_bound:
        return call_with_optional_kwargs(
            op_sort_and_swizzle_servo_with_perm,
            {
                "safe_gather_policy": None,
                "guard_cfg": guard_cfg,
            },
            arena,
            morton,
            servo_mask,
            safe_gather_fn=safe_gather_fn,
        )
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    return call_with_optional_kwargs(
        op_sort_and_swizzle_servo_with_perm,
        {
            "safe_gather_policy": safe_gather_policy,
            "guard_cfg": guard_cfg,
        },
        arena,
        morton,
        servo_mask,
        safe_gather_fn=safe_gather_fn,
    )


def cnf2_config_with_policy(
    safety_policy: SafetyPolicy | None,
    *,
    cfg: Cnf2Config = DEFAULT_CNF2_CONFIG,
) -> Cnf2Config:
    """Return a Cnf2Config with safe_gather_policy set."""
    binding = resolve_policy_binding(
        policy=safety_policy,
        policy_value=None,
        context="cnf2_config_with_policy",
    )
    return replace(
        cfg,
        policy_binding=binding,
        safe_gather_policy=None,
        safe_gather_policy_value=None,
    )


def cnf2_config_bound(
    policy_binding: PolicyBinding,
    *,
    cfg: Cnf2Config = DEFAULT_CNF2_CONFIG,
) -> Cnf2BoundConfig:
    """Return a Cnf2BoundConfig with required policy binding."""
    if policy_binding.mode == PolicyMode.VALUE:
        return Cnf2ValueBoundConfig.bind(cfg, policy_binding)
    return Cnf2StaticBoundConfig.bind(cfg, policy_binding)


def cnf2_config_with_policy_value(
    policy_value: PolicyValue | None,
    *,
    cfg: Cnf2Config = DEFAULT_CNF2_CONFIG,
) -> Cnf2Config:
    """Return a Cnf2Config with safe_gather_policy_value set."""
    binding = resolve_policy_binding(
        policy=None,
        policy_value=policy_value,
        context="cnf2_config_with_policy_value",
    )
    return replace(
        cfg,
        policy_binding=binding,
        safe_gather_policy=None,
        safe_gather_policy_value=None,
    )


def cnf2_config_with_guard(
    guard_cfg: GuardConfig | None,
    *,
    cfg: Cnf2Config = DEFAULT_CNF2_CONFIG,
) -> Cnf2Config:
    """Return a Cnf2Config with guard_cfg set."""
    return replace(cfg, guard_cfg=guard_cfg)
from prism_vm_core.gating import (
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
    safe_gather_1d_cfg as _safe_gather_1d_cfg,
    safe_gather_1d_ok_cfg as _safe_gather_1d_ok_cfg,
    safe_index_1d_cfg as _safe_index_1d_cfg,
    make_safe_gather_fn,
    make_safe_gather_ok_fn,
    make_safe_index_fn,
    make_safe_gather_value_fn,
    make_safe_gather_ok_value_fn,
    make_safe_index_value_fn,
    resolve_safe_gather_fn,
    resolve_safe_gather_ok_fn,
    resolve_safe_index_fn,
    resolve_safe_gather_value_fn,
    resolve_safe_gather_ok_value_fn,
    resolve_safe_index_value_fn,
    guard_slot0_perm_cfg,
    guard_null_row_cfg,
    guard_zero_row_cfg,
    guard_zero_args_cfg,
    guard_swizzle_args_cfg,
)


@dataclass(frozen=True)
class _SafeGatherCfgArgs:
    cfg: GuardConfig
    guard: object
    policy: SafetyPolicy | None


@dataclass(frozen=True)
class _SafeGatherCfgReturnOkArgs:
    cfg: GuardConfig
    guard: object
    policy: SafetyPolicy | None
    return_ok: bool


@dataclass(frozen=True)
class _EmitInternFns:
    emit_candidates_fn: object
    intern_fn: object


@dataclass(frozen=True)
class _GuardSafeGatherValueBundle:
    guard_cfg: GuardConfig
    safe_gather_value_fn: object
from prism_semantics.commit import (
    commit_stratum as _commit_stratum_impl,
    commit_stratum_static as _commit_stratum_static_impl,
    commit_stratum_value as _commit_stratum_value_impl,
)
from prism_bsp.cnf2 import (
    cycle_candidates as _cycle_candidates_impl,
    cycle_candidates_bound as _cycle_candidates_bound,
    cycle_candidates_bound_state as _cycle_candidates_bound_state,
    cycle_candidates_state as _cycle_candidates_state,
    cycle_candidates_static as _cycle_candidates_static,
    cycle_candidates_static_state as _cycle_candidates_static_state,
    cycle_candidates_value as _cycle_candidates_value,
    cycle_candidates_value_state as _cycle_candidates_value_state,
)
from prism_vm_core.jit_entrypoints import (
    coord_norm_batch_jit,
    cycle_candidates_jit,
    cycle_candidates_state_jit,
    cycle_candidates_static_jit,
    cycle_candidates_static_state_jit,
    cycle_candidates_value_jit,
    cycle_candidates_value_state_jit,
    cycle_intrinsic_jit,
    cycle_intrinsic_jit_cfg,
    cycle_jit as _cycle_jit_factory,
    cycle_jit_cfg,
    cycle_jit_bound_cfg,
    emit_candidates_jit,
    emit_candidates_jit_cfg,
    compact_candidates_jit,
    compact_candidates_jit_cfg,
    compact_candidates_result_jit,
    compact_candidates_result_jit_cfg,
    compact_candidates_with_index_jit,
    compact_candidates_with_index_jit_cfg,
    compact_candidates_with_index_result_jit,
    compact_candidates_with_index_result_jit_cfg,
    intern_candidates_jit,
    intern_candidates_jit_cfg,
    intern_nodes_jit,
    intern_nodes_with_index_jit,
    op_interact_jit,
    op_interact_jit_cfg,
    op_interact_jit_bound_cfg,
    op_interact_value_jit,
    cycle_value_jit,
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
PolicyMode = PolicyMode
coerce_policy_mode = coerce_policy_mode
SafetyMode = SafetyMode
coerce_safety_mode = coerce_safety_mode
ValidateMode = ValidateMode
require_validate_mode = require_validate_mode
BspMode = BspMode
coerce_bsp_mode = coerce_bsp_mode
PolicyValue = PolicyValue
POLICY_VALUE_CORRUPT = POLICY_VALUE_CORRUPT
POLICY_VALUE_CLAMP = POLICY_VALUE_CLAMP
POLICY_VALUE_DROP = POLICY_VALUE_DROP
POLICY_VALUE_DEFAULT = POLICY_VALUE_DEFAULT
policy_to_value = policy_to_value
oob_mask = oob_mask
oob_mask_value = oob_mask_value
oob_any_value = oob_any_value
GuardConfig = GuardConfig
DEFAULT_GUARD_CONFIG = DEFAULT_GUARD_CONFIG
guards_enabled_cfg = guards_enabled_cfg
guard_max_cfg = guard_max_cfg
guard_gather_index_cfg = guard_gather_index_cfg
make_safe_gather_fn = make_safe_gather_fn
make_safe_gather_ok_fn = make_safe_gather_ok_fn
make_safe_index_fn = make_safe_index_fn
make_safe_gather_value_fn = make_safe_gather_value_fn
make_safe_gather_ok_value_fn = make_safe_gather_ok_value_fn
make_safe_index_value_fn = make_safe_index_value_fn
resolve_safe_gather_fn = resolve_safe_gather_fn
resolve_safe_gather_ok_fn = resolve_safe_gather_ok_fn
resolve_safe_index_fn = resolve_safe_index_fn
resolve_safe_gather_value_fn = resolve_safe_gather_value_fn
resolve_safe_gather_ok_value_fn = resolve_safe_gather_ok_value_fn
resolve_safe_index_value_fn = resolve_safe_index_value_fn
guard_slot0_perm_cfg = guard_slot0_perm_cfg
guard_null_row_cfg = guard_null_row_cfg
guard_zero_row_cfg = guard_zero_row_cfg
guard_zero_args_cfg = guard_zero_args_cfg
guard_swizzle_args_cfg = guard_swizzle_args_cfg

def _safe_gather_is_bound(safe_gather_fn) -> bool:
    if safe_gather_fn is None:
        return False
    return bool(getattr(safe_gather_fn, "_prism_policy_bound", False))

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
    safe_gather_fn = resolve_safe_gather_fn(
        safe_gather_fn=_jax_safe.safe_gather_1d,
        policy=policy,
    )
    return call_with_optional_kwargs(
        safe_gather_fn,
        {"guard": guard, "return_ok": return_ok},
        arr,
        idx,
        label,
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
    safe_gather_ok_fn = resolve_safe_gather_ok_fn(
        safe_gather_ok_fn=_jax_safe.safe_gather_1d_ok,
        policy=policy,
    )
    return call_with_optional_kwargs(
        safe_gather_ok_fn,
        {"guard": guard},
        arr,
        idx,
        label,
    )


def safe_gather_1d_cfg(
    arr,
    idx,
    label="safe_gather_1d",
    *,
    guard=None,
    policy: SafetyPolicy | None = None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    return_ok: bool = False,
):
    """Interface/Control wrapper for safe_gather_1d with guard config."""
    bundle = _SafeGatherCfgReturnOkArgs(
        cfg=cfg, guard=guard, policy=policy, return_ok=return_ok
    )
    return call_with_optional_kwargs(
        _safe_gather_1d_cfg,
        {
            "guard": bundle.guard,
            "policy": bundle.policy,
            "cfg": bundle.cfg,
            "return_ok": bundle.return_ok,
        },
        arr,
        idx,
        label,
    )


def safe_gather_1d_ok_cfg(
    arr,
    idx,
    label="safe_gather_1d_ok",
    *,
    guard=None,
    policy: SafetyPolicy | None = None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    """Interface/Control wrapper for safe_gather_1d_ok with guard config."""
    bundle = _SafeGatherCfgArgs(cfg=cfg, guard=guard, policy=policy)
    return call_with_optional_kwargs(
        _safe_gather_1d_ok_cfg,
        {"guard": bundle.guard, "policy": bundle.policy, "cfg": bundle.cfg},
        arr,
        idx,
        label,
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
    safe_index_fn = resolve_safe_index_fn(
        safe_index_fn=_jax_safe.safe_index_1d,
        policy=policy,
    )
    return call_with_optional_kwargs(
        safe_index_fn, {"guard": guard}, idx, size, label
    )


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
    bundle = _SafeGatherCfgArgs(cfg=cfg, guard=guard, policy=policy)
    return call_with_optional_kwargs(
        _safe_index_1d_cfg,
        {"guard": bundle.guard, "policy": bundle.policy, "cfg": bundle.cfg},
        idx,
        size,
        label,
    )

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


def init_ledger_state(*, cfg: InternConfig | None = None):
    """Initialize a LedgerState with a derived index."""
    ledger = init_ledger()
    if cfg is None:
        cfg = DEFAULT_INTERN_CONFIG
    return derive_ledger_state(
        ledger, op_buckets_full_range=cfg.op_buckets_full_range
    )


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
emit_candidates = _emit_candidates
emit_candidates_cfg = _emit_candidates_cfg
compact_candidates = _compact_candidates
compact_candidates_result = _compact_candidates_result
compact_candidates_cfg = _compact_candidates_cfg
compact_candidates_with_index = _compact_candidates_with_index
compact_candidates_with_index_result = _compact_candidates_with_index_result
compact_candidates_with_index_cfg = _compact_candidates_with_index_cfg
intern_candidates = _intern_candidates
intern_candidates_cfg = _intern_candidates_cfg
cycle = _cycle
cycle_core = _cycle_core
cycle_core_value = _cycle_core_value
cycle_value = _cycle_value
op_interact = _op_interact
op_interact_value = _op_interact_value
cycle_intrinsic = _cycle_intrinsic
RANK_COLD = RANK_COLD
RANK_FREE = _RANK_FREE_EXPORT
RANK_HOT = RANK_HOT
RANK_WARM = RANK_WARM
_blocked_perm = _blocked_perm
op_morton = op_morton
op_rank = op_rank
op_sort_and_swizzle = op_sort_and_swizzle
op_sort_and_swizzle_value = op_sort_and_swizzle_value
op_sort_and_swizzle_blocked = op_sort_and_swizzle_blocked
op_sort_and_swizzle_blocked_value = op_sort_and_swizzle_blocked_value
op_sort_and_swizzle_blocked_with_perm = op_sort_and_swizzle_blocked_with_perm
op_sort_and_swizzle_blocked_with_perm_value = op_sort_and_swizzle_blocked_with_perm_value
op_sort_and_swizzle_hierarchical = op_sort_and_swizzle_hierarchical
op_sort_and_swizzle_hierarchical_value = op_sort_and_swizzle_hierarchical_value
op_sort_and_swizzle_hierarchical_with_perm = op_sort_and_swizzle_hierarchical_with_perm
op_sort_and_swizzle_hierarchical_with_perm_value = op_sort_and_swizzle_hierarchical_with_perm_value
op_sort_and_swizzle_morton = op_sort_and_swizzle_morton
op_sort_and_swizzle_morton_value = op_sort_and_swizzle_morton_value
op_sort_and_swizzle_morton_with_perm = op_sort_and_swizzle_morton_with_perm
op_sort_and_swizzle_morton_with_perm_value = op_sort_and_swizzle_morton_with_perm_value
op_sort_and_swizzle_servo = op_sort_and_swizzle_servo
op_sort_and_swizzle_servo_value = op_sort_and_swizzle_servo_value
op_sort_and_swizzle_servo_with_perm = op_sort_and_swizzle_servo_with_perm
op_sort_and_swizzle_servo_with_perm_value = op_sort_and_swizzle_servo_with_perm_value
op_sort_and_swizzle_with_perm = op_sort_and_swizzle_with_perm
op_sort_and_swizzle_with_perm_value = op_sort_and_swizzle_with_perm_value
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
LedgerIndex = LedgerIndex
derive_ledger_index = derive_ledger_index
LedgerState = LedgerState
derive_ledger_state = derive_ledger_state
Cnf2Config = Cnf2Config
DEFAULT_CNF2_CONFIG = DEFAULT_CNF2_CONFIG
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
    ledger: Ledger | LedgerState,
    batch_or_ops,
    a1=None,
    a2=None,
    *,
    cfg: InternConfig | None = None,
    op_buckets_full_range: Optional[bool] = None,
    force_spawn_clip: Optional[bool] = None,
    ledger_index: LedgerIndex | None = None,
):
    """Interface/Control wrapper for intern_nodes behavior flags.

    If ledger_index is provided, the bound LedgerIndex path is used to avoid
    recomputing opcode buckets for the same ledger state.

    If ledger is a LedgerState, the LedgerIndex path is used implicitly and the
    updated LedgerState is returned.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_m1_gate.py
    """
    if isinstance(ledger, LedgerState):
        if ledger_index is not None:
            raise ValueError("Pass either LedgerState or ledger_index, not both.")
        if cfg is not None and (op_buckets_full_range is not None or force_spawn_clip is not None):
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
        return intern_nodes_state(ledger, batch, cfg=cfg)
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
    if ledger_index is None:
        ledger_index = derive_ledger_index(
            ledger, op_buckets_full_range=cfg.op_buckets_full_range
        )
    return intern_nodes_jit(cfg)(ledger, ledger_index, batch)


def intern_nodes_with_index(
    ledger: Ledger,
    ledger_index: LedgerIndex,
    batch_or_ops,
    a1=None,
    a2=None,
    *,
    cfg: InternConfig | None = None,
):
    """Interface/Control wrapper for intern_nodes with a bound LedgerIndex."""
    if cfg is None:
        cfg = DEFAULT_INTERN_CONFIG
    if a1 is None and a2 is None:
        if not isinstance(batch_or_ops, NodeBatch):
            raise TypeError("intern_nodes_with_index expects a NodeBatch or (ops, a1, a2)")
        batch = batch_or_ops
    else:
        if a1 is None or a2 is None:
            raise TypeError("intern_nodes_with_index expects both a1 and a2 arrays")
        batch = NodeBatch(batch_or_ops, a1, a2)
    return intern_nodes_with_index_jit(cfg)(ledger, ledger_index, batch)


def intern_nodes_state(
    state: LedgerState,
    batch_or_ops,
    a1=None,
    a2=None,
    *,
    cfg: InternConfig | None = None,
):
    """Interface/Control wrapper for intern_nodes on LedgerState."""
    if cfg is None:
        cfg = DEFAULT_INTERN_CONFIG
    if a1 is None and a2 is None:
        if not isinstance(batch_or_ops, NodeBatch):
            raise TypeError("intern_nodes_state expects a NodeBatch or (ops, a1, a2)")
        batch = batch_or_ops
    else:
        if a1 is None or a2 is None:
            raise TypeError("intern_nodes_state expects both a1 and a2 arrays")
        batch = NodeBatch(batch_or_ops, a1, a2)
    ids, new_state = _ledger_intern.intern_nodes_state(
        state,
        batch,
        cfg=cfg,
    )
    return ids, new_state


def cycle_jit(
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound | None = None,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    op_interact_fn=_op_interact,
):
    """Return a jitted cycle entrypoint for fixed DI."""
    if swizzle_with_perm_fns is None:
        swizzle_with_perm_fns = SwizzleWithPermFnsBound(
            with_perm=op_sort_and_swizzle_with_perm,
            morton_with_perm=op_sort_and_swizzle_morton_with_perm,
            blocked_with_perm=op_sort_and_swizzle_blocked_with_perm,
            hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm,
            servo_with_perm=op_sort_and_swizzle_servo_with_perm,
        )
    return _cycle_jit_factory(
        sort_cfg=sort_cfg,
        op_rank_fn=op_rank_fn,
        servo_enabled_fn=servo_enabled_fn,
        servo_update_fn=servo_update_fn,
        op_morton_fn=op_morton_fn,
        swizzle_with_perm_fns=swizzle_with_perm_fns,
        safe_gather_fn=safe_gather_fn,
        op_interact_fn=op_interact_fn,
        test_guards=_TEST_GUARDS,
    )


def commit_stratum_static(
    ledger: Ledger,
    stratum: Stratum,
    prior_q: QMap | None = None,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    *,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
    policy_binding: PolicyBinding | None = None,
):
    """Static-policy wrapper for commit_stratum injection."""
    if intern_fn is None:
        intern_fn = intern_nodes
    ledger_index = derive_ledger_index(
        ledger, op_buckets_full_range=DEFAULT_INTERN_CONFIG.op_buckets_full_range
    )
    def _intern_with_index(ledger_in, batch_or_ops, a1=None, a2=None):
        return call_with_optional_kwargs(
            intern_fn,
            {"ledger_index": ledger_index},
            ledger_in,
            batch_or_ops,
            a1,
            a2,
        )

    setattr(_intern_with_index, "_prism_ledger_index_bound", True)
    return _commit_stratum_static_impl(
        ledger,
        stratum,
        prior_q=prior_q,
        validate_mode=validate_mode,
        intern_fn=_intern_with_index,
        ledger_index=ledger_index,
        intern_cfg=DEFAULT_INTERN_CONFIG,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
        policy_binding=policy_binding,
    )


def commit_stratum_value(
    ledger: Ledger,
    stratum: Stratum,
    prior_q: QMap | None = None,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    *,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    policy_binding: PolicyBinding | None = None,
):
    """Policy-value wrapper for commit_stratum injection."""
    if intern_fn is None:
        intern_fn = intern_nodes
    ledger_index = derive_ledger_index(
        ledger, op_buckets_full_range=DEFAULT_INTERN_CONFIG.op_buckets_full_range
    )
    def _intern_with_index(ledger_in, batch_or_ops, a1=None, a2=None):
        return call_with_optional_kwargs(
            intern_fn,
            {"ledger_index": ledger_index},
            ledger_in,
            batch_or_ops,
            a1,
            a2,
        )

    setattr(_intern_with_index, "_prism_ledger_index_bound", True)
    return _commit_stratum_value_impl(
        ledger,
        stratum,
        prior_q=prior_q,
        validate_mode=validate_mode,
        intern_fn=_intern_with_index,
        ledger_index=ledger_index,
        intern_cfg=DEFAULT_INTERN_CONFIG,
        safe_gather_policy_value=safe_gather_policy_value,
        guard_cfg=guard_cfg,
        policy_binding=policy_binding,
    )


def commit_stratum(
    ledger: Ledger,
    stratum: Stratum,
    prior_q: QMap | None = None,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    *,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    policy_binding: PolicyBinding | None = None,
):
    """Interface/Control wrapper for commit_stratum injection.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_commit_stratum.py
    """
    if policy_binding is not None:
        if safe_gather_policy is not None or safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "commit_stratum received both policy_binding and "
                "safe_gather_policy/safe_gather_policy_value",
                context="commit_stratum",
                policy_mode="ambiguous",
            )
        binding = policy_binding
    else:
        binding = resolve_policy_binding(
            policy=safe_gather_policy,
            policy_value=safe_gather_policy_value,
            context="commit_stratum",
        )
    if intern_fn is None:
        intern_fn = intern_nodes
    if binding.mode == PolicyMode.VALUE:
        return commit_stratum_value(
            ledger,
            stratum,
            prior_q=prior_q,
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            policy_binding=binding,
            guard_cfg=guard_cfg,
        )
    return commit_stratum_static(
        ledger,
        stratum,
        prior_q=prior_q,
        validate_mode=validate_mode,
        intern_fn=intern_fn,
        policy_binding=binding,
        guard_cfg=guard_cfg,
    )


def _cycle_candidates_common(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode,
    *,
    policy_mode: PolicyMode,
    intern_fn: InternFn | None,
    intern_cfg: InternConfig | None,
    emit_candidates_fn: EmitCandidatesFn | None,
    host_raise_if_bad_fn: HostRaiseFn | None,
    safe_gather_policy: SafetyPolicy | None,
    safe_gather_policy_value: PolicyValue | None,
    guard_cfg: GuardConfig | None,
    cnf2_cfg: Cnf2Config | None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Shared wrapper for CNF-2 entrypoints with explicit policy mode."""
    if not isinstance(policy_mode, PolicyMode):
        raise PrismPolicyModeError(mode=policy_mode, context="cycle_candidates")
    if intern_fn is None:
        intern_fn = intern_nodes
    if cnf2_cfg is not None:
        if intern_cfg is None:
            intern_cfg = cnf2_cfg.intern_cfg
        if guard_cfg is None and cnf2_cfg.guard_cfg is not None:
            guard_cfg = cnf2_cfg.guard_cfg
        if intern_fn is None and cnf2_cfg.intern_fn is not None:
            intern_fn = cnf2_cfg.intern_fn
        if emit_candidates_fn is None and cnf2_cfg.emit_candidates_fn is not None:
            emit_candidates_fn = cnf2_cfg.emit_candidates_fn
        if runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
            runtime_fns = cnf2_cfg.runtime_fns
        if policy_mode == PolicyMode.STATIC:
            if cnf2_cfg.policy_binding is not None:
                if cnf2_cfg.policy_binding.mode == PolicyMode.VALUE:
                    raise PrismPolicyBindingError(
                        "cycle_candidates_static received cfg.policy_binding value-mode; "
                        "use cycle_candidates_value",
                        context="cycle_candidates_static",
                        policy_mode=PolicyMode.STATIC,
                    )
                if safe_gather_policy is None:
                    safe_gather_policy = require_static_policy(
                        cnf2_cfg.policy_binding, context="cycle_candidates_static"
                    )
            if cnf2_cfg.safe_gather_policy_value is not None:
                raise PrismPolicyBindingError(
                    "cycle_candidates_static received cfg.safe_gather_policy_value; "
                    "use cycle_candidates_value",
                    context="cycle_candidates_static",
                    policy_mode=PolicyMode.STATIC,
                )
            if safe_gather_policy is None and cnf2_cfg.safe_gather_policy is not None:
                safe_gather_policy = cnf2_cfg.safe_gather_policy
        else:
            if cnf2_cfg.policy_binding is not None:
                if cnf2_cfg.policy_binding.mode == PolicyMode.STATIC:
                    raise PrismPolicyBindingError(
                        "cycle_candidates_value received cfg.policy_binding static-mode; "
                        "use cycle_candidates_static",
                        context="cycle_candidates_value",
                        policy_mode=PolicyMode.VALUE,
                    )
                if safe_gather_policy_value is None:
                    safe_gather_policy_value = require_value_policy(
                        cnf2_cfg.policy_binding, context="cycle_candidates_value"
                    )
            if cnf2_cfg.safe_gather_policy is not None:
                raise PrismPolicyBindingError(
                    "cycle_candidates_value received cfg.safe_gather_policy; "
                    "use cycle_candidates_static",
                    context="cycle_candidates_value",
                    policy_mode=PolicyMode.VALUE,
                )
            if (
                safe_gather_policy_value is None
                and cnf2_cfg.safe_gather_policy_value is not None
            ):
                safe_gather_policy_value = cnf2_cfg.safe_gather_policy_value
    if policy_mode == PolicyMode.STATIC:
        if safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_static received safe_gather_policy_value; "
                "use cycle_candidates_value",
                context="cycle_candidates_static",
                policy_mode=PolicyMode.STATIC,
            )
    else:
        if safe_gather_policy is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_value received safe_gather_policy; "
                "use cycle_candidates_static",
                context="cycle_candidates_value",
                policy_mode=PolicyMode.VALUE,
            )
    if emit_candidates_fn is None:
        emit_candidates_fn = _emit_candidates_default
    if host_raise_if_bad_fn is None:
        host_raise_if_bad_fn = _host_raise_if_bad
    if policy_mode == PolicyMode.STATIC:
        if safe_gather_policy is None:
            safe_gather_policy = DEFAULT_SAFETY_POLICY
        ledger, frontier_ids, strata, q_map = _cycle_candidates_static(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cnf2_cfg,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            runtime_fns=runtime_fns,
        )
    else:
        if safe_gather_policy_value is None:
            safe_gather_policy_value = POLICY_VALUE_DEFAULT
        ledger, frontier_ids, strata, q_map = _cycle_candidates_value(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cnf2_cfg,
            safe_gather_policy_value=safe_gather_policy_value,
            guard_cfg=guard_cfg,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            runtime_fns=runtime_fns,
        )
    if not bool(jax.device_get(ledger.corrupt)):
        host_raise_if_bad_fn(ledger, "Ledger capacity exceeded during cycle")
    return ledger, frontier_ids, strata, q_map


def _cycle_candidates_common_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode,
    *,
    policy_mode: PolicyMode,
    intern_fn: InternFn | None,
    intern_cfg: InternConfig | None,
    emit_candidates_fn: EmitCandidatesFn | None,
    host_raise_if_bad_fn: HostRaiseFn | None,
    safe_gather_policy: SafetyPolicy | None,
    safe_gather_policy_value: PolicyValue | None,
    guard_cfg: GuardConfig | None,
    cnf2_cfg: Cnf2Config | None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Shared wrapper for CNF-2 entrypoints returning LedgerState."""
    if not isinstance(policy_mode, PolicyMode):
        raise PrismPolicyModeError(mode=policy_mode, context="cycle_candidates_state")
    if intern_fn is None:
        intern_fn = intern_nodes
    if cnf2_cfg is not None:
        if intern_cfg is None:
            intern_cfg = cnf2_cfg.intern_cfg
        if guard_cfg is None and cnf2_cfg.guard_cfg is not None:
            guard_cfg = cnf2_cfg.guard_cfg
        if intern_fn is None and cnf2_cfg.intern_fn is not None:
            intern_fn = cnf2_cfg.intern_fn
        if emit_candidates_fn is None and cnf2_cfg.emit_candidates_fn is not None:
            emit_candidates_fn = cnf2_cfg.emit_candidates_fn
        if runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
            runtime_fns = cnf2_cfg.runtime_fns
        if policy_mode == PolicyMode.STATIC:
            if cnf2_cfg.policy_binding is not None:
                if cnf2_cfg.policy_binding.mode == PolicyMode.VALUE:
                    raise PrismPolicyBindingError(
                        "cycle_candidates_static_state received cfg.policy_binding value-mode; "
                        "use cycle_candidates_value_state",
                        context="cycle_candidates_static_state",
                        policy_mode=PolicyMode.STATIC,
                    )
                if safe_gather_policy is None:
                    safe_gather_policy = require_static_policy(
                        cnf2_cfg.policy_binding,
                        context="cycle_candidates_static_state",
                    )
            if cnf2_cfg.safe_gather_policy_value is not None:
                raise PrismPolicyBindingError(
                    "cycle_candidates_static_state received cfg.safe_gather_policy_value; "
                    "use cycle_candidates_value_state",
                    context="cycle_candidates_static_state",
                    policy_mode=PolicyMode.STATIC,
                )
            if safe_gather_policy is None and cnf2_cfg.safe_gather_policy is not None:
                safe_gather_policy = cnf2_cfg.safe_gather_policy
        else:
            if cnf2_cfg.policy_binding is not None:
                if cnf2_cfg.policy_binding.mode == PolicyMode.STATIC:
                    raise PrismPolicyBindingError(
                        "cycle_candidates_value_state received cfg.policy_binding static-mode; "
                        "use cycle_candidates_static_state",
                        context="cycle_candidates_value_state",
                        policy_mode=PolicyMode.VALUE,
                    )
                if safe_gather_policy_value is None:
                    safe_gather_policy_value = require_value_policy(
                        cnf2_cfg.policy_binding,
                        context="cycle_candidates_value_state",
                    )
            if cnf2_cfg.safe_gather_policy is not None:
                raise PrismPolicyBindingError(
                    "cycle_candidates_value_state received cfg.safe_gather_policy; "
                    "use cycle_candidates_static_state",
                    context="cycle_candidates_value_state",
                    policy_mode=PolicyMode.VALUE,
                )
            if (
                safe_gather_policy_value is None
                and cnf2_cfg.safe_gather_policy_value is not None
            ):
                safe_gather_policy_value = cnf2_cfg.safe_gather_policy_value
    if policy_mode == PolicyMode.STATIC:
        if safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_static_state received safe_gather_policy_value; "
                "use cycle_candidates_value_state",
                context="cycle_candidates_static_state",
                policy_mode=PolicyMode.STATIC,
            )
    else:
        if safe_gather_policy is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_value_state received safe_gather_policy; "
                "use cycle_candidates_static_state",
                context="cycle_candidates_value_state",
                policy_mode=PolicyMode.VALUE,
            )
    if emit_candidates_fn is None:
        emit_candidates_fn = _emit_candidates_default
    if host_raise_if_bad_fn is None:
        host_raise_if_bad_fn = _host_raise_if_bad
    if policy_mode == PolicyMode.STATIC:
        if safe_gather_policy is None:
            safe_gather_policy = DEFAULT_SAFETY_POLICY
        state, frontier_ids, strata, q_map = _cycle_candidates_static_state(
            state,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cnf2_cfg,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            runtime_fns=runtime_fns,
        )
    else:
        if safe_gather_policy_value is None:
            safe_gather_policy_value = POLICY_VALUE_DEFAULT
        state, frontier_ids, strata, q_map = _cycle_candidates_value_state(
            state,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cnf2_cfg,
            safe_gather_policy_value=safe_gather_policy_value,
            guard_cfg=guard_cfg,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            runtime_fns=runtime_fns,
        )
    if not bool(jax.device_get(state.ledger.corrupt)):
        host_raise_if_bad_fn(state.ledger, "Ledger capacity exceeded during cycle")
    return state, frontier_ids, strata, q_map


def cycle_candidates_static(
    ledger: Ledger | LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation (static policy).

    If ``ledger`` is a LedgerState, returns a LedgerState to preserve the index.
    Otherwise returns a Ledger.
    """
    if isinstance(ledger, LedgerState):
        return cycle_candidates_static_state(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            host_raise_if_bad_fn=host_raise_if_bad_fn,
            safe_gather_policy=safe_gather_policy,
            guard_cfg=guard_cfg,
            cnf2_cfg=cnf2_cfg,
            runtime_fns=runtime_fns,
        )
    return _cycle_candidates_common(
        ledger,
        frontier_ids,
        validate_mode,
        policy_mode=PolicyMode.STATIC,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        host_raise_if_bad_fn=host_raise_if_bad_fn,
        safe_gather_policy=safe_gather_policy,
        safe_gather_policy_value=None,
        guard_cfg=guard_cfg,
        cnf2_cfg=cnf2_cfg,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_value(
    ledger: Ledger | LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation (policy as JAX value).

    If ``ledger`` is a LedgerState, returns a LedgerState to preserve the index.
    Otherwise returns a Ledger.
    """
    if isinstance(ledger, LedgerState):
        return cycle_candidates_value_state(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            host_raise_if_bad_fn=host_raise_if_bad_fn,
            safe_gather_policy_value=safe_gather_policy_value,
            guard_cfg=guard_cfg,
            cnf2_cfg=cnf2_cfg,
            runtime_fns=runtime_fns,
        )
    return _cycle_candidates_common(
        ledger,
        frontier_ids,
        validate_mode,
        policy_mode=PolicyMode.VALUE,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        host_raise_if_bad_fn=host_raise_if_bad_fn,
        safe_gather_policy=None,
        safe_gather_policy_value=safe_gather_policy_value,
        guard_cfg=guard_cfg,
        cnf2_cfg=cnf2_cfg,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_static_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation (static policy, state)."""
    return _cycle_candidates_common_state(
        state,
        frontier_ids,
        validate_mode,
        policy_mode=PolicyMode.STATIC,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        host_raise_if_bad_fn=host_raise_if_bad_fn,
        safe_gather_policy=safe_gather_policy,
        safe_gather_policy_value=None,
        guard_cfg=guard_cfg,
        cnf2_cfg=cnf2_cfg,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_value_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation (policy value, state)."""
    return _cycle_candidates_common_state(
        state,
        frontier_ids,
        validate_mode,
        policy_mode=PolicyMode.VALUE,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        host_raise_if_bad_fn=host_raise_if_bad_fn,
        safe_gather_policy=None,
        safe_gather_policy_value=safe_gather_policy_value,
        guard_cfg=guard_cfg,
        cnf2_cfg=cnf2_cfg,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation on LedgerState."""
    binding = resolve_policy_binding(
        policy=safe_gather_policy,
        policy_value=safe_gather_policy_value,
        context="cycle_candidates_state",
    )
    if binding.mode == PolicyMode.VALUE:
        return cycle_candidates_value_state(
            state,
            frontier_ids,
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            host_raise_if_bad_fn=host_raise_if_bad_fn,
            safe_gather_policy_value=require_value_policy(
                binding, context="cycle_candidates_state"
            ),
            guard_cfg=guard_cfg,
            cnf2_cfg=cnf2_cfg,
            runtime_fns=runtime_fns,
        )
    return cycle_candidates_static_state(
        state,
        frontier_ids,
        validate_mode=validate_mode,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        host_raise_if_bad_fn=host_raise_if_bad_fn,
        safe_gather_policy=require_static_policy(
            binding, context="cycle_candidates_state"
        ),
        guard_cfg=guard_cfg,
        cnf2_cfg=cnf2_cfg,
        runtime_fns=runtime_fns,
    )


def cycle_candidates(
    ledger: Ledger | LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation with DI hooks.

    Axis: Interface/Control. Commutes with q. Erased by q.
    Test: tests/test_candidate_cycle.py

    If ``ledger`` is a LedgerState, returns a LedgerState to preserve the index.
    Otherwise returns a Ledger.
    """
    if isinstance(ledger, LedgerState):
        return cycle_candidates_state(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            host_raise_if_bad_fn=host_raise_if_bad_fn,
            safe_gather_policy=safe_gather_policy,
            safe_gather_policy_value=safe_gather_policy_value,
            guard_cfg=guard_cfg,
            cnf2_cfg=cnf2_cfg,
            runtime_fns=runtime_fns,
        )
    binding = resolve_policy_binding(
        policy=safe_gather_policy,
        policy_value=safe_gather_policy_value,
        context="cycle_candidates",
    )
    if binding.mode == PolicyMode.VALUE:
        return cycle_candidates_value(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            host_raise_if_bad_fn=host_raise_if_bad_fn,
            safe_gather_policy_value=require_value_policy(
                binding, context="cycle_candidates"
            ),
            guard_cfg=guard_cfg,
            cnf2_cfg=cnf2_cfg,
            runtime_fns=runtime_fns,
        )
    return cycle_candidates_static(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        host_raise_if_bad_fn=host_raise_if_bad_fn,
        safe_gather_policy=require_static_policy(
            binding, context="cycle_candidates"
        ),
        guard_cfg=guard_cfg,
        cnf2_cfg=cnf2_cfg,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_bound(
    ledger: Ledger | LedgerState,
    frontier_ids,
    cfg: Cnf2BoundConfig,
    *,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    guard_cfg: GuardConfig | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation with required PolicyBinding.

    If ``ledger`` is a LedgerState, returns a LedgerState to preserve the index.
    Otherwise returns a Ledger.
    """
    if isinstance(ledger, LedgerState):
        return cycle_candidates_bound_state(
            ledger,
            frontier_ids,
            cfg,
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            host_raise_if_bad_fn=host_raise_if_bad_fn,
            guard_cfg=guard_cfg,
            runtime_fns=runtime_fns,
        )
    if host_raise_if_bad_fn is None:
        host_raise_if_bad_fn = _host_raise_if_bad
    base_cfg = cfg.as_cfg()
    if base_cfg.policy_binding is not None or base_cfg.safe_gather_policy is not None or base_cfg.safe_gather_policy_value is not None:
        base_cfg = replace(
            base_cfg,
            policy_binding=None,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )
    cfg = cnf2_config_bound(cfg.policy_binding, cfg=base_cfg)
    bundle = _EmitInternFns(emit_candidates_fn=emit_candidates_fn, intern_fn=intern_fn)
    ledger, frontier_ids, strata, q_map = _cycle_candidates_bound(
        ledger,
        frontier_ids,
        cfg,
        validate_mode=validate_mode,
        guard_cfg=guard_cfg,
        intern_fn=bundle.intern_fn if bundle.intern_fn is not None else _ledger_intern.intern_nodes,
        intern_cfg=intern_cfg,
        emit_candidates_fn=bundle.emit_candidates_fn if bundle.emit_candidates_fn is not None else _emit_candidates_default,
        runtime_fns=runtime_fns,
    )
    if not bool(jax.device_get(ledger.corrupt)):
        host_raise_if_bad_fn(ledger, "Ledger capacity exceeded during cycle")
    return ledger, frontier_ids, strata, q_map


def cycle_candidates_bound_state(
    state: LedgerState,
    frontier_ids,
    cfg: Cnf2BoundConfig,
    *,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    guard_cfg: GuardConfig | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Interface/Control wrapper for CNF-2 evaluation (PolicyBinding, state)."""
    if host_raise_if_bad_fn is None:
        host_raise_if_bad_fn = _host_raise_if_bad
    base_cfg = cfg.as_cfg()
    if (
        base_cfg.policy_binding is not None
        or base_cfg.safe_gather_policy is not None
        or base_cfg.safe_gather_policy_value is not None
    ):
        base_cfg = replace(
            base_cfg,
            policy_binding=None,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )
    cfg = cnf2_config_bound(cfg.policy_binding, cfg=base_cfg)
    bundle = _EmitInternFns(emit_candidates_fn=emit_candidates_fn, intern_fn=intern_fn)
    state, frontier_ids, strata, q_map = _cycle_candidates_bound_state(
        state,
        frontier_ids,
        cfg,
        validate_mode=validate_mode,
        guard_cfg=guard_cfg,
        intern_fn=bundle.intern_fn if bundle.intern_fn is not None else _ledger_intern.intern_nodes,
        intern_cfg=intern_cfg,
        emit_candidates_fn=bundle.emit_candidates_fn if bundle.emit_candidates_fn is not None else _emit_candidates_default,
        runtime_fns=runtime_fns,
    )
    if not bool(jax.device_get(state.ledger.corrupt)):
        host_raise_if_bad_fn(state.ledger, "Ledger capacity exceeded during cycle")
    return state, frontier_ids, strata, q_map
