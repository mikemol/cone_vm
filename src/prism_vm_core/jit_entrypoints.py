from __future__ import annotations

"""JIT entrypoint factories with explicit DI and static args."""

# dataflow-bundle: _arena, _root
# dataflow-bundle: _arena, _tile_size
# dataflow-bundle: _args, _kwargs

from dataclasses import dataclass, replace
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_core.errors import (
    PrismPolicyBindingError,
    PrismCnf2ModeError,
    PrismCnf2ModeConflictError,
)
from prism_core.di import bind_optional_kwargs, cached_jit, resolve
from prism_core.guards import (
    GuardConfig,
    resolve_safe_gather_fn,
    resolve_safe_gather_value_fn,
)
from prism_core.safety import (
    PolicyBinding,
    PolicyMode,
    SafetyPolicy,
    DEFAULT_SAFETY_POLICY,
    POLICY_VALUE_DEFAULT,
    resolve_policy_binding,
    require_static_policy,
    require_value_policy,
)
from prism_core.modes import ValidateMode, Cnf2Mode, coerce_cnf2_mode
from prism_ledger import intern as _ledger_intern
from prism_ledger.config import InternConfig, DEFAULT_INTERN_CONFIG
from prism_bsp.config import (
    Cnf2Config,
    Cnf2Flags,
    Cnf2RuntimeFns,
    DEFAULT_CNF2_RUNTIME_FNS,
    ArenaInteractConfig,
    ArenaCycleConfig,
    ArenaSortConfig,
    IntrinsicConfig,
    DEFAULT_ARENA_INTERACT_CONFIG,
    DEFAULT_ARENA_CYCLE_CONFIG,
    DEFAULT_ARENA_SORT_CONFIG,
    DEFAULT_INTRINSIC_CONFIG,
    SwizzleWithPermFnsBound,
)
from prism_vm_core.protocols import EmitCandidatesFn, HostRaiseFn, InternFn


@dataclass(frozen=True)
class _ArenaRootArgs:
    arena: object
    root: object


@dataclass(frozen=True)
class _ArenaTileArgs:
    arena: object
    tile_size: object


@dataclass(frozen=True)
class _ArgsKwargs:
    args: tuple
    kwargs: dict


def _safe_gather_is_bound(safe_gather_fn) -> bool:
    if safe_gather_fn is None:
        return False
    return bool(getattr(safe_gather_fn, "_prism_policy_bound", False))

from prism_vm_core.structures import NodeBatch
from prism_vm_core.candidates import _candidate_indices, candidate_indices_cfg
from prism_bsp.cnf2 import (
    emit_candidates as _emit_candidates_default,
    compact_candidates as _compact_candidates,
    compact_candidates_result as _compact_candidates_result,
    compact_candidates_with_index as _compact_candidates_with_index,
    compact_candidates_with_index_result as _compact_candidates_with_index_result,
    intern_candidates as _intern_candidates,
    cycle_candidates as _cycle_candidates_impl,
    cycle_candidates_static as _cycle_candidates_static,
    cycle_candidates_value as _cycle_candidates_value,
)
from prism_bsp.arena_step import (
    cycle_core as _cycle_core,
    cycle_value as _cycle_value,
    op_interact as _op_interact,
    op_interact_value as _op_interact_value,
)
from prism_bsp.intrinsic import _cycle_intrinsic_jit as _cycle_intrinsic_jit_impl
from prism_bsp.space import (
    _servo_update,
    op_morton,
    op_rank,
    op_sort_and_swizzle_blocked_with_perm,
    op_sort_and_swizzle_blocked_with_perm_value,
    op_sort_and_swizzle_hierarchical_with_perm,
    op_sort_and_swizzle_hierarchical_with_perm_value,
    op_sort_and_swizzle_morton_with_perm,
    op_sort_and_swizzle_morton_with_perm_value,
    op_sort_and_swizzle_servo_with_perm,
    op_sort_and_swizzle_servo_with_perm_value,
    op_sort_and_swizzle_with_perm,
    op_sort_and_swizzle_with_perm_value,
)
DEFAULT_SWIZZLE_WITH_PERM_FNS_BOUND = SwizzleWithPermFnsBound(
    with_perm=op_sort_and_swizzle_with_perm,
    morton_with_perm=op_sort_and_swizzle_morton_with_perm,
    blocked_with_perm=op_sort_and_swizzle_blocked_with_perm,
    hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm,
    servo_with_perm=op_sort_and_swizzle_servo_with_perm,
)

DEFAULT_SWIZZLE_WITH_PERM_VALUE_FNS_BOUND = SwizzleWithPermFnsBound(
    with_perm=op_sort_and_swizzle_with_perm_value,
    morton_with_perm=op_sort_and_swizzle_morton_with_perm_value,
    blocked_with_perm=op_sort_and_swizzle_blocked_with_perm_value,
    hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm_value,
    servo_with_perm=op_sort_and_swizzle_servo_with_perm_value,
)
from prism_vm_core.domains import _host_raise_if_bad
from prism_vm_core.gating import _servo_enabled
from prism_ledger.intern import _coord_norm_id_jax


def _noop_root_hash(_arena, _root):
    _ = _ArenaRootArgs(arena=_arena, root=_root)
    return jnp.int32(0)


def _noop_tile_size(*_args, **_kwargs):
    _ = _ArgsKwargs(args=_args, kwargs=_kwargs)
    return jnp.int32(0)


def _noop_metrics(_arena, _tile_size):
    _ = _ArenaTileArgs(arena=_arena, tile_size=_tile_size)
    return jnp.int32(0)


@cached_jit
def _intern_nodes_jit(cfg: InternConfig):
    def _impl(ledger, batch: NodeBatch):
        return _ledger_intern.intern_nodes(ledger, batch, cfg=cfg)

    return _impl


def intern_nodes_jit(cfg: InternConfig | None = None):
    """Return a jitted intern_nodes entrypoint for a fixed config."""
    if cfg is None:
        cfg = DEFAULT_INTERN_CONFIG
    return _intern_nodes_jit(cfg)


@cached_jit
def _op_interact_jit(safe_gather_fn, scatter_drop_fn, guard_max_fn):
    def _impl(arena):
        return _op_interact(
            arena,
            safe_gather_fn=safe_gather_fn,
            scatter_drop_fn=scatter_drop_fn,
            guard_max_fn=guard_max_fn,
        )

    return _impl


@cached_jit
def _op_interact_value_jit(safe_gather_value_fn, scatter_drop_fn, guard_max_fn):
    def _impl(arena, policy_value):
        return _op_interact_value(
            arena,
            policy_value,
            safe_gather_value_fn=safe_gather_value_fn,
            scatter_drop_fn=scatter_drop_fn,
            guard_max_fn=guard_max_fn,
        )

    return _impl


def op_interact_jit(
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    scatter_drop_fn=_jax_safe.scatter_drop,
    guard_max_fn=None,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a jitted op_interact entrypoint for fixed DI."""
    if guard_max_fn is None:
        from prism_vm_core.guards import _guard_max as guard_max_fn  # type: ignore
    safe_gather_fn = resolve_safe_gather_fn(
        safe_gather_fn=safe_gather_fn,
        policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    return _op_interact_jit(safe_gather_fn, scatter_drop_fn, guard_max_fn)


def op_interact_value_jit(
    *,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    scatter_drop_fn=_jax_safe.scatter_drop,
    guard_max_fn=None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a jitted op_interact entrypoint that accepts policy_value."""
    if guard_max_fn is None:
        from prism_vm_core.guards import _guard_max as guard_max_fn  # type: ignore
    safe_gather_value_fn = resolve_safe_gather_value_fn(
        safe_gather_value_fn=safe_gather_value_fn,
        guard_cfg=guard_cfg,
    )
    return _op_interact_value_jit(safe_gather_value_fn, scatter_drop_fn, guard_max_fn)


def op_interact_jit_cfg(
    cfg: ArenaInteractConfig | None = None,
):
    """Return a jitted op_interact entrypoint for a fixed config."""
    if cfg is None:
        cfg = DEFAULT_ARENA_INTERACT_CONFIG
    safe_gather_policy = cfg.safe_gather_policy
    safe_gather_policy_value = cfg.safe_gather_policy_value
    if cfg.policy_binding is not None:
        if safe_gather_policy is not None or safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "op_interact_jit_cfg received both policy_binding and "
                "safe_gather_policy/safe_gather_policy_value",
                context="op_interact_jit_cfg",
                policy_mode="ambiguous",
            )
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            safe_gather_policy_value = require_value_policy(
                cfg.policy_binding, context="op_interact_jit_cfg"
            )
        else:
            safe_gather_policy = require_static_policy(
                cfg.policy_binding, context="op_interact_jit_cfg"
            )
    if safe_gather_policy_value is not None:
        return op_interact_value_jit(
            safe_gather_value_fn=resolve(
                cfg.safe_gather_value_fn, _jax_safe.safe_gather_1d_value
            ),
            scatter_drop_fn=resolve(cfg.scatter_drop_fn, _jax_safe.scatter_drop),
            guard_max_fn=cfg.guard_max_fn,
            guard_cfg=cfg.guard_cfg,
        )
    policy_bound = _safe_gather_is_bound(cfg.safe_gather_fn)
    if safe_gather_policy is None and policy_bound:
        return op_interact_jit(
            safe_gather_fn=resolve(cfg.safe_gather_fn, _jax_safe.safe_gather_1d),
            scatter_drop_fn=resolve(cfg.scatter_drop_fn, _jax_safe.scatter_drop),
            guard_max_fn=cfg.guard_max_fn,
            safe_gather_policy=None,
            guard_cfg=cfg.guard_cfg,
        )
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    return op_interact_jit(
        safe_gather_fn=resolve(cfg.safe_gather_fn, _jax_safe.safe_gather_1d),
        scatter_drop_fn=resolve(cfg.scatter_drop_fn, _jax_safe.scatter_drop),
        guard_max_fn=cfg.guard_max_fn,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=cfg.guard_cfg,
    )


def op_interact_jit_bound_cfg(
    policy_binding: PolicyBinding,
    *,
    cfg: ArenaInteractConfig | None = None,
):
    """Return a jitted op_interact entrypoint with required PolicyBinding."""
    if cfg is None:
        cfg = ArenaInteractConfig(policy_binding=policy_binding)
    else:
        cfg = replace(
            cfg,
            policy_binding=policy_binding,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )
    return op_interact_jit_cfg(cfg)


def emit_candidates_jit(emit_candidates_fn: EmitCandidatesFn | None = None):
    """Return a jitted emit_candidates entrypoint."""
    if emit_candidates_fn is None:
        emit_candidates_fn = _emit_candidates_default

    @jax.jit
    def _impl(ledger, frontier_ids):
        return emit_candidates_fn(ledger, frontier_ids)

    return _impl


def emit_candidates_jit_cfg(cfg: Cnf2Config | None = None):
    """Return a jitted emit_candidates entrypoint for a fixed config."""
    emit_candidates_fn = None
    if cfg is not None:
        emit_candidates_fn = cfg.emit_candidates_fn
    return emit_candidates_jit(emit_candidates_fn=emit_candidates_fn)


def compact_candidates_jit(*, candidate_indices_fn=_candidate_indices):
    """Return a jitted compact_candidates entrypoint for fixed DI."""
    @jax.jit
    def _impl(candidates):
        return _compact_candidates(
            candidates, candidate_indices_fn=candidate_indices_fn
        )

    return _impl


def compact_candidates_result_jit(*, candidate_indices_fn=_candidate_indices):
    """Return a jitted compact_candidates_result entrypoint for fixed DI."""
    @jax.jit
    def _impl(candidates):
        return _compact_candidates_result(
            candidates, candidate_indices_fn=candidate_indices_fn
        )

    return _impl


def compact_candidates_jit_cfg(cfg: Cnf2Config | None = None):
    """Return a jitted compact_candidates entrypoint for a fixed config."""
    candidate_indices_fn = _candidate_indices
    if cfg is not None and cfg.candidate_indices_fn is not None:
        candidate_indices_fn = cfg.candidate_indices_fn
    if cfg is not None and cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
        candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
    return compact_candidates_jit(candidate_indices_fn=candidate_indices_fn)


def compact_candidates_result_jit_cfg(cfg: Cnf2Config | None = None):
    """Return a jitted compact_candidates_result entrypoint for a fixed config."""
    candidate_indices_fn = _candidate_indices
    if cfg is not None and cfg.candidate_indices_fn is not None:
        candidate_indices_fn = cfg.candidate_indices_fn
    if cfg is not None and cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
        candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
    return compact_candidates_result_jit(candidate_indices_fn=candidate_indices_fn)


def compact_candidates_with_index_jit(*, candidate_indices_fn=_candidate_indices):
    """Return a jitted compact_candidates_with_index entrypoint for fixed DI."""
    @jax.jit
    def _impl(candidates):
        return _compact_candidates_with_index(
            candidates, candidate_indices_fn=candidate_indices_fn
        )

    return _impl


def compact_candidates_with_index_result_jit(*, candidate_indices_fn=_candidate_indices):
    """Return a jitted compact_candidates_with_index_result entrypoint for fixed DI."""
    @jax.jit
    def _impl(candidates):
        return _compact_candidates_with_index_result(
            candidates, candidate_indices_fn=candidate_indices_fn
        )

    return _impl


def compact_candidates_with_index_jit_cfg(cfg: Cnf2Config | None = None):
    """Return a jitted compact_candidates_with_index entrypoint for a fixed config."""
    candidate_indices_fn = _candidate_indices
    if cfg is not None and cfg.candidate_indices_fn is not None:
        candidate_indices_fn = cfg.candidate_indices_fn
    if cfg is not None and cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
        candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
    return compact_candidates_with_index_jit(candidate_indices_fn=candidate_indices_fn)


def compact_candidates_with_index_result_jit_cfg(cfg: Cnf2Config | None = None):
    """Return a jitted compact_candidates_with_index_result entrypoint for a fixed config."""
    candidate_indices_fn = _candidate_indices
    if cfg is not None and cfg.candidate_indices_fn is not None:
        candidate_indices_fn = cfg.candidate_indices_fn
    if cfg is not None and cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
        candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
    return compact_candidates_with_index_result_jit(
        candidate_indices_fn=candidate_indices_fn
    )


def intern_candidates_jit(
    *,
    compact_candidates_fn=_compact_candidates,
    intern_fn: InternFn = _ledger_intern.intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn=None,
):
    """Return a jitted intern_candidates entrypoint for fixed DI."""
    if intern_cfg is not None and intern_fn is _ledger_intern.intern_nodes:
        intern_fn = partial(_ledger_intern.intern_nodes, cfg=intern_cfg)
    if node_batch_fn is None:
        node_batch_fn = NodeBatch

    @jax.jit
    def _impl(ledger, candidates):
        return _intern_candidates(
            ledger,
            candidates,
            compact_candidates_fn=compact_candidates_fn,
            intern_fn=intern_fn,
            intern_cfg=None,
            node_batch_fn=node_batch_fn,
        )

    return _impl


def intern_candidates_jit_cfg(
    cfg: Cnf2Config | None = None,
    *,
    compact_candidates_fn=_compact_candidates,
    intern_fn: InternFn = _ledger_intern.intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn=None,
):
    """Return a jitted intern_candidates entrypoint for a fixed config."""
    if cfg is not None:
        if intern_cfg is None:
            intern_cfg = cfg.intern_cfg
        if intern_fn is _ledger_intern.intern_nodes and cfg.intern_fn is not None:
            intern_fn = cfg.intern_fn
        if node_batch_fn is None and cfg.node_batch_fn is not None:
            node_batch_fn = cfg.node_batch_fn
        if cfg.candidate_indices_fn is not None:
            compact_candidates_fn = partial(
                _compact_candidates, candidate_indices_fn=cfg.candidate_indices_fn
            )
    return intern_candidates_jit(
        compact_candidates_fn=compact_candidates_fn,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
    )


def cycle_candidates_static_jit(
    *,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    cnf2_flags: Cnf2Flags | None = None,
    cnf2_mode: Cnf2Mode | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Return a jitted cycle_candidates entrypoint (static policy)."""
    if cnf2_mode is not None and not isinstance(cnf2_mode, Cnf2Mode):
        raise PrismCnf2ModeError(mode=cnf2_mode)

    if intern_fn is None:
        intern_fn = _ledger_intern.intern_nodes
    if cnf2_cfg is not None and cnf2_flags is not None:
        cnf2_cfg = replace(cnf2_cfg, flags=cnf2_flags)
    elif cnf2_cfg is None and cnf2_flags is not None:
        cnf2_cfg = Cnf2Config(flags=cnf2_flags)
    if cnf2_cfg is not None:
        if cnf2_mode is None and cnf2_cfg.cnf2_mode is not None:
            cnf2_mode = cnf2_cfg.cnf2_mode
        elif cnf2_mode is not None and cnf2_cfg.cnf2_mode is not None:
            mode_a = coerce_cnf2_mode(cnf2_mode, context="cycle_candidates_static_jit")
            mode_b = coerce_cnf2_mode(cnf2_cfg.cnf2_mode, context="cycle_candidates_static_jit")
            if mode_a != mode_b:
                raise PrismCnf2ModeConflictError(
                    "cycle_candidates_static_jit received both cnf2_mode and cfg.cnf2_mode",
                    context="cycle_candidates_static_jit",
                )
        if intern_cfg is None:
            intern_cfg = cnf2_cfg.intern_cfg
        if intern_fn is None and cnf2_cfg.intern_fn is not None:
            intern_fn = cnf2_cfg.intern_fn
        if emit_candidates_fn is None and cnf2_cfg.emit_candidates_fn is not None:
            emit_candidates_fn = cnf2_cfg.emit_candidates_fn
        if cnf2_cfg.policy_binding is not None:
            if cnf2_cfg.policy_binding.mode == PolicyMode.VALUE:
                raise PrismPolicyBindingError(
                    "cycle_candidates_static_jit received cfg.policy_binding value-mode; "
                    "use cycle_candidates_value_jit",
                    context="cycle_candidates_static_jit",
                    policy_mode="static",
                )
            if safe_gather_policy is None:
                safe_gather_policy = require_static_policy(
                    cnf2_cfg.policy_binding, context="cycle_candidates_static_jit"
                )
        if safe_gather_policy is None and cnf2_cfg.safe_gather_policy is not None:
            safe_gather_policy = cnf2_cfg.safe_gather_policy
        if (
            cnf2_cfg.safe_gather_policy_value is not None
        ):
            raise PrismPolicyBindingError(
                "cycle_candidates_static_jit received cfg.safe_gather_policy_value; "
                "use cycle_candidates_value_jit",
                context="cycle_candidates_static_jit",
                policy_mode="static",
            )
        if guard_cfg is None and cnf2_cfg.guard_cfg is not None:
            guard_cfg = cnf2_cfg.guard_cfg
        if runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
            runtime_fns = cnf2_cfg.runtime_fns
        cnf2_flags = cnf2_cfg.flags if cnf2_flags is None else cnf2_flags
    if cnf2_mode is not None:
        mode = coerce_cnf2_mode(cnf2_mode, context="cycle_candidates_static_jit")
        if mode != Cnf2Mode.AUTO:
            if cnf2_flags is not None or runtime_fns is not DEFAULT_CNF2_RUNTIME_FNS:
                raise PrismCnf2ModeConflictError(
                    "cycle_candidates_static_jit received cnf2_mode alongside cnf2_flags or runtime_fns overrides",
                    context="cycle_candidates_static_jit",
                )
            enabled_value = mode in (Cnf2Mode.BASE, Cnf2Mode.SLOT1)
            slot1_value = mode == Cnf2Mode.SLOT1
            runtime_fns = replace(
                runtime_fns,
                cnf2_enabled_fn=lambda: enabled_value,
                cnf2_slot1_enabled_fn=lambda: slot1_value,
            )
            cnf2_flags = None
    if cnf2_flags is not None:
        if cnf2_flags.enabled is not None:
            enabled_value = bool(cnf2_flags.enabled)
            runtime_fns = replace(
                runtime_fns, cnf2_enabled_fn=lambda: enabled_value
            )
        if cnf2_flags.slot1_enabled is not None:
            slot1_value = bool(cnf2_flags.slot1_enabled)
            runtime_fns = replace(
                runtime_fns, cnf2_slot1_enabled_fn=lambda: slot1_value
            )
    if emit_candidates_fn is None:
        emit_candidates_fn = _emit_candidates_default
    if host_raise_if_bad_fn is None:
        host_raise_if_bad_fn = _host_raise_if_bad
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY

    @jax.jit
    def _impl(ledger, frontier_ids):
        return _cycle_candidates_static(
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

    def _run(ledger, frontier_ids):
        out = _impl(ledger, frontier_ids)
        out_ledger = out[0]
        if not bool(jax.device_get(out_ledger.corrupt)):
            host_raise_if_bad_fn(out_ledger, "Ledger capacity exceeded during cycle")
        return out

    return _run


def cycle_candidates_value_jit(
    *,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy_value: jnp.ndarray | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    cnf2_flags: Cnf2Flags | None = None,
    cnf2_mode: Cnf2Mode | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Return a jitted cycle_candidates entrypoint (policy as JAX value)."""
    if cnf2_mode is not None and not isinstance(cnf2_mode, Cnf2Mode):
        raise PrismCnf2ModeError(mode=cnf2_mode)

    if intern_fn is None:
        intern_fn = _ledger_intern.intern_nodes
    if cnf2_cfg is not None and cnf2_flags is not None:
        cnf2_cfg = replace(cnf2_cfg, flags=cnf2_flags)
    elif cnf2_cfg is None and cnf2_flags is not None:
        cnf2_cfg = Cnf2Config(flags=cnf2_flags)
    if cnf2_cfg is not None:
        if cnf2_mode is None and cnf2_cfg.cnf2_mode is not None:
            cnf2_mode = cnf2_cfg.cnf2_mode
        elif cnf2_mode is not None and cnf2_cfg.cnf2_mode is not None:
            mode_a = coerce_cnf2_mode(cnf2_mode, context="cycle_candidates_value_jit")
            mode_b = coerce_cnf2_mode(cnf2_cfg.cnf2_mode, context="cycle_candidates_value_jit")
            if mode_a != mode_b:
                raise PrismCnf2ModeConflictError(
                    "cycle_candidates_value_jit received both cnf2_mode and cfg.cnf2_mode",
                    context="cycle_candidates_value_jit",
                )
        if intern_cfg is None:
            intern_cfg = cnf2_cfg.intern_cfg
        if intern_fn is None and cnf2_cfg.intern_fn is not None:
            intern_fn = cnf2_cfg.intern_fn
        if emit_candidates_fn is None and cnf2_cfg.emit_candidates_fn is not None:
            emit_candidates_fn = cnf2_cfg.emit_candidates_fn
        if cnf2_cfg.policy_binding is not None:
            if cnf2_cfg.policy_binding.mode == PolicyMode.STATIC:
                raise PrismPolicyBindingError(
                    "cycle_candidates_value_jit received cfg.policy_binding static-mode; "
                    "use cycle_candidates_static_jit",
                    context="cycle_candidates_value_jit",
                    policy_mode="value",
                )
            if safe_gather_policy_value is None:
                safe_gather_policy_value = require_value_policy(
                    cnf2_cfg.policy_binding, context="cycle_candidates_value_jit"
                )
        if cnf2_cfg.safe_gather_policy is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_value_jit received cfg.safe_gather_policy; "
                "use cycle_candidates_static_jit",
                context="cycle_candidates_value_jit",
                policy_mode="value",
            )
        if (
            safe_gather_policy_value is None
            and cnf2_cfg.safe_gather_policy_value is not None
        ):
            safe_gather_policy_value = cnf2_cfg.safe_gather_policy_value
        if guard_cfg is None and cnf2_cfg.guard_cfg is not None:
            guard_cfg = cnf2_cfg.guard_cfg
        if runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
            runtime_fns = cnf2_cfg.runtime_fns
        cnf2_flags = cnf2_cfg.flags if cnf2_flags is None else cnf2_flags
    if cnf2_mode is not None:
        mode = coerce_cnf2_mode(cnf2_mode, context="cycle_candidates_value_jit")
        if mode != Cnf2Mode.AUTO:
            if cnf2_flags is not None or runtime_fns is not DEFAULT_CNF2_RUNTIME_FNS:
                raise PrismCnf2ModeConflictError(
                    "cycle_candidates_value_jit received cnf2_mode alongside cnf2_flags or runtime_fns overrides",
                    context="cycle_candidates_value_jit",
                )
            enabled_value = mode in (Cnf2Mode.BASE, Cnf2Mode.SLOT1)
            slot1_value = mode == Cnf2Mode.SLOT1
            runtime_fns = replace(
                runtime_fns,
                cnf2_enabled_fn=lambda: enabled_value,
                cnf2_slot1_enabled_fn=lambda: slot1_value,
            )
            cnf2_flags = None
    if cnf2_flags is not None:
        if cnf2_flags.enabled is not None:
            enabled_value = bool(cnf2_flags.enabled)
            runtime_fns = replace(
                runtime_fns, cnf2_enabled_fn=lambda: enabled_value
            )
        if cnf2_flags.slot1_enabled is not None:
            slot1_value = bool(cnf2_flags.slot1_enabled)
            runtime_fns = replace(
                runtime_fns, cnf2_slot1_enabled_fn=lambda: slot1_value
            )
    if emit_candidates_fn is None:
        emit_candidates_fn = _emit_candidates_default
    if host_raise_if_bad_fn is None:
        host_raise_if_bad_fn = _host_raise_if_bad
    if safe_gather_policy_value is None:
        safe_gather_policy_value = POLICY_VALUE_DEFAULT

    @jax.jit
    def _impl(ledger, frontier_ids):
        return _cycle_candidates_value(
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

    def _run(ledger, frontier_ids):
        out = _impl(ledger, frontier_ids)
        out_ledger = out[0]
        if not bool(jax.device_get(out_ledger.corrupt)):
            host_raise_if_bad_fn(out_ledger, "Ledger capacity exceeded during cycle")
        return out

    return _run


def cycle_candidates_jit(
    *,
    validate_mode: ValidateMode = ValidateMode.NONE,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    emit_candidates_fn: EmitCandidatesFn | None = None,
    host_raise_if_bad_fn: HostRaiseFn | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: jnp.ndarray | None = None,
    guard_cfg: GuardConfig | None = None,
    cnf2_cfg: Cnf2Config | None = None,
    cnf2_flags: Cnf2Flags | None = None,
    cnf2_mode: Cnf2Mode | None = None,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """Return a jitted cycle_candidates entrypoint for fixed DI."""
    if cnf2_mode is not None and not isinstance(cnf2_mode, Cnf2Mode):
        raise PrismCnf2ModeError(mode=cnf2_mode)
    binding = resolve_policy_binding(
        policy=safe_gather_policy,
        policy_value=safe_gather_policy_value,
        context="cycle_candidates_jit",
    )
    if binding.mode == PolicyMode.VALUE:
        return cycle_candidates_value_jit(
            validate_mode=validate_mode,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            emit_candidates_fn=emit_candidates_fn,
            host_raise_if_bad_fn=host_raise_if_bad_fn,
            safe_gather_policy_value=require_value_policy(
                binding, context="cycle_candidates_jit"
            ),
            guard_cfg=guard_cfg,
            cnf2_cfg=cnf2_cfg,
            cnf2_flags=cnf2_flags,
            cnf2_mode=cnf2_mode,
            runtime_fns=runtime_fns,
        )
    return cycle_candidates_static_jit(
        validate_mode=validate_mode,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        emit_candidates_fn=emit_candidates_fn,
        host_raise_if_bad_fn=host_raise_if_bad_fn,
        safe_gather_policy=require_static_policy(
            binding, context="cycle_candidates_jit"
        ),
        guard_cfg=guard_cfg,
        cnf2_cfg=cnf2_cfg,
        cnf2_flags=cnf2_flags,
        cnf2_mode=cnf2_mode,
        runtime_fns=runtime_fns,
    )
@cached_jit
def _cycle_jit(
    sort_cfg: ArenaSortConfig,
    op_rank_fn,
    servo_enabled_value,
    servo_update_fn,
    op_morton_fn,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound,
    safe_gather_fn,
    guard_cfg,
    op_interact_fn,
):
    cycle_core_fn = bind_optional_kwargs(_cycle_core, guard_cfg=guard_cfg)

    def _impl(arena, root_ptr):
        return cycle_core_fn(
            arena,
            root_ptr,
            sort_cfg=sort_cfg,
            op_rank_fn=op_rank_fn,
            servo_enabled_fn=lambda: servo_enabled_value,
            servo_update_fn=servo_update_fn,
            op_morton_fn=op_morton_fn,
            swizzle_with_perm_fns=swizzle_with_perm_fns,
            safe_gather_fn=safe_gather_fn,
            arena_root_hash_fn=_noop_root_hash,
            damage_tile_size_fn=_noop_tile_size,
            damage_metrics_update_fn=_noop_metrics,
            op_interact_fn=op_interact_fn,
        )

    return _impl


def cycle_jit(
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound = DEFAULT_SWIZZLE_WITH_PERM_FNS_BOUND,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    op_interact_fn=_op_interact,
    test_guards: bool = False,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a jitted cycle entrypoint for fixed DI."""
    if test_guards:
        raise RuntimeError("cycle_jit is disabled under TEST_GUARDS")
    safe_gather_fn = resolve_safe_gather_fn(
        safe_gather_fn=safe_gather_fn,
        policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    if sort_cfg.morton is not None:
        raise ValueError("cycle_jit requires sort_cfg.morton is None")
    servo_enabled_value = bool(servo_enabled_fn())
    return _cycle_jit(
        sort_cfg,
        op_rank_fn,
        servo_enabled_value,
        servo_update_fn,
        op_morton_fn,
        swizzle_with_perm_fns,
        safe_gather_fn,
        guard_cfg,
        op_interact_fn,
    )


@cached_jit
def _cycle_value_jit(
    sort_cfg: ArenaSortConfig,
    op_rank_fn,
    servo_enabled_value,
    servo_update_fn,
    op_morton_fn,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound,
    safe_gather_value_fn,
    guard_cfg,
):
    cycle_value_fn = bind_optional_kwargs(_cycle_value, guard_cfg=guard_cfg)

    def _impl(arena, root_ptr, policy_value):
        return cycle_value_fn(
            arena,
            root_ptr,
            policy_value,
            sort_cfg=sort_cfg,
            op_rank_fn=op_rank_fn,
            servo_enabled_fn=lambda: servo_enabled_value,
            servo_update_fn=servo_update_fn,
            op_morton_fn=op_morton_fn,
            swizzle_with_perm_fns=swizzle_with_perm_fns,
            safe_gather_value_fn=safe_gather_value_fn,
            arena_root_hash_fn=_noop_root_hash,
            damage_tile_size_fn=_noop_tile_size,
            damage_metrics_update_fn=_noop_metrics,
        )

    return _impl


def cycle_value_jit(
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound = DEFAULT_SWIZZLE_WITH_PERM_VALUE_FNS_BOUND,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    guard_cfg: GuardConfig | None = None,
):
    """Return a jitted cycle entrypoint that accepts policy_value."""
    if sort_cfg.morton is not None:
        raise ValueError("cycle_value_jit requires sort_cfg.morton is None")
    servo_enabled_value = bool(servo_enabled_fn())
    return _cycle_value_jit(
        sort_cfg,
        op_rank_fn,
        servo_enabled_value,
        servo_update_fn,
        op_morton_fn,
        swizzle_with_perm_fns,
        safe_gather_value_fn,
        guard_cfg,
    )


def cycle_jit_cfg(
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    cfg: ArenaCycleConfig | None = None,
    test_guards: bool = False,
):
    """Return a jitted cycle entrypoint for a fixed config."""
    if cfg is None:
        cfg = DEFAULT_ARENA_CYCLE_CONFIG
    safe_gather_policy = cfg.safe_gather_policy
    safe_gather_policy_value = cfg.safe_gather_policy_value
    if cfg.policy_binding is not None:
        if safe_gather_policy is not None or safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "cycle_jit_cfg received both policy_binding and "
                "safe_gather_policy/safe_gather_policy_value",
                context="cycle_jit_cfg",
                policy_mode="ambiguous",
            )
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            safe_gather_policy_value = require_value_policy(
                cfg.policy_binding, context="cycle_jit_cfg"
            )
        else:
            safe_gather_policy = require_static_policy(
                cfg.policy_binding, context="cycle_jit_cfg"
            )
    if safe_gather_policy_value is not None:
        swizzle_with_perm_fns = SwizzleWithPermFnsBound(
            with_perm=op_sort_and_swizzle_with_perm_fn,
            morton_with_perm=op_sort_and_swizzle_morton_with_perm_fn,
            blocked_with_perm=op_sort_and_swizzle_blocked_with_perm_fn,
            hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm_fn,
            servo_with_perm=op_sort_and_swizzle_servo_with_perm_fn,
        )
        return cycle_value_jit(
            sort_cfg=sort_cfg,
            op_rank_fn=cfg.op_rank_fn or op_rank,
            servo_enabled_fn=cfg.servo_enabled_fn or _servo_enabled,
            servo_update_fn=cfg.servo_update_fn or _servo_update,
            op_morton_fn=cfg.op_morton_fn or op_morton,
            swizzle_with_perm_fns=swizzle_with_perm_fns,
            safe_gather_value_fn=resolve(
                cfg.safe_gather_value_fn, _jax_safe.safe_gather_1d_value
            ),
            guard_cfg=cfg.guard_cfg,
        )
    policy_bound = _safe_gather_is_bound(cfg.safe_gather_fn)
    if safe_gather_policy is None and not policy_bound:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    op_rank_fn = cfg.op_rank_fn or op_rank
    servo_enabled_fn = cfg.servo_enabled_fn or _servo_enabled
    servo_update_fn = cfg.servo_update_fn or _servo_update
    op_morton_fn = cfg.op_morton_fn or op_morton
    op_sort_and_swizzle_with_perm_fn = (
        cfg.op_sort_and_swizzle_with_perm_fn or op_sort_and_swizzle_with_perm
    )
    op_sort_and_swizzle_morton_with_perm_fn = (
        cfg.op_sort_and_swizzle_morton_with_perm_fn
        or op_sort_and_swizzle_morton_with_perm
    )
    op_sort_and_swizzle_blocked_with_perm_fn = (
        cfg.op_sort_and_swizzle_blocked_with_perm_fn
        or op_sort_and_swizzle_blocked_with_perm
    )
    op_sort_and_swizzle_hierarchical_with_perm_fn = (
        cfg.op_sort_and_swizzle_hierarchical_with_perm_fn
        or op_sort_and_swizzle_hierarchical_with_perm
    )
    op_sort_and_swizzle_servo_with_perm_fn = (
        cfg.op_sort_and_swizzle_servo_with_perm_fn
        or op_sort_and_swizzle_servo_with_perm
    )
    swizzle_with_perm_fns = SwizzleWithPermFnsBound(
        with_perm=op_sort_and_swizzle_with_perm_fn,
        morton_with_perm=op_sort_and_swizzle_morton_with_perm_fn,
        blocked_with_perm=op_sort_and_swizzle_blocked_with_perm_fn,
        hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm_fn,
        servo_with_perm=op_sort_and_swizzle_servo_with_perm_fn,
    )
    safe_gather_fn = cfg.safe_gather_fn or _jax_safe.safe_gather_1d
    op_interact_fn = cfg.op_interact_fn
    if op_interact_fn is None and cfg.interact_cfg is not None:
        op_interact_fn = op_interact_jit_cfg(cfg.interact_cfg)
    if op_interact_fn is None:
        if policy_bound or safe_gather_policy is not None or cfg.guard_cfg is not None:
            op_interact_fn = op_interact_jit(
                safe_gather_fn=safe_gather_fn,
                safe_gather_policy=safe_gather_policy,
                guard_cfg=cfg.guard_cfg,
            )
        else:
            op_interact_fn = _op_interact
    return cycle_jit(
        sort_cfg=sort_cfg,
        op_rank_fn=op_rank_fn,
        servo_enabled_fn=servo_enabled_fn,
        servo_update_fn=servo_update_fn,
        op_morton_fn=op_morton_fn,
        swizzle_with_perm_fns=swizzle_with_perm_fns,
        safe_gather_fn=safe_gather_fn,
        op_interact_fn=op_interact_fn,
        test_guards=test_guards,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=cfg.guard_cfg,
    )


def cycle_jit_bound_cfg(
    policy_binding: PolicyBinding,
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    cfg: ArenaCycleConfig | None = None,
    test_guards: bool = False,
):
    """Return a jitted cycle entrypoint with required PolicyBinding."""
    interact_cfg = None
    if cfg is None:
        cfg = ArenaCycleConfig(policy_binding=policy_binding)
    else:
        if cfg.interact_cfg is not None:
            interact_cfg = replace(
                cfg.interact_cfg,
                policy_binding=policy_binding,
                safe_gather_policy=None,
                safe_gather_policy_value=None,
            )
        cfg = replace(
            cfg,
            policy_binding=policy_binding,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
            interact_cfg=interact_cfg,
        )
    return cycle_jit_cfg(
        sort_cfg=sort_cfg,
        cfg=cfg,
        test_guards=test_guards,
    )


def cycle_intrinsic_jit(
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    node_batch_fn=None,
):
    """Return a jitted intrinsic cycle entrypoint for fixed DI."""
    if intern_fn is None:
        intern_fn = _ledger_intern.intern_nodes
    if intern_cfg is not None and intern_fn is _ledger_intern.intern_nodes:
        intern_fn = partial(_ledger_intern.intern_nodes, cfg=intern_cfg)
    if node_batch_fn is None:
        node_batch_fn = NodeBatch
    return _cycle_intrinsic_jit_impl(intern_fn, node_batch_fn)


def cycle_intrinsic_jit_cfg(
    cfg: IntrinsicConfig | None = None,
    *,
    intern_fn: InternFn | None = None,
    intern_cfg: InternConfig | None = None,
    node_batch_fn=None,
):
    """Return a jitted intrinsic entrypoint for a fixed config."""
    if cfg is None:
        cfg = DEFAULT_INTRINSIC_CONFIG
    intern_fn = cfg.intern_fn or intern_fn
    node_batch_fn = cfg.node_batch_fn or node_batch_fn
    if cfg.intern_cfg is not None and intern_cfg is not None:
        raise ValueError("Pass either cfg.intern_cfg or intern_cfg, not both.")
    intern_cfg = intern_cfg if intern_cfg is not None else cfg.intern_cfg
    return cycle_intrinsic_jit(
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
    )


def coord_norm_batch_jit(coord_norm_id_jax_fn=None):
    """Return a jitted coord_norm_batch entrypoint for fixed DI."""
    if coord_norm_id_jax_fn is None:
        coord_norm_id_jax_fn = _coord_norm_id_jax
    from prism_coord.coord import _coord_norm_batch_jit as _coord_norm_batch_jit_impl
    return _coord_norm_batch_jit_impl(coord_norm_id_jax_fn)


__all__ = [
    "intern_nodes_jit",
    "op_interact_jit",
    "op_interact_jit_cfg",
    "op_interact_jit_bound_cfg",
    "op_interact_value_jit",
    "emit_candidates_jit",
    "emit_candidates_jit_cfg",
    "compact_candidates_jit",
    "compact_candidates_jit_cfg",
    "compact_candidates_result_jit",
    "compact_candidates_result_jit_cfg",
    "compact_candidates_with_index_jit",
    "compact_candidates_with_index_jit_cfg",
    "compact_candidates_with_index_result_jit",
    "compact_candidates_with_index_result_jit_cfg",
    "intern_candidates_jit",
    "intern_candidates_jit_cfg",
    "cycle_candidates_jit",
    "cycle_candidates_static_jit",
    "cycle_candidates_value_jit",
    "cycle_jit",
    "cycle_jit_cfg",
    "cycle_jit_bound_cfg",
    "cycle_value_jit",
    "cycle_intrinsic_jit",
    "cycle_intrinsic_jit_cfg",
    "coord_norm_batch_jit",
]
