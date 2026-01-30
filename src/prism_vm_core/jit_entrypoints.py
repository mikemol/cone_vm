from __future__ import annotations

"""JIT entrypoint factories with explicit DI and static args."""

from dataclasses import replace
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_core.di import bind_optional_kwargs, cached_jit, resolve, wrap_policy
from prism_core.guards import GuardConfig, make_safe_gather_fn
from prism_core.safety import SafetyPolicy
from prism_ledger import intern as _ledger_intern
from prism_ledger.config import InternConfig, DEFAULT_INTERN_CONFIG
from prism_bsp.config import (
    Cnf2Config,
    Cnf2Flags,
    ArenaInteractConfig,
    ArenaCycleConfig,
    IntrinsicConfig,
    DEFAULT_ARENA_INTERACT_CONFIG,
    DEFAULT_ARENA_CYCLE_CONFIG,
    DEFAULT_INTRINSIC_CONFIG,
)
from prism_vm_core.protocols import EmitCandidatesFn, HostRaiseFn, InternFn
from prism_vm_core.structures import NodeBatch
from prism_vm_core.candidates import _candidate_indices, candidate_indices_cfg
from prism_bsp.cnf2 import (
    emit_candidates as _emit_candidates_default,
    compact_candidates as _compact_candidates,
    compact_candidates_with_index as _compact_candidates_with_index,
    intern_candidates as _intern_candidates,
)
from prism_bsp.arena_step import cycle_core as _cycle_core, op_interact as _op_interact
from prism_bsp.intrinsic import _cycle_intrinsic_jit as _cycle_intrinsic_jit_impl
from prism_bsp.space import (
    _servo_update,
    op_morton,
    op_rank,
    op_sort_and_swizzle_blocked_with_perm,
    op_sort_and_swizzle_hierarchical_with_perm,
    op_sort_and_swizzle_morton_with_perm,
    op_sort_and_swizzle_servo_with_perm,
    op_sort_and_swizzle_with_perm,
)
from prism_vm_core.domains import _host_raise_if_bad
from prism_vm_core.gating import (
    _cnf2_enabled as _cnf2_enabled_default,
    _cnf2_slot1_enabled as _cnf2_slot1_enabled_default,
    _servo_enabled,
)
from prism_ledger.intern import _coord_norm_id_jax
from prism_bsp.cnf2 import cycle_candidates as _cycle_candidates_impl


def _noop_root_hash(_arena, _root):
    return jnp.int32(0)


def _noop_tile_size(*_args, **_kwargs):
    return jnp.int32(0)


def _noop_metrics(_arena, _tile_size):
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
    if guard_cfg is not None:
        safe_gather_fn = make_safe_gather_fn(
            cfg=guard_cfg,
            policy=safe_gather_policy,
            safe_gather_fn=safe_gather_fn,
        )
    else:
        safe_gather_fn = wrap_policy(safe_gather_fn, safe_gather_policy)
    return _op_interact_jit(safe_gather_fn, scatter_drop_fn, guard_max_fn)


def op_interact_jit_cfg(
    cfg: ArenaInteractConfig | None = None,
):
    """Return a jitted op_interact entrypoint for a fixed config."""
    if cfg is None:
        cfg = DEFAULT_ARENA_INTERACT_CONFIG
    return op_interact_jit(
        safe_gather_fn=resolve(cfg.safe_gather_fn, _jax_safe.safe_gather_1d),
        scatter_drop_fn=resolve(cfg.scatter_drop_fn, _jax_safe.scatter_drop),
        guard_max_fn=cfg.guard_max_fn,
        safe_gather_policy=cfg.safe_gather_policy,
        guard_cfg=cfg.guard_cfg,
    )


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


def compact_candidates_jit_cfg(cfg: Cnf2Config | None = None):
    """Return a jitted compact_candidates entrypoint for a fixed config."""
    candidate_indices_fn = _candidate_indices
    if cfg is not None and cfg.candidate_indices_fn is not None:
        candidate_indices_fn = cfg.candidate_indices_fn
    if cfg is not None and cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
        candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
    return compact_candidates_jit(candidate_indices_fn=candidate_indices_fn)


def compact_candidates_with_index_jit(*, candidate_indices_fn=_candidate_indices):
    """Return a jitted compact_candidates_with_index entrypoint for fixed DI."""
    @jax.jit
    def _impl(candidates):
        return _compact_candidates_with_index(
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


def cycle_candidates_jit(
    *,
    validate_stratum: bool = False,
    validate_mode: str = "strict",
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
    """Return a jitted cycle_candidates entrypoint for fixed DI."""
    def _resolve_gate(flag_value, fn_value, default_fn):
        if flag_value is not None:
            return bool(flag_value)
        if fn_value is not None:
            return bool(fn_value())
        return bool(default_fn())

    if intern_fn is None:
        intern_fn = _ledger_intern.intern_nodes
    if cnf2_cfg is not None and cnf2_flags is not None:
        cnf2_cfg = replace(cnf2_cfg, flags=cnf2_flags)
    elif cnf2_cfg is None and cnf2_flags is not None:
        cnf2_cfg = Cnf2Config(flags=cnf2_flags)
    if cnf2_cfg is not None:
        if intern_cfg is None:
            intern_cfg = cnf2_cfg.intern_cfg
        if intern_fn is None and cnf2_cfg.intern_fn is not None:
            intern_fn = cnf2_cfg.intern_fn
        if emit_candidates_fn is None and cnf2_cfg.emit_candidates_fn is not None:
            emit_candidates_fn = cnf2_cfg.emit_candidates_fn
        if safe_gather_policy is None and cnf2_cfg.safe_gather_policy is not None:
            safe_gather_policy = cnf2_cfg.safe_gather_policy
        if cnf2_enabled_fn is None and cnf2_cfg.cnf2_enabled_fn is not None:
            cnf2_enabled_fn = cnf2_cfg.cnf2_enabled_fn
        if cnf2_slot1_enabled_fn is None and cnf2_cfg.cnf2_slot1_enabled_fn is not None:
            cnf2_slot1_enabled_fn = cnf2_cfg.cnf2_slot1_enabled_fn
        cnf2_flags = cnf2_cfg.flags if cnf2_flags is None else cnf2_flags
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

    @jax.jit
    def _impl(ledger, frontier_ids):
        return _cycle_candidates_impl(
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

    def _run(ledger, frontier_ids):
        out = _impl(ledger, frontier_ids)
        out_ledger = out[0]
        if not bool(jax.device_get(out_ledger.corrupt)):
            host_raise_if_bad_fn(out_ledger, "Ledger capacity exceeded during cycle")
        return out

    return _run


@cached_jit
def _cycle_jit(
    do_sort,
    use_morton,
    block_size,
    l2_block_size,
    l1_block_size,
    do_global,
    op_rank_fn,
    servo_enabled_value,
    servo_update_fn,
    op_morton_fn,
    op_sort_and_swizzle_with_perm_fn,
    op_sort_and_swizzle_morton_with_perm_fn,
    op_sort_and_swizzle_blocked_with_perm_fn,
    op_sort_and_swizzle_hierarchical_with_perm_fn,
    op_sort_and_swizzle_servo_with_perm_fn,
    safe_gather_fn,
    guard_cfg,
    op_interact_fn,
):
    cycle_core_fn = bind_optional_kwargs(_cycle_core, guard_cfg=guard_cfg)

    def _impl(arena, root_ptr):
        return cycle_core_fn(
            arena,
            root_ptr,
            do_sort=do_sort,
            use_morton=use_morton,
            block_size=block_size,
            morton=None,
            l2_block_size=l2_block_size,
            l1_block_size=l1_block_size,
            do_global=do_global,
            op_rank_fn=op_rank_fn,
            servo_enabled_fn=lambda: servo_enabled_value,
            servo_update_fn=servo_update_fn,
            op_morton_fn=op_morton_fn,
            op_sort_and_swizzle_with_perm_fn=op_sort_and_swizzle_with_perm_fn,
            op_sort_and_swizzle_morton_with_perm_fn=op_sort_and_swizzle_morton_with_perm_fn,
            op_sort_and_swizzle_blocked_with_perm_fn=op_sort_and_swizzle_blocked_with_perm_fn,
            op_sort_and_swizzle_hierarchical_with_perm_fn=op_sort_and_swizzle_hierarchical_with_perm_fn,
            op_sort_and_swizzle_servo_with_perm_fn=op_sort_and_swizzle_servo_with_perm_fn,
            safe_gather_fn=safe_gather_fn,
            arena_root_hash_fn=_noop_root_hash,
            damage_tile_size_fn=_noop_tile_size,
            damage_metrics_update_fn=_noop_metrics,
            op_interact_fn=op_interact_fn,
        )

    return _impl


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
    test_guards: bool = False,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a jitted cycle entrypoint for fixed DI."""
    if test_guards:
        raise RuntimeError("cycle_jit is disabled under TEST_GUARDS")
    if guard_cfg is not None:
        safe_gather_fn = make_safe_gather_fn(
            cfg=guard_cfg,
            policy=safe_gather_policy,
            safe_gather_fn=safe_gather_fn,
        )
    else:
        safe_gather_fn = wrap_policy(safe_gather_fn, safe_gather_policy)
    servo_enabled_value = bool(servo_enabled_fn())
    return _cycle_jit(
        do_sort,
        use_morton,
        block_size,
        l2_block_size,
        l1_block_size,
        do_global,
        op_rank_fn,
        servo_enabled_value,
        servo_update_fn,
        op_morton_fn,
        op_sort_and_swizzle_with_perm_fn,
        op_sort_and_swizzle_morton_with_perm_fn,
        op_sort_and_swizzle_blocked_with_perm_fn,
        op_sort_and_swizzle_hierarchical_with_perm_fn,
        op_sort_and_swizzle_servo_with_perm_fn,
        safe_gather_fn,
        guard_cfg,
        op_interact_fn,
    )


def cycle_jit_cfg(
    *,
    do_sort=True,
    use_morton=False,
    block_size=None,
    l2_block_size=None,
    l1_block_size=None,
    do_global=False,
    cfg: ArenaCycleConfig | None = None,
    test_guards: bool = False,
):
    """Return a jitted cycle entrypoint for a fixed config."""
    if cfg is None:
        cfg = DEFAULT_ARENA_CYCLE_CONFIG
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
    safe_gather_fn = cfg.safe_gather_fn or _jax_safe.safe_gather_1d
    op_interact_fn = cfg.op_interact_fn
    if op_interact_fn is None and cfg.interact_cfg is not None:
        op_interact_fn = op_interact_jit_cfg(cfg.interact_cfg)
    if op_interact_fn is None:
        if cfg.safe_gather_policy is not None:
            op_interact_fn = op_interact_jit(
                safe_gather_fn=safe_gather_fn,
                safe_gather_policy=cfg.safe_gather_policy,
                guard_cfg=cfg.guard_cfg,
            )
        else:
            op_interact_fn = _op_interact
    return cycle_jit(
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
        test_guards=test_guards,
        safe_gather_policy=cfg.safe_gather_policy,
        guard_cfg=cfg.guard_cfg,
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
    "emit_candidates_jit",
    "emit_candidates_jit_cfg",
    "compact_candidates_jit",
    "compact_candidates_jit_cfg",
    "compact_candidates_with_index_jit",
    "compact_candidates_with_index_jit_cfg",
    "intern_candidates_jit",
    "intern_candidates_jit_cfg",
    "cycle_candidates_jit",
    "cycle_jit",
    "cycle_jit_cfg",
    "cycle_intrinsic_jit",
    "cycle_intrinsic_jit_cfg",
    "coord_norm_batch_jit",
]
