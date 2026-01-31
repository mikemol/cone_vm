from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import jit, lax

from prism_core import jax_safe as _jax_safe
from prism_core.errors import PrismPolicyBindingError
from prism_core.di import call_with_optional_kwargs
from prism_core.guards import resolve_safe_gather_fn, resolve_safe_gather_value_fn
from prism_core.safety import (
    PolicyBinding,
    PolicyMode,
    require_static_policy,
    require_value_policy,
)
from prism_metrics.metrics import _damage_metrics_update, _damage_tile_size
from prism_vm_core.domains import _host_int_value
from prism_vm_core.gating import _servo_enabled
from prism_vm_core.guards import _guard_max
from prism_vm_core.hashes import _arena_root_hash_host
from prism_vm_core.ontology import OP_ADD, OP_SUC, OP_ZERO

from prism_bsp.space import (
    RANK_HOT,
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
from prism_bsp.config import (
    ArenaInteractConfig,
    ArenaCycleConfig,
    ArenaSortConfig,
    DEFAULT_ARENA_SORT_CONFIG,
    DEFAULT_ARENA_INTERACT_CONFIG,
    DEFAULT_ARENA_CYCLE_CONFIG,
    SwizzleWithPermFns,
    SwizzleWithPermFnsBound,
)


_TEST_GUARDS = _jax_safe.TEST_GUARDS

_DEFAULT_SWIZZLE_WITH_PERM_FNS = SwizzleWithPermFnsBound(
    with_perm=op_sort_and_swizzle_with_perm,
    morton_with_perm=op_sort_and_swizzle_morton_with_perm,
    blocked_with_perm=op_sort_and_swizzle_blocked_with_perm,
    hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm,
    servo_with_perm=op_sort_and_swizzle_servo_with_perm,
)

_DEFAULT_SWIZZLE_WITH_PERM_VALUE_FNS = SwizzleWithPermFnsBound(
    with_perm=op_sort_and_swizzle_with_perm_value,
    morton_with_perm=op_sort_and_swizzle_morton_with_perm_value,
    blocked_with_perm=op_sort_and_swizzle_blocked_with_perm_value,
    hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm_value,
    servo_with_perm=op_sort_and_swizzle_servo_with_perm_value,
)


def _op_interact_core(
    arena,
    safe_gather_fn,
    scatter_drop_fn,
    guard_max_fn,
):
    ops = arena.opcode
    a1 = arena.arg1
    a2 = arena.arg2
    cap = jnp.int32(ops.shape[0])
    is_hot = arena.rank == RANK_HOT
    is_add = ops == OP_ADD
    # Guard pointer gathers in test mode; avoid touching inactive garbage rows.
    hot_add = is_hot & is_add
    a1_for_op = jnp.where(hot_add, a1, jnp.int32(0))
    a2_for_op = jnp.where(hot_add, a2, jnp.int32(0))
    op_a1 = safe_gather_fn(ops, a1_for_op, "op_interact.op_a1")
    op_a2 = safe_gather_fn(ops, a2_for_op, "op_interact.op_a2")
    is_zero_a1 = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    is_suc_a1 = op_a1 == OP_SUC
    is_suc_a2 = op_a2 == OP_SUC
    mask_zero = hot_add & (is_zero_a1 | is_zero_a2)
    mask_suc = hot_add & (is_suc_a1 | is_suc_a2) & (~mask_zero) & (~arena.oom)

    # First: local rewrites that don't allocate.
    zero_other = jnp.where(is_zero_a1, a2, a1)
    zero_other = jnp.where(mask_zero, zero_other, jnp.int32(0))
    y_op = safe_gather_fn(ops, zero_other, "op_interact.zero_op")
    y_a1 = safe_gather_fn(a1, zero_other, "op_interact.zero_a1")
    y_a2 = safe_gather_fn(a2, zero_other, "op_interact.zero_a2")
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
    choose_a1 = is_suc_a1 & (~is_suc_a2 | (a1 <= a2))
    suc_node = jnp.where(choose_a1, a1, a2)
    other_node = jnp.where(choose_a1, a2, a1)
    suc_for_spawn = jnp.where(spawn_mask, suc_node, jnp.int32(0))
    other_for_spawn = jnp.where(spawn_mask, other_node, jnp.int32(0))
    grandchild_x = safe_gather_fn(a1, suc_for_spawn, "op_interact.grandchild_x")
    payload_op = jnp.full_like(new_add_idx, OP_ADD)
    payload_a1_raw = jnp.where(spawn_mask, grandchild_x, jnp.int32(0))
    payload_a2_raw = jnp.where(spawn_mask, other_for_spawn, jnp.int32(0))
    payload_swap = payload_a2_raw < payload_a1_raw
    payload_a1 = jnp.where(payload_swap, payload_a2_raw, payload_a1_raw)
    payload_a2 = jnp.where(payload_swap, payload_a1_raw, payload_a2_raw)

    valid = spawn_mask
    idxs2 = jnp.where(valid, new_add_idx, cap)
    # idxs2 uses cap as a drop sentinel for _scatter_drop (see helper note).

    final_ops = scatter_drop_fn(
        new_ops,
        idxs2,
        jnp.where(valid, payload_op, new_ops[0]),
        "op_interact.final_ops",
    )
    final_a1 = scatter_drop_fn(
        new_a1,
        idxs2,
        jnp.where(valid, payload_a1, new_a1[0]),
        "op_interact.final_a1",
    )
    final_a2 = scatter_drop_fn(
        new_a2,
        idxs2,
        jnp.where(valid, payload_a2, new_a2[0]),
        "op_interact.final_a2",
    )

    overflow = jnp.sum(mask_suc.astype(jnp.int32)) > available
    new_oom = arena.oom | overflow
    new_count = arena.count + total_spawn
    guard_max_fn(new_count, cap, "arena.count")
    return arena._replace(
        opcode=final_ops,
        arg1=final_a1,
        arg2=final_a2,
        count=new_count,
        oom=new_oom,
    )


def _op_interact_core_value(
    arena,
    policy_value,
    safe_gather_value_fn,
    scatter_drop_fn,
    guard_max_fn,
):
    ops = arena.opcode
    a1 = arena.arg1
    a2 = arena.arg2
    cap = jnp.int32(ops.shape[0])
    is_hot = arena.rank == RANK_HOT
    is_add = ops == OP_ADD
    hot_add = is_hot & is_add
    a1_for_op = jnp.where(hot_add, a1, jnp.int32(0))
    a2_for_op = jnp.where(hot_add, a2, jnp.int32(0))
    op_a1 = safe_gather_value_fn(
        ops, a1_for_op, "op_interact.op_a1", policy_value=policy_value
    )
    op_a2 = safe_gather_value_fn(
        ops, a2_for_op, "op_interact.op_a2", policy_value=policy_value
    )
    is_zero_a1 = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    is_suc_a1 = op_a1 == OP_SUC
    is_suc_a2 = op_a2 == OP_SUC
    mask_zero = hot_add & (is_zero_a1 | is_zero_a2)
    mask_suc = hot_add & (is_suc_a1 | is_suc_a2) & (~mask_zero) & (~arena.oom)

    zero_other = jnp.where(is_zero_a1, a2, a1)
    zero_other = jnp.where(mask_zero, zero_other, jnp.int32(0))
    y_op = safe_gather_value_fn(
        ops, zero_other, "op_interact.zero_op", policy_value=policy_value
    )
    y_a1 = safe_gather_value_fn(
        a1, zero_other, "op_interact.zero_a1", policy_value=policy_value
    )
    y_a2 = safe_gather_value_fn(
        a2, zero_other, "op_interact.zero_a2", policy_value=policy_value
    )
    new_ops = jnp.where(mask_zero, y_op, ops)
    new_a1 = jnp.where(mask_zero, y_a1, a1)
    new_a2 = jnp.where(mask_zero, y_a2, a2)

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

    choose_a1 = is_suc_a1 & (~is_suc_a2 | (a1 <= a2))
    suc_node = jnp.where(choose_a1, a1, a2)
    other_node = jnp.where(choose_a1, a2, a1)
    suc_for_spawn = jnp.where(spawn_mask, suc_node, jnp.int32(0))
    other_for_spawn = jnp.where(spawn_mask, other_node, jnp.int32(0))
    grandchild_x = safe_gather_value_fn(
        a1, suc_for_spawn, "op_interact.grandchild_x", policy_value=policy_value
    )
    payload_op = jnp.full_like(new_add_idx, OP_ADD)
    payload_a1_raw = jnp.where(spawn_mask, grandchild_x, jnp.int32(0))
    payload_a2_raw = jnp.where(spawn_mask, other_for_spawn, jnp.int32(0))
    payload_swap = payload_a2_raw < payload_a1_raw
    payload_a1 = jnp.where(payload_swap, payload_a2_raw, payload_a1_raw)
    payload_a2 = jnp.where(payload_swap, payload_a1_raw, payload_a2_raw)

    valid = spawn_mask
    idxs2 = jnp.where(valid, new_add_idx, cap)

    final_ops = scatter_drop_fn(
        new_ops,
        idxs2,
        jnp.where(valid, payload_op, new_ops[0]),
        "op_interact.final_ops",
    )
    final_a1 = scatter_drop_fn(
        new_a1,
        idxs2,
        jnp.where(valid, payload_a1, new_a1[0]),
        "op_interact.final_a1",
    )
    final_a2 = scatter_drop_fn(
        new_a2,
        idxs2,
        jnp.where(valid, payload_a2, new_a2[0]),
        "op_interact.final_a2",
    )

    overflow = jnp.sum(mask_suc.astype(jnp.int32)) > available
    new_oom = arena.oom | overflow
    new_count = arena.count + total_spawn
    guard_max_fn(new_count, cap, "arena.count")
    return arena._replace(
        opcode=final_ops,
        arg1=final_a1,
        arg2=final_a2,
        count=new_count,
        oom=new_oom,
    )


@jax.jit(
    static_argnames=(
        "safe_gather_fn",
        "scatter_drop_fn",
        "guard_max_fn",
    )
)
def op_interact(
    arena,
    *,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    scatter_drop_fn=_jax_safe.scatter_drop,
    guard_max_fn=_guard_max,
):
    return _op_interact_core(arena, safe_gather_fn, scatter_drop_fn, guard_max_fn)


@jax.jit(
    static_argnames=(
        "safe_gather_value_fn",
        "scatter_drop_fn",
        "guard_max_fn",
    )
)
def op_interact_value(
    arena,
    policy_value,
    *,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    scatter_drop_fn=_jax_safe.scatter_drop,
    guard_max_fn=_guard_max,
):
    return _op_interact_core_value(
        arena, policy_value, safe_gather_value_fn, scatter_drop_fn, guard_max_fn
    )


def op_interact_cfg(
    arena, *, cfg: ArenaInteractConfig = DEFAULT_ARENA_INTERACT_CONFIG
):
    """Interface/Control wrapper for op_interact with DI bundle."""
    scatter_drop_fn = cfg.scatter_drop_fn or _jax_safe.scatter_drop
    guard_max_fn = cfg.guard_max_fn or _guard_max
    safe_gather_policy = cfg.safe_gather_policy
    safe_gather_policy_value = cfg.safe_gather_policy_value
    if cfg.policy_binding is not None:
        if safe_gather_policy is not None or safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "op_interact_cfg received both policy_binding and "
                "safe_gather_policy/safe_gather_policy_value",
                context="op_interact_cfg",
                policy_mode="ambiguous",
            )
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            safe_gather_policy_value = require_value_policy(
                cfg.policy_binding, context="op_interact_cfg"
            )
        else:
            safe_gather_policy = require_static_policy(
                cfg.policy_binding, context="op_interact_cfg"
            )
    if (
        safe_gather_policy is not None
        and safe_gather_policy_value is not None
    ):
        raise PrismPolicyBindingError(
            "op_interact_cfg received both safe_gather_policy and "
            "safe_gather_policy_value",
            context="op_interact_cfg",
            policy_mode="ambiguous",
        )
    if safe_gather_policy_value is not None:
        safe_gather_value_fn = resolve_safe_gather_value_fn(
            safe_gather_value_fn=cfg.safe_gather_value_fn,
            guard_cfg=cfg.guard_cfg,
        )
        return op_interact_value(
            arena,
            safe_gather_policy_value,
            safe_gather_value_fn=safe_gather_value_fn,
            scatter_drop_fn=scatter_drop_fn,
            guard_max_fn=guard_max_fn,
        )
    safe_gather_fn = resolve_safe_gather_fn(
        safe_gather_fn=cfg.safe_gather_fn,
        policy=safe_gather_policy,
        guard_cfg=cfg.guard_cfg,
    )
    return op_interact(
        arena,
        safe_gather_fn=safe_gather_fn,
        scatter_drop_fn=scatter_drop_fn,
        guard_max_fn=guard_max_fn,
    )


def op_interact_bound_cfg(
    arena,
    policy_binding: PolicyBinding,
    *,
    cfg: ArenaInteractConfig | None = None,
):
    """PolicyBinding-required wrapper for op_interact_cfg."""
    if cfg is None:
        cfg = ArenaInteractConfig(policy_binding=policy_binding)
    else:
        cfg = replace(
            cfg,
            policy_binding=policy_binding,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )
    return op_interact_cfg(arena, cfg=cfg)


def cycle_core(
    arena,
    root_ptr,
    sort_cfg: ArenaSortConfig,
    *,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound = _DEFAULT_SWIZZLE_WITH_PERM_FNS,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    guard_cfg=None,
    arena_root_hash_fn=_arena_root_hash_host,
    damage_tile_size_fn=_damage_tile_size,
    damage_metrics_update_fn=_damage_metrics_update,
    op_interact_fn=op_interact,
):
    do_sort = sort_cfg.do_sort
    use_morton = sort_cfg.use_morton
    block_size = sort_cfg.block_size
    morton = sort_cfg.morton
    l2_block_size = sort_cfg.l2_block_size
    l1_block_size = sort_cfg.l1_block_size
    do_global = sort_cfg.do_global
    # BSPˢ is renormalization only: must preserve denotation after q (m3).
    # BSPᵗ controls when identity is created via commit_stratum barriers.
    # COMMUTES: BSPᵗ ⟂ BSPˢ  [test: tests/test_arena_denotation_invariance.py::test_arena_denotation_invariance_random_suite]
    """Run one BSP cycle; sorting/scheduling is renormalization only."""
    # See IMPLEMENTATION_PLAN.md (m3 denotation invariance).
    arena = op_rank_fn(arena)
    root_arr = jnp.asarray(root_ptr, dtype=jnp.int32)
    if do_sort:
        swizzle = swizzle_with_perm_fns
        pre_hash = arena_root_hash_fn(arena, root_arr)
        morton_arr = None
        if servo_enabled_fn():
            arena = servo_update_fn(arena)
            morton_arr = morton if morton is not None else op_morton_fn(arena)
            servo_mask = arena.servo[0]
            arena, inv_perm = call_with_optional_kwargs(
                swizzle.servo_with_perm,
                {"guard_cfg": guard_cfg, "safe_gather_fn": safe_gather_fn},
                arena,
                morton_arr,
                servo_mask,
            )
        else:
            if use_morton or morton is not None:
                morton_arr = morton if morton is not None else op_morton_fn(arena)
            if l2_block_size is not None or l1_block_size is not None:
                if l2_block_size is None:
                    l2_block_size = l1_block_size
                if l1_block_size is None:
                    l1_block_size = l2_block_size
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.hierarchical_with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_fn": safe_gather_fn},
                    arena,
                    l2_block_size,
                    l1_block_size,
                    morton=morton_arr,
                    do_global=do_global,
                )
            elif block_size is not None:
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.blocked_with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_fn": safe_gather_fn},
                    arena,
                    block_size,
                    morton=morton_arr,
                )
            elif morton_arr is not None:
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.morton_with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_fn": safe_gather_fn},
                    arena,
                    morton_arr,
                )
            else:
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_fn": safe_gather_fn},
                    arena,
                )
        # Root remap is a pointer gather; guard when guard_cfg is supplied.
        safe_gather_root = resolve_safe_gather_fn(
            safe_gather_fn=safe_gather_fn,
            guard_cfg=guard_cfg,
        )
        root_idx = jnp.where(root_arr != 0, root_arr, jnp.int32(0))
        root_g = safe_gather_root(inv_perm, root_idx, "cycle.root_remap")
        root_arr = jnp.where(root_arr != 0, root_g, 0)
        if _TEST_GUARDS and pre_hash != arena_root_hash_fn(arena, root_arr):
            raise RuntimeError("BSPˢ renormalization changed root structure")
    tile_size = damage_tile_size_fn(block_size, l2_block_size, l1_block_size)
    damage_metrics_update_fn(arena, tile_size)
    arena = op_interact_fn(arena)
    return arena, root_arr


def cycle_core_value(
    arena,
    root_ptr,
    policy_value,
    sort_cfg: ArenaSortConfig,
    *,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound = _DEFAULT_SWIZZLE_WITH_PERM_VALUE_FNS,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    guard_cfg=None,
    arena_root_hash_fn=_arena_root_hash_host,
    damage_tile_size_fn=_damage_tile_size,
    damage_metrics_update_fn=_damage_metrics_update,
    op_interact_value_fn=op_interact_value,
):
    """Run one BSP cycle with policy_value as data (JAX value)."""
    do_sort = sort_cfg.do_sort
    use_morton = sort_cfg.use_morton
    block_size = sort_cfg.block_size
    morton = sort_cfg.morton
    l2_block_size = sort_cfg.l2_block_size
    l1_block_size = sort_cfg.l1_block_size
    do_global = sort_cfg.do_global
    safe_gather_value_fn_guarded = resolve_safe_gather_value_fn(
        safe_gather_value_fn=safe_gather_value_fn,
        guard_cfg=guard_cfg,
    )
    arena = op_rank_fn(arena)
    root_arr = jnp.asarray(root_ptr, dtype=jnp.int32)
    if do_sort:
        swizzle = swizzle_with_perm_fns
        pre_hash = arena_root_hash_fn(arena, root_arr)
        morton_arr = None
        if servo_enabled_fn():
            arena = servo_update_fn(arena)
            morton_arr = morton if morton is not None else op_morton_fn(arena)
            servo_mask = arena.servo[0]
            arena, inv_perm = call_with_optional_kwargs(
                swizzle.servo_with_perm,
                {"guard_cfg": guard_cfg, "safe_gather_value_fn": safe_gather_value_fn},
                arena,
                morton_arr,
                servo_mask,
                policy_value,
            )
        else:
            if use_morton or morton is not None:
                morton_arr = morton if morton is not None else op_morton_fn(arena)
            if l2_block_size is not None or l1_block_size is not None:
                if l2_block_size is None:
                    l2_block_size = l1_block_size
                if l1_block_size is None:
                    l1_block_size = l2_block_size
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.hierarchical_with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_value_fn": safe_gather_value_fn},
                    arena,
                    l2_block_size,
                    l1_block_size,
                    policy_value,
                    morton=morton_arr,
                    do_global=do_global,
                )
            elif block_size is not None:
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.blocked_with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_value_fn": safe_gather_value_fn},
                    arena,
                    block_size,
                    policy_value,
                    morton=morton_arr,
                )
            elif morton_arr is not None:
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.morton_with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_value_fn": safe_gather_value_fn},
                    arena,
                    morton_arr,
                    policy_value,
                )
            else:
                arena, inv_perm = call_with_optional_kwargs(
                    swizzle.with_perm,
                    {"guard_cfg": guard_cfg, "safe_gather_value_fn": safe_gather_value_fn},
                    arena,
                    policy_value,
                )
        root_idx = jnp.where(root_arr != 0, root_arr, jnp.int32(0))
        root_g = safe_gather_value_fn_guarded(
            inv_perm, root_idx, "cycle.root_remap", policy_value=policy_value
        )
        root_arr = jnp.where(root_arr != 0, root_g, 0)
        if _TEST_GUARDS and pre_hash != arena_root_hash_fn(arena, root_arr):
            raise RuntimeError("BSPˢ renormalization changed root structure")
    tile_size = damage_tile_size_fn(block_size, l2_block_size, l1_block_size)
    damage_metrics_update_fn(arena, tile_size)
    arena = call_with_optional_kwargs(
        op_interact_value_fn,
        {"safe_gather_value_fn": safe_gather_value_fn_guarded},
        arena,
        policy_value,
    )
    return arena, root_arr


def cycle(
    arena,
    root_ptr,
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound = _DEFAULT_SWIZZLE_WITH_PERM_FNS,
    safe_gather_fn=_jax_safe.safe_gather_1d,
    guard_cfg=None,
    arena_root_hash_fn=_arena_root_hash_host,
    damage_tile_size_fn=_damage_tile_size,
    damage_metrics_update_fn=_damage_metrics_update,
    op_interact_fn=op_interact,
):
    return cycle_core(
        arena,
        root_ptr,
        sort_cfg=sort_cfg,
        op_rank_fn=op_rank_fn,
        servo_enabled_fn=servo_enabled_fn,
        servo_update_fn=servo_update_fn,
        op_morton_fn=op_morton_fn,
        swizzle_with_perm_fns=swizzle_with_perm_fns,
        safe_gather_fn=safe_gather_fn,
        guard_cfg=guard_cfg,
        arena_root_hash_fn=arena_root_hash_fn,
        damage_tile_size_fn=damage_tile_size_fn,
        damage_metrics_update_fn=damage_metrics_update_fn,
        op_interact_fn=op_interact_fn,
    )


def cycle_value(
    arena,
    root_ptr,
    policy_value,
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    op_rank_fn=op_rank,
    servo_enabled_fn=_servo_enabled,
    servo_update_fn=_servo_update,
    op_morton_fn=op_morton,
    swizzle_with_perm_fns: SwizzleWithPermFnsBound = _DEFAULT_SWIZZLE_WITH_PERM_VALUE_FNS,
    safe_gather_value_fn=_jax_safe.safe_gather_1d_value,
    guard_cfg=None,
    arena_root_hash_fn=_arena_root_hash_host,
    damage_tile_size_fn=_damage_tile_size,
    damage_metrics_update_fn=_damage_metrics_update,
    op_interact_value_fn=op_interact_value,
):
    return cycle_core_value(
        arena,
        root_ptr,
        policy_value,
        sort_cfg=sort_cfg,
        op_rank_fn=op_rank_fn,
        servo_enabled_fn=servo_enabled_fn,
        servo_update_fn=servo_update_fn,
        op_morton_fn=op_morton_fn,
        swizzle_with_perm_fns=swizzle_with_perm_fns,
        safe_gather_value_fn=safe_gather_value_fn,
        guard_cfg=guard_cfg,
        arena_root_hash_fn=arena_root_hash_fn,
        damage_tile_size_fn=damage_tile_size_fn,
        damage_metrics_update_fn=damage_metrics_update_fn,
        op_interact_value_fn=op_interact_value_fn,
    )

def cycle_cfg(
    arena,
    root_ptr,
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    cfg: ArenaCycleConfig = DEFAULT_ARENA_CYCLE_CONFIG,
):
    """Interface/Control wrapper for cycle_core with DI bundle."""
    op_rank_fn = cfg.op_rank_fn or op_rank
    servo_enabled_fn = cfg.servo_enabled_fn or _servo_enabled
    servo_update_fn = cfg.servo_update_fn or _servo_update
    op_morton_fn = cfg.op_morton_fn or op_morton
    op_sort_and_swizzle_with_perm_fn = (
        cfg.op_sort_and_swizzle_with_perm_fn
        or op_sort_and_swizzle_with_perm
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
    if cfg.swizzle_with_perm_fns is not None:
        swizzle_bundle = cfg.swizzle_with_perm_fns
        if swizzle_bundle.with_perm is not None:
            op_sort_and_swizzle_with_perm_fn = swizzle_bundle.with_perm
        if swizzle_bundle.morton_with_perm is not None:
            op_sort_and_swizzle_morton_with_perm_fn = swizzle_bundle.morton_with_perm
        if swizzle_bundle.blocked_with_perm is not None:
            op_sort_and_swizzle_blocked_with_perm_fn = swizzle_bundle.blocked_with_perm
        if swizzle_bundle.hierarchical_with_perm is not None:
            op_sort_and_swizzle_hierarchical_with_perm_fn = (
                swizzle_bundle.hierarchical_with_perm
            )
        if swizzle_bundle.servo_with_perm is not None:
            op_sort_and_swizzle_servo_with_perm_fn = swizzle_bundle.servo_with_perm
    safe_gather_policy = cfg.safe_gather_policy
    safe_gather_policy_value = cfg.safe_gather_policy_value
    if cfg.policy_binding is not None:
        if safe_gather_policy is not None or safe_gather_policy_value is not None:
            raise PrismPolicyBindingError(
                "cycle_cfg received both policy_binding and "
                "safe_gather_policy/safe_gather_policy_value",
                context="cycle_cfg",
                policy_mode="ambiguous",
            )
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            safe_gather_policy_value = require_value_policy(
                cfg.policy_binding, context="cycle_cfg"
            )
        else:
            safe_gather_policy = require_static_policy(
                cfg.policy_binding, context="cycle_cfg"
            )
    if (
        safe_gather_policy is not None
        and safe_gather_policy_value is not None
    ):
        raise PrismPolicyBindingError(
            "cycle_cfg received both safe_gather_policy and safe_gather_policy_value",
            context="cycle_cfg",
            policy_mode="ambiguous",
        )
    safe_gather_fn = resolve_safe_gather_fn(
        safe_gather_fn=cfg.safe_gather_fn,
        policy=safe_gather_policy,
        guard_cfg=cfg.guard_cfg,
    )
    safe_gather_value_fn = cfg.safe_gather_value_fn
    if safe_gather_policy_value is not None:
        if safe_gather_value_fn is None:
            safe_gather_value_fn = _jax_safe.safe_gather_1d_value
        if cfg.swizzle_with_perm_value_fns is not None:
            swizzle_value_bundle = cfg.swizzle_with_perm_value_fns
            if swizzle_value_bundle.with_perm is not None:
                op_sort_and_swizzle_with_perm_fn = swizzle_value_bundle.with_perm
            if swizzle_value_bundle.morton_with_perm is not None:
                op_sort_and_swizzle_morton_with_perm_fn = swizzle_value_bundle.morton_with_perm
            if swizzle_value_bundle.blocked_with_perm is not None:
                op_sort_and_swizzle_blocked_with_perm_fn = swizzle_value_bundle.blocked_with_perm
            if swizzle_value_bundle.hierarchical_with_perm is not None:
                op_sort_and_swizzle_hierarchical_with_perm_fn = (
                    swizzle_value_bundle.hierarchical_with_perm
                )
            if swizzle_value_bundle.servo_with_perm is not None:
                op_sort_and_swizzle_servo_with_perm_fn = swizzle_value_bundle.servo_with_perm
        if cfg.op_sort_and_swizzle_with_perm_fn is None and cfg.swizzle_with_perm_fns is None:
            op_sort_and_swizzle_with_perm_fn = op_sort_and_swizzle_with_perm_value
        if cfg.op_sort_and_swizzle_morton_with_perm_fn is None and cfg.swizzle_with_perm_fns is None:
            op_sort_and_swizzle_morton_with_perm_fn = (
                op_sort_and_swizzle_morton_with_perm_value
            )
        if cfg.op_sort_and_swizzle_blocked_with_perm_fn is None and cfg.swizzle_with_perm_fns is None:
            op_sort_and_swizzle_blocked_with_perm_fn = (
                op_sort_and_swizzle_blocked_with_perm_value
            )
        if cfg.op_sort_and_swizzle_hierarchical_with_perm_fn is None and cfg.swizzle_with_perm_fns is None:
            op_sort_and_swizzle_hierarchical_with_perm_fn = (
                op_sort_and_swizzle_hierarchical_with_perm_value
            )
        if cfg.op_sort_and_swizzle_servo_with_perm_fn is None and cfg.swizzle_with_perm_fns is None:
            op_sort_and_swizzle_servo_with_perm_fn = (
                op_sort_and_swizzle_servo_with_perm_value
            )
    arena_root_hash_fn = cfg.arena_root_hash_fn or _arena_root_hash_host
    damage_tile_size_fn = cfg.damage_tile_size_fn or _damage_tile_size
    damage_metrics_update_fn = cfg.damage_metrics_update_fn or _damage_metrics_update
    op_interact_fn = cfg.op_interact_fn
    if op_interact_fn is None and cfg.interact_cfg is not None:
        op_interact_fn = lambda a: op_interact_cfg(a, cfg=cfg.interact_cfg)
    if op_interact_fn is None:
        if safe_gather_policy_value is not None:
            op_interact_fn = op_interact_value
        else:
            op_interact_fn = lambda a: op_interact(a, safe_gather_fn=safe_gather_fn)
    if safe_gather_policy_value is not None and safe_gather_value_fn is not None:
        swizzle_with_perm_fns = SwizzleWithPermFnsBound(
            with_perm=op_sort_and_swizzle_with_perm_fn,
            morton_with_perm=op_sort_and_swizzle_morton_with_perm_fn,
            blocked_with_perm=op_sort_and_swizzle_blocked_with_perm_fn,
            hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm_fn,
            servo_with_perm=op_sort_and_swizzle_servo_with_perm_fn,
        )
        return cycle_value(
            arena,
            root_ptr,
            safe_gather_policy_value,
            sort_cfg=sort_cfg,
            op_rank_fn=op_rank_fn,
            servo_enabled_fn=servo_enabled_fn,
            servo_update_fn=servo_update_fn,
            op_morton_fn=op_morton_fn,
            swizzle_with_perm_fns=swizzle_with_perm_fns,
            safe_gather_value_fn=safe_gather_value_fn,
            guard_cfg=cfg.guard_cfg,
            arena_root_hash_fn=arena_root_hash_fn,
            damage_tile_size_fn=damage_tile_size_fn,
            damage_metrics_update_fn=damage_metrics_update_fn,
            op_interact_value_fn=op_interact_fn,
        )
    swizzle_with_perm_fns = SwizzleWithPermFnsBound(
        with_perm=op_sort_and_swizzle_with_perm_fn,
        morton_with_perm=op_sort_and_swizzle_morton_with_perm_fn,
        blocked_with_perm=op_sort_and_swizzle_blocked_with_perm_fn,
        hierarchical_with_perm=op_sort_and_swizzle_hierarchical_with_perm_fn,
        servo_with_perm=op_sort_and_swizzle_servo_with_perm_fn,
    )
    return cycle_core(
        arena,
        root_ptr,
        sort_cfg=sort_cfg,
        op_rank_fn=op_rank_fn,
        servo_enabled_fn=servo_enabled_fn,
        servo_update_fn=servo_update_fn,
        op_morton_fn=op_morton_fn,
        swizzle_with_perm_fns=swizzle_with_perm_fns,
        safe_gather_fn=safe_gather_fn,
        arena_root_hash_fn=arena_root_hash_fn,
        damage_tile_size_fn=damage_tile_size_fn,
        damage_metrics_update_fn=damage_metrics_update_fn,
        op_interact_fn=op_interact_fn,
    )


def cycle_bound_cfg(
    arena,
    root_ptr,
    policy_binding: PolicyBinding,
    *,
    sort_cfg: ArenaSortConfig = DEFAULT_ARENA_SORT_CONFIG,
    cfg: ArenaCycleConfig | None = None,
):
    """PolicyBinding-required wrapper for cycle_cfg."""
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
    return cycle_cfg(
        arena,
        root_ptr,
        sort_cfg=sort_cfg,
        cfg=cfg,
    )


__all__ = [
    "op_interact",
    "op_interact_value",
    "op_interact_cfg",
    "op_interact_bound_cfg",
    "cycle_core",
    "cycle_core_value",
    "cycle",
    "cycle_value",
    "cycle_cfg",
    "cycle_bound_cfg",
]
