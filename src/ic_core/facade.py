"""Facade wrappers with explicit DI and glossary contracts for IC.

Axis: Interface/Control (host-visible). Wrappers should commute with q and be
erased by q; device kernels remain in ic_core.engine / ic_core.rules.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from prism_core.safety import SafetyPolicy
from prism_core.jax_safe import safe_index_1d
from prism_core.alloc import (
    AllocConfig,
    DEFAULT_ALLOC_CONFIG,
    alloc2_cfg,
    alloc4_cfg,
    free2_cfg,
)
from ic_core.guards import (
    ICGuardConfig,
    DEFAULT_IC_GUARD_CONFIG,
    safe_index_1d_cfg,
)
from prism_core.guards import make_safe_index_fn

from ic_core.config import ICEngineConfig, ICGraphConfig, ICRuleConfig, DEFAULT_GRAPH_CONFIG
from ic_core.domains import (
    HostBool,
    HostInt,
    ICPtr,
    ICNodeId,
    ICPortId,
    _host_bool,
    _host_bool_value,
    _host_int,
    _host_int_value,
    _ic_ptr,
    _node_id,
    _port_id,
    _require_ic_ptr,
    _require_node_id,
    _require_port_id,
    _require_ptr_domain,
)
from ic_core.protocols import (
    AllocPlanFn,
    ApplyAnnFn,
    ApplyCommuteFn,
    ApplyEraseFn,
    ApplyTemplateFn,
    ApplyTemplatePlannedFn,
    CompactPairsFn,
    DecodePortFn,
    HaltedFn,
    RuleForTypesFn,
    ScanCorruptFn,
)
from ic_core.engine import (
    DEFAULT_ENGINE_CONFIG,
    ICRewriteStats,
    ic_apply_active_pairs_cfg,
    ic_reduce_cfg,
)
from ic_core.graph import (
    ICState,
    PORT_AUX_LEFT,
    PORT_AUX_RIGHT,
    PORT_PRINCIPAL,
    TYPE_CON,
    TYPE_DUP,
    TYPE_ERA,
    TYPE_FREE,
    _halted,
    _scan_corrupt_ports,
    decode_port,
    encode_port,
    ic_alloc,
    ic_alloc_jax,
    ic_alloc_jax_cfg,
    ic_compact_active_pairs,
    ic_compact_active_pairs_result,
    ic_find_active_pairs,
    ic_init,
    ic_wire,
    ic_wire_jax,
    ic_wire_jax_safe,
    ic_wire_pairs_jax,
    ic_wire_ptr_pairs_jax,
    ic_wire_ptrs_jax,
    ic_wire_star_jax,
)
from ic_core.jit_entrypoints import (
    apply_active_pairs_jit,
    apply_active_pairs_jit_cfg,
    reduce_jit,
    reduce_jit_cfg,
    find_active_pairs_jit,
    find_active_pairs_jit_cfg,
    compact_active_pairs_jit,
    compact_active_pairs_jit_cfg,
    compact_active_pairs_result_jit,
    compact_active_pairs_result_jit_cfg,
    wire_jax_jit,
    wire_jax_jit_cfg,
    wire_jax_safe_jit,
    wire_jax_safe_jit_cfg,
    wire_ptrs_jit,
    wire_ptrs_jit_cfg,
    wire_pairs_jit,
    wire_pairs_jit_cfg,
    wire_ptr_pairs_jit,
    wire_ptr_pairs_jit_cfg,
    wire_star_jit,
    wire_star_jit_cfg,
)
from ic_core.rules import (
    DEFAULT_RULE_CONFIG,
    RULE_ALLOC_ANNIHILATE,
    RULE_ALLOC_COMMUTE,
    RULE_ALLOC_ERASE,
    RULE_TABLE,
    TEMPLATE_ANNIHILATE,
    TEMPLATE_COMMUTE,
    TEMPLATE_ERASE,
    TEMPLATE_NONE,
    ic_apply_annihilate,
    ic_apply_annihilate_cfg,
    ic_apply_commute,
    ic_apply_commute_cfg,
    ic_apply_erase,
    ic_apply_erase_cfg,
    ic_apply_template,
    ic_apply_template_cfg,
    ic_apply_template_planned_cfg,
    ic_alloc_plan_cfg,
    ic_rule_for_types,
    ic_rule_for_types_cfg,
    ic_select_template,
    ic_select_template_cfg,
)



def ic_apply_active_pairs(
    state: ICState, *, cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG
):
    """Interface/Control wrapper for IC apply_active_pairs.

    Axis: Interface/Control. Commutes with q. Erased by q.
    """
    return ic_apply_active_pairs_cfg(state, cfg=cfg)


def ic_reduce(
    state: ICState, max_steps: int, *, cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG
):
    """Interface/Control wrapper for IC reduce.

    Axis: Interface/Control. Commutes with q. Erased by q.
    """
    return ic_reduce_cfg(state, max_steps, cfg=cfg)


def graph_config_with_policy(
    safety_policy: SafetyPolicy | None,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICGraphConfig:
    """Return a graph config with safety_policy set."""
    return replace(cfg, safety_policy=safety_policy)


def graph_config_with_index_fn(
    safe_index_fn,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICGraphConfig:
    """Return a graph config with safe_index_fn set."""
    return replace(cfg, safe_index_fn=safe_index_fn)


def graph_config_with_compact_cfg(
    compact_cfg,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICGraphConfig:
    """Return a graph config with compact_cfg set."""
    return replace(cfg, compact_cfg=compact_cfg)


def graph_config_with_guard(
    *,
    safety_policy: SafetyPolicy | None = None,
    guard_cfg: ICGuardConfig = DEFAULT_IC_GUARD_CONFIG,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
    compact_cfg=None,
) -> ICGraphConfig:
    """Return a graph config using safe_index_1d_cfg with guard config."""
    cfg = replace(cfg, safe_index_fn=make_safe_index_fn(cfg=guard_cfg))
    if safety_policy is not None:
        cfg = replace(cfg, safety_policy=safety_policy)
    if compact_cfg is not None:
        cfg = replace(cfg, compact_cfg=compact_cfg)
    return cfg


def rule_config_with_alloc(
    alloc_cfg: AllocConfig | None,
    *,
    cfg: ICRuleConfig = DEFAULT_RULE_CONFIG,
) -> ICRuleConfig:
    """Return a rule config with allocator helpers wired from AllocConfig."""
    if alloc_cfg is None:
        return cfg
    alloc2_fn = partial(alloc2_cfg, cfg=alloc_cfg)
    alloc4_fn = partial(alloc4_cfg, cfg=alloc_cfg)
    free2_fn = partial(free2_cfg, cfg=alloc_cfg)
    apply_erase_fn = partial(
        ic_apply_erase,
        alloc2_fn=alloc2_fn,
        free2_fn=free2_fn,
    )
    apply_commute_fn = partial(
        ic_apply_commute,
        alloc4_fn=alloc4_fn,
        free2_fn=free2_fn,
    )
    apply_template_fn = partial(
        ic_apply_template,
        apply_annihilate_fn=cfg.apply_annihilate_fn,
        apply_erase_fn=apply_erase_fn,
        apply_commute_fn=apply_commute_fn,
    )
    return replace(
        cfg,
        apply_erase_fn=apply_erase_fn,
        apply_commute_fn=apply_commute_fn,
        apply_template_fn=apply_template_fn,
    )


def ic_wire_jax_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    port_a: jnp.ndarray,
    node_b: jnp.ndarray,
    port_b: jnp.ndarray,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax with safety policy."""
    return ic_wire_jax(
        state,
        node_a,
        port_a,
        node_b,
        port_b,
        safety_policy=cfg.safety_policy,
        safe_index_fn=cfg.safe_index_fn or safe_index_1d,
    )


def ic_wire_jax_safe_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    port_a: jnp.ndarray,
    node_b: jnp.ndarray,
    port_b: jnp.ndarray,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax_safe with safety policy."""
    return ic_wire_jax_safe(
        state,
        node_a,
        port_a,
        node_b,
        port_b,
        safety_policy=cfg.safety_policy,
        safe_index_fn=cfg.safe_index_fn or safe_index_1d,
    )


def ic_wire_ptrs_jax_cfg(
    state: ICState,
    ptr_a: jnp.ndarray,
    ptr_b: jnp.ndarray,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptrs_jax with safety policy."""
    return ic_wire_ptrs_jax(
        state,
        ptr_a,
        ptr_b,
        safety_policy=cfg.safety_policy,
        safe_index_fn=cfg.safe_index_fn or safe_index_1d,
    )


def ic_wire_pairs_jax_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    port_a: jnp.ndarray,
    node_b: jnp.ndarray,
    port_b: jnp.ndarray,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_pairs_jax with safety policy."""
    return ic_wire_pairs_jax(
        state,
        node_a,
        port_a,
        node_b,
        port_b,
        safety_policy=cfg.safety_policy,
        safe_index_fn=cfg.safe_index_fn or safe_index_1d,
    )


def ic_wire_ptr_pairs_jax_cfg(
    state: ICState,
    ptr_a: jnp.ndarray,
    ptr_b: jnp.ndarray,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptr_pairs_jax with safety policy."""
    return ic_wire_ptr_pairs_jax(
        state,
        ptr_a,
        ptr_b,
        safety_policy=cfg.safety_policy,
        safe_index_fn=cfg.safe_index_fn or safe_index_1d,
    )


def ic_wire_star_jax_cfg(
    state: ICState,
    center_node: jnp.ndarray,
    center_port: jnp.ndarray,
    leaf_nodes: jnp.ndarray,
    leaf_ports: jnp.ndarray,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_star_jax with safety policy."""
    return ic_wire_star_jax(
        state,
        center_node,
        center_port,
        leaf_nodes,
        leaf_ports,
        safety_policy=cfg.safety_policy,
        safe_index_fn=cfg.safe_index_fn or safe_index_1d,
    )


def ic_find_active_pairs_cfg(
    state: ICState, *, cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for active pair detection with DI bundle."""
    safe_index_fn = cfg.safe_index_fn or safe_index_1d
    return ic_find_active_pairs(
        state,
        safety_policy=cfg.safety_policy,
        safe_index_fn=safe_index_fn,
        compact_cfg=cfg.compact_cfg,
    )


def ic_compact_active_pairs_cfg(
    state: ICState, *, cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for compact active pairs with DI bundle."""
    safe_index_fn = cfg.safe_index_fn or safe_index_1d
    return ic_compact_active_pairs(
        state,
        safety_policy=cfg.safety_policy,
        safe_index_fn=safe_index_fn,
        compact_cfg=cfg.compact_cfg,
    )


def ic_compact_active_pairs_result_cfg(
    state: ICState, *, cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG
):
    """Interface/Control wrapper for CompactResult active pairs with DI bundle."""
    safe_index_fn = cfg.safe_index_fn or safe_index_1d
    return ic_compact_active_pairs_result(
        state,
        safety_policy=cfg.safety_policy,
        safe_index_fn=safe_index_fn,
        compact_cfg=cfg.compact_cfg,
    )


def engine_config_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_fn=ic_compact_active_pairs,
    compact_pairs_result_fn=None,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
) -> ICEngineConfig:
    """Build an engine config from a rule config (Interface/Control)."""
    return ICEngineConfig(
        compact_pairs_fn=compact_pairs_fn,
        compact_pairs_result_fn=compact_pairs_result_fn,
        decode_port_fn=decode_port_fn,
        alloc_plan_fn=rule_cfg.alloc_plan_fn,
        apply_template_planned_fn=rule_cfg.apply_template_planned_fn,
        halted_fn=halted_fn,
        scan_corrupt_fn=scan_corrupt_fn,
    )


def apply_active_pairs_jit_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_fn=ic_compact_active_pairs,
    compact_pairs_result_fn=None,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
):
    """Return jitted apply_active_pairs using a rule config."""
    cfg = engine_config_from_rules(
        rule_cfg,
        compact_pairs_fn=compact_pairs_fn,
        compact_pairs_result_fn=compact_pairs_result_fn,
        decode_port_fn=decode_port_fn,
        halted_fn=halted_fn,
        scan_corrupt_fn=scan_corrupt_fn,
    )
    return apply_active_pairs_jit(cfg)


def reduce_jit_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_fn=ic_compact_active_pairs,
    compact_pairs_result_fn=None,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
):
    """Return jitted reduce using a rule config."""
    cfg = engine_config_from_rules(
        rule_cfg,
        compact_pairs_fn=compact_pairs_fn,
        compact_pairs_result_fn=compact_pairs_result_fn,
        decode_port_fn=decode_port_fn,
        halted_fn=halted_fn,
        scan_corrupt_fn=scan_corrupt_fn,
    )
    return reduce_jit(cfg)


__all__ = [
    "TYPE_FREE",
    "TYPE_ERA",
    "TYPE_CON",
    "TYPE_DUP",
    "PORT_PRINCIPAL",
    "PORT_AUX_LEFT",
    "PORT_AUX_RIGHT",
    "RULE_ALLOC_ANNIHILATE",
    "RULE_ALLOC_ERASE",
    "RULE_ALLOC_COMMUTE",
    "TEMPLATE_NONE",
    "TEMPLATE_ANNIHILATE",
    "TEMPLATE_ERASE",
    "TEMPLATE_COMMUTE",
    "RULE_TABLE",
    "ICState",
    "ICRewriteStats",
    "ICRuleConfig",
    "ICEngineConfig",
    "ICGraphConfig",
    "AllocConfig",
    "DEFAULT_ALLOC_CONFIG",
    "CompactPairsFn",
    "DecodePortFn",
    "AllocPlanFn",
    "ApplyTemplatePlannedFn",
    "HaltedFn",
    "ScanCorruptFn",
    "RuleForTypesFn",
    "ApplyAnnFn",
    "ApplyEraseFn",
    "ApplyCommuteFn",
    "ApplyTemplateFn",
    "DEFAULT_RULE_CONFIG",
    "DEFAULT_ENGINE_CONFIG",
    "DEFAULT_GRAPH_CONFIG",
    "graph_config_with_policy",
    "graph_config_with_index_fn",
    "graph_config_with_compact_cfg",
    "graph_config_with_guard",
    "rule_config_with_alloc",
    "ICGuardConfig",
    "DEFAULT_IC_GUARD_CONFIG",
    "safe_index_1d_cfg",
    "make_safe_index_fn",
    "graph_config_with_index_fn",
    "ICNodeId",
    "ICPortId",
    "ICPtr",
    "HostInt",
    "HostBool",
    "_node_id",
    "_port_id",
    "_ic_ptr",
    "_require_node_id",
    "_require_port_id",
    "_require_ic_ptr",
    "_require_ptr_domain",
    "_host_int",
    "_host_bool",
    "_host_int_value",
    "_host_bool_value",
    "ic_alloc_jax",
    "ic_alloc_jax_cfg",
    "encode_port",
    "decode_port",
    "ic_init",
    "ic_wire",
    "ic_wire_jax",
    "ic_wire_ptrs_jax",
    "ic_wire_jax_safe",
    "ic_wire_pairs_jax",
    "ic_wire_ptr_pairs_jax",
    "ic_wire_star_jax",
    "ic_wire_jax_cfg",
    "ic_wire_jax_safe_cfg",
    "ic_wire_ptrs_jax_cfg",
    "ic_wire_pairs_jax_cfg",
    "ic_wire_ptr_pairs_jax_cfg",
    "ic_wire_star_jax_cfg",
    "ic_find_active_pairs",
    "ic_find_active_pairs_cfg",
    "ic_compact_active_pairs",
    "ic_compact_active_pairs_cfg",
    "ic_compact_active_pairs_result",
    "ic_compact_active_pairs_result_cfg",
    "ic_rule_for_types",
    "ic_rule_for_types_cfg",
    "ic_select_template",
    "ic_select_template_cfg",
    "ic_apply_annihilate",
    "ic_apply_annihilate_cfg",
    "ic_apply_erase",
    "ic_apply_erase_cfg",
    "ic_apply_commute",
    "ic_apply_commute_cfg",
    "ic_apply_template",
    "ic_apply_template_cfg",
    "ic_alloc_plan_cfg",
    "ic_apply_template_planned_cfg",
    "ic_apply_active_pairs",
    "ic_reduce",
    "ic_apply_active_pairs_cfg",
    "ic_reduce_cfg",
    "engine_config_from_rules",
    "apply_active_pairs_jit_from_rules",
    "reduce_jit_from_rules",
    "apply_active_pairs_jit",
    "apply_active_pairs_jit_cfg",
    "reduce_jit",
    "reduce_jit_cfg",
    "find_active_pairs_jit",
    "find_active_pairs_jit_cfg",
    "compact_active_pairs_jit",
    "compact_active_pairs_jit_cfg",
    "compact_active_pairs_result_jit",
    "compact_active_pairs_result_jit_cfg",
    "wire_jax_jit",
    "wire_jax_jit_cfg",
    "wire_jax_safe_jit",
    "wire_jax_safe_jit_cfg",
    "wire_ptrs_jit",
    "wire_ptrs_jit_cfg",
    "wire_pairs_jit",
    "wire_pairs_jit_cfg",
    "wire_ptr_pairs_jit",
    "wire_ptr_pairs_jit_cfg",
    "wire_star_jit",
    "wire_star_jit_cfg",
    "ic_alloc",
]
