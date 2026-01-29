"""Facade wrappers with explicit DI and glossary contracts for IC.

Axis: Interface/Control (host-visible). Wrappers should commute with q and be
erased by q; device kernels remain in ic_core.engine / ic_core.rules.
"""

from __future__ import annotations

from ic_core.config import ICEngineConfig, ICRuleConfig
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
    ic_compact_active_pairs,
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
from ic_core.jit_entrypoints import apply_active_pairs_jit, reduce_jit
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


def engine_config_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_fn=ic_compact_active_pairs,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
) -> ICEngineConfig:
    """Build an engine config from a rule config (Interface/Control)."""
    return ICEngineConfig(
        compact_pairs_fn=compact_pairs_fn,
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
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
):
    """Return jitted apply_active_pairs using a rule config."""
    cfg = engine_config_from_rules(
        rule_cfg,
        compact_pairs_fn=compact_pairs_fn,
        decode_port_fn=decode_port_fn,
        halted_fn=halted_fn,
        scan_corrupt_fn=scan_corrupt_fn,
    )
    return apply_active_pairs_jit(cfg)


def reduce_jit_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_fn=ic_compact_active_pairs,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
):
    """Return jitted reduce using a rule config."""
    cfg = engine_config_from_rules(
        rule_cfg,
        compact_pairs_fn=compact_pairs_fn,
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
    "_host_int",
    "_host_bool",
    "_host_int_value",
    "_host_bool_value",
    "ic_alloc_jax",
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
    "ic_find_active_pairs",
    "ic_compact_active_pairs",
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
    "reduce_jit",
    "ic_alloc",
]
