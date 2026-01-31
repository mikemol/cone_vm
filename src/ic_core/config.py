from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from prism_core.errors import PrismPolicyBindingError
from prism_core.safety import (
    DEFAULT_SAFETY_POLICY,
    PolicyBinding,
    PolicyMode,
    SafetyPolicy,
    require_static_policy,
)
from prism_core.compact import CompactConfig

from ic_core.guards import ICGuardConfig
from ic_core.guards import resolve_safe_index_fn
from ic_core.protocols import (
    AllocPlanFn,
    ApplyAnnFn,
    ApplyCommuteFn,
    ApplyEraseFn,
    ApplyTemplateFn,
    ApplyTemplatePlannedFn,
    CompactPairsResultFn,
    DecodePortFn,
    HaltedFn,
    RuleForTypesFn,
    ScanCorruptFn,
    SafeIndexFn,
)


@dataclass(frozen=True, slots=True)
class ICGraphConfig:
    """Graph-level DI bundle for IC wiring safety."""

    safety_policy: SafetyPolicy | None = None
    policy_binding: PolicyBinding | None = None
    safe_index_fn: SafeIndexFn | None = None
    guard_cfg: ICGuardConfig | None = None
    compact_cfg: CompactConfig | None = None


DEFAULT_GRAPH_CONFIG = ICGraphConfig()
DEFAULT_IC_COMPACT_CONFIG = CompactConfig(
    index_dtype=jnp.uint32,
    count_dtype=jnp.uint32,
)


@dataclass(frozen=True, slots=True)
class ICWireConfig:
    """Resolved wiring config (policy + safe_index_fn)."""

    safety_policy: SafetyPolicy
    safe_index_fn: SafeIndexFn


def resolve_wire_config(cfg: ICGraphConfig) -> ICWireConfig:
    """Resolve an ICWireConfig from ICGraphConfig.

    This enforces a single policy binding path; if safe_index_fn is already
    policy-bound, callers must pass an unbound function and policy separately.
    """
    safety_policy = cfg.safety_policy
    if cfg.policy_binding is not None:
        if safety_policy is not None:
            raise PrismPolicyBindingError(
                "graph config received both policy_binding and safety_policy",
                context="ic_graph_config",
                policy_mode="ambiguous",
            )
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            raise PrismPolicyBindingError(
                "ic graph config does not support value-mode policy_binding",
                context="ic_graph_config",
                policy_mode=PolicyMode.VALUE,
            )
        safety_policy = require_static_policy(
            cfg.policy_binding, context="ic_graph_config"
        )
    if safety_policy is None:
        safety_policy = DEFAULT_SAFETY_POLICY
    safe_index_fn = resolve_safe_index_fn(
        safe_index_fn=cfg.safe_index_fn,
        policy=safety_policy,
        guard_cfg=cfg.guard_cfg,
    )
    return ICWireConfig(safety_policy=safety_policy, safe_index_fn=safe_index_fn)


DEFAULT_WIRE_CONFIG = resolve_wire_config(DEFAULT_GRAPH_CONFIG)


@dataclass(frozen=True, slots=True)
class ICScanConfig:
    """Resolved scan config (policy + safe_index_fn + compact_cfg)."""

    safety_policy: SafetyPolicy
    safe_index_fn: SafeIndexFn
    compact_cfg: CompactConfig


def resolve_scan_config(cfg: ICGraphConfig) -> ICScanConfig:
    """Resolve an ICScanConfig from ICGraphConfig."""
    safety_policy = cfg.safety_policy
    if cfg.policy_binding is not None:
        if safety_policy is not None:
            raise PrismPolicyBindingError(
                "graph config received both policy_binding and safety_policy",
                context="ic_graph_config",
                policy_mode="ambiguous",
            )
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            raise PrismPolicyBindingError(
                "ic graph config does not support value-mode policy_binding",
                context="ic_graph_config",
                policy_mode=PolicyMode.VALUE,
            )
        safety_policy = require_static_policy(
            cfg.policy_binding, context="ic_graph_config"
        )
    if safety_policy is None:
        safety_policy = DEFAULT_SAFETY_POLICY
    safe_index_fn = resolve_safe_index_fn(
        safe_index_fn=cfg.safe_index_fn,
        policy=safety_policy,
        guard_cfg=cfg.guard_cfg,
    )
    compact_cfg = cfg.compact_cfg or DEFAULT_IC_COMPACT_CONFIG
    return ICScanConfig(
        safety_policy=safety_policy,
        safe_index_fn=safe_index_fn,
        compact_cfg=compact_cfg,
    )


DEFAULT_SCAN_CONFIG = resolve_scan_config(DEFAULT_GRAPH_CONFIG)


@dataclass(frozen=True, slots=True)
class ICGraphResolved:
    """Resolved graph bundle (wire + scan configs)."""

    wire: ICWireConfig
    scan: ICScanConfig


def resolve_graph_config(cfg: ICGraphConfig) -> ICGraphResolved:
    """Resolve ICGraphConfig into fully bound wire/scan configs."""
    return ICGraphResolved(
        wire=resolve_wire_config(cfg),
        scan=resolve_scan_config(cfg),
    )


DEFAULT_GRAPH_RESOLVED = resolve_graph_config(DEFAULT_GRAPH_CONFIG)


@dataclass(frozen=True, slots=True)
class ICRuleConfig:
    """Rule-level DI bundle for IC rewrite templates."""

    rule_for_types_fn: RuleForTypesFn
    apply_annihilate_fn: ApplyAnnFn
    apply_erase_fn: ApplyEraseFn
    apply_commute_fn: ApplyCommuteFn
    apply_template_fn: ApplyTemplateFn
    alloc_plan_fn: AllocPlanFn
    apply_template_planned_fn: ApplyTemplatePlannedFn


@dataclass(frozen=True, slots=True)
class ICEngineConfig:
    """Engine-level DI bundle for IC reduction."""

    decode_port_fn: DecodePortFn
    alloc_plan_fn: AllocPlanFn
    apply_template_planned_fn: ApplyTemplatePlannedFn
    halted_fn: HaltedFn
    scan_corrupt_fn: ScanCorruptFn
    compact_pairs_result_fn: CompactPairsResultFn


@dataclass(frozen=True, slots=True)
class ICEngineResolved:
    """Resolved engine bundle (no optional branches)."""

    decode_port_fn: DecodePortFn
    alloc_plan_fn: AllocPlanFn
    apply_template_planned_fn: ApplyTemplatePlannedFn
    halted_fn: HaltedFn
    scan_corrupt_fn: ScanCorruptFn
    compact_pairs_result_fn: CompactPairsResultFn


def resolve_engine_config(cfg: ICEngineConfig) -> ICEngineResolved:
    """Resolve ICEngineConfig into an ICEngineResolved bundle."""
    return ICEngineResolved(
        decode_port_fn=cfg.decode_port_fn,
        alloc_plan_fn=cfg.alloc_plan_fn,
        apply_template_planned_fn=cfg.apply_template_planned_fn,
        halted_fn=cfg.halted_fn,
        scan_corrupt_fn=cfg.scan_corrupt_fn,
        compact_pairs_result_fn=cfg.compact_pairs_result_fn,
    )


@dataclass(frozen=True, slots=True)
class ICExecutionConfig:
    """Execution-level bundle: graph + engine config."""

    graph_cfg: ICGraphConfig
    engine_cfg: ICEngineConfig


@dataclass(frozen=True, slots=True)
class ICExecutionResolved:
    """Resolved execution bundle (graph + engine)."""

    graph: ICGraphResolved
    engine: ICEngineResolved


def resolve_execution_config(cfg: ICExecutionConfig) -> ICExecutionResolved:
    """Resolve ICExecutionConfig into fully bound graph + engine bundles."""
    return ICExecutionResolved(
        graph=resolve_graph_config(cfg.graph_cfg),
        engine=resolve_engine_config(cfg.engine_cfg),
    )


@dataclass(frozen=True, slots=True)
class ICRuntimeConfig:
    """Runtime bundle: graph + rule + engine configs."""

    graph_cfg: ICGraphConfig
    rule_cfg: ICRuleConfig
    engine_cfg: ICEngineConfig


@dataclass(frozen=True, slots=True)
class ICRuntimeResolved:
    """Resolved runtime bundle (graph + rule + engine)."""

    graph: ICGraphResolved
    rule: ICRuleConfig
    engine: ICEngineResolved


def resolve_runtime_config(cfg: ICRuntimeConfig) -> ICRuntimeResolved:
    """Resolve ICRuntimeConfig into bound graph + engine bundles."""
    return ICRuntimeResolved(
        graph=resolve_graph_config(cfg.graph_cfg),
        rule=cfg.rule_cfg,
        engine=resolve_engine_config(cfg.engine_cfg),
    )


__all__ = [
    "ICRuleConfig",
    "ICEngineConfig",
    "ICEngineResolved",
    "ICExecutionConfig",
    "ICExecutionResolved",
    "ICRuntimeConfig",
    "ICRuntimeResolved",
    "ICGraphConfig",
    "ICWireConfig",
    "ICScanConfig",
    "ICGraphResolved",
    "DEFAULT_GRAPH_CONFIG",
    "DEFAULT_GRAPH_RESOLVED",
    "DEFAULT_IC_COMPACT_CONFIG",
    "DEFAULT_WIRE_CONFIG",
    "DEFAULT_SCAN_CONFIG",
    "resolve_wire_config",
    "resolve_scan_config",
    "resolve_graph_config",
    "resolve_execution_config",
    "resolve_engine_config",
    "resolve_runtime_config",
]
