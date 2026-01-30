from __future__ import annotations

from dataclasses import dataclass

from prism_core.safety import PolicyBinding, SafetyPolicy
from prism_core.compact import CompactConfig

from ic_core.guards import ICGuardConfig
from ic_core.protocols import (
    AllocPlanFn,
    ApplyAnnFn,
    ApplyCommuteFn,
    ApplyEraseFn,
    ApplyTemplateFn,
    ApplyTemplatePlannedFn,
    CompactPairsFn,
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

    compact_pairs_fn: CompactPairsFn
    decode_port_fn: DecodePortFn
    alloc_plan_fn: AllocPlanFn
    apply_template_planned_fn: ApplyTemplatePlannedFn
    halted_fn: HaltedFn
    scan_corrupt_fn: ScanCorruptFn
    compact_pairs_result_fn: CompactPairsResultFn | None = None


__all__ = [
    "ICRuleConfig",
    "ICEngineConfig",
    "ICGraphConfig",
    "DEFAULT_GRAPH_CONFIG",
]
