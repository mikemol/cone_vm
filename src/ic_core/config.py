from __future__ import annotations

from dataclasses import dataclass

from prism_core.safety import SafetyPolicy

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
    SafeIndexFn,
)


@dataclass(frozen=True, slots=True)
class ICGraphConfig:
    """Graph-level DI bundle for IC wiring safety."""

    safety_policy: SafetyPolicy | None = None
    safe_index_fn: SafeIndexFn | None = None


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


__all__ = ["ICRuleConfig", "ICEngineConfig", "ICGraphConfig"]
