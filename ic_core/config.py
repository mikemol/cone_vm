from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class ICRuleConfig:
    """Rule-level DI bundle for IC rewrite templates."""

    rule_for_types_fn: Callable
    apply_annihilate_fn: Callable
    apply_erase_fn: Callable
    apply_commute_fn: Callable
    apply_template_fn: Callable
    alloc_plan_fn: Callable
    apply_template_planned_fn: Callable


@dataclass(frozen=True, slots=True)
class ICEngineConfig:
    """Engine-level DI bundle for IC reduction."""

    compact_pairs_fn: Callable
    decode_port_fn: Callable
    alloc_plan_fn: Callable
    apply_template_planned_fn: Callable
    halted_fn: Callable
    scan_corrupt_fn: Callable


__all__ = ["ICRuleConfig", "ICEngineConfig"]
