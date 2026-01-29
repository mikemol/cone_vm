from __future__ import annotations

from functools import partial

from prism_core.di import cached_jit
from ic_core.config import ICEngineConfig
from ic_core.engine import (
    DEFAULT_ENGINE_CONFIG,
    ic_apply_active_pairs,
    ic_reduce,
)


@cached_jit
def _apply_active_pairs_jit(cfg: ICEngineConfig):
    def _impl(state):
        return ic_apply_active_pairs(
            state,
            compact_pairs_fn=cfg.compact_pairs_fn,
            decode_port_fn=cfg.decode_port_fn,
            alloc_plan_fn=cfg.alloc_plan_fn,
            apply_template_planned_fn=cfg.apply_template_planned_fn,
            halted_fn=cfg.halted_fn,
            scan_corrupt_fn=cfg.scan_corrupt_fn,
        )

    return _impl


def apply_active_pairs_jit(cfg: ICEngineConfig | None = None):
    """Return a jitted apply_active_pairs entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_ENGINE_CONFIG
    return _apply_active_pairs_jit(cfg)


@cached_jit
def _reduce_jit(cfg: ICEngineConfig):
    apply_fn = partial(
        ic_apply_active_pairs,
        compact_pairs_fn=cfg.compact_pairs_fn,
        decode_port_fn=cfg.decode_port_fn,
        alloc_plan_fn=cfg.alloc_plan_fn,
        apply_template_planned_fn=cfg.apply_template_planned_fn,
        halted_fn=cfg.halted_fn,
        scan_corrupt_fn=cfg.scan_corrupt_fn,
    )

    def _impl(state, max_steps):
        return ic_reduce(
            state,
            max_steps,
            apply_active_pairs_fn=apply_fn,
            scan_corrupt_fn=cfg.scan_corrupt_fn,
        )

    return _impl


def reduce_jit(cfg: ICEngineConfig | None = None):
    """Return a jitted reduce entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_ENGINE_CONFIG
    return _reduce_jit(cfg)


__all__ = [
    "apply_active_pairs_jit",
    "reduce_jit",
]
