from __future__ import annotations

from functools import partial

from prism_core.di import cached_jit
from prism_core.jax_safe import safe_index_1d
from ic_core.config import ICGraphConfig, ICEngineConfig, DEFAULT_GRAPH_CONFIG
from ic_core.engine import (
    DEFAULT_ENGINE_CONFIG,
    ic_apply_active_pairs,
    ic_reduce,
)
from ic_core.graph import ic_find_active_pairs, ic_compact_active_pairs


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


def apply_active_pairs_jit_cfg(cfg: ICEngineConfig | None = None):
    """Alias for apply_active_pairs_jit (config-first naming)."""
    return apply_active_pairs_jit(cfg)


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


def reduce_jit_cfg(cfg: ICEngineConfig | None = None):
    """Alias for reduce_jit (config-first naming)."""
    return reduce_jit(cfg)


@cached_jit
def _find_active_pairs_jit(cfg: ICGraphConfig):
    safe_index_fn = cfg.safe_index_fn or safe_index_1d

    def _impl(state):
        return ic_find_active_pairs(
            state,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
            compact_cfg=cfg.compact_cfg,
        )

    return _impl


def find_active_pairs_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted find_active_pairs entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _find_active_pairs_jit(cfg)


def find_active_pairs_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for find_active_pairs_jit (config-first naming)."""
    return find_active_pairs_jit(cfg)


@cached_jit
def _compact_active_pairs_jit(cfg: ICGraphConfig):
    safe_index_fn = cfg.safe_index_fn or safe_index_1d

    def _impl(state):
        return ic_compact_active_pairs(
            state,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
            compact_cfg=cfg.compact_cfg,
        )

    return _impl


def compact_active_pairs_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted compact_active_pairs entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _compact_active_pairs_jit(cfg)


def compact_active_pairs_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for compact_active_pairs_jit (config-first naming)."""
    return compact_active_pairs_jit(cfg)


__all__ = [
    "apply_active_pairs_jit",
    "apply_active_pairs_jit_cfg",
    "reduce_jit",
    "reduce_jit_cfg",
    "find_active_pairs_jit",
    "find_active_pairs_jit_cfg",
    "compact_active_pairs_jit",
    "compact_active_pairs_jit_cfg",
]
