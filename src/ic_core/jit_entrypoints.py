from __future__ import annotations

from functools import partial

from prism_core.di import cached_jit
from prism_core.guards import resolve_safe_index_fn
from ic_core.config import ICGraphConfig, ICEngineConfig, DEFAULT_GRAPH_CONFIG
from ic_core.engine import (
    DEFAULT_ENGINE_CONFIG,
    ic_apply_active_pairs,
    ic_reduce,
)
from ic_core.graph import (
    ic_compact_active_pairs,
    ic_compact_active_pairs_result,
    ic_find_active_pairs,
    ic_wire_jax,
    ic_wire_jax_safe,
    ic_wire_pairs_jax,
    ic_wire_ptr_pairs_jax,
    ic_wire_ptrs_jax,
    ic_wire_star_jax,
)

def _resolve_safe_index_fn(cfg: ICGraphConfig):
    return resolve_safe_index_fn(
        safe_index_fn=cfg.safe_index_fn,
        policy=cfg.safety_policy,
        guard_cfg=cfg.guard_cfg,
    )


@cached_jit
def _apply_active_pairs_jit(cfg: ICEngineConfig):
    def _impl(state):
        return ic_apply_active_pairs(
            state,
            compact_pairs_fn=cfg.compact_pairs_fn,
            compact_pairs_result_fn=cfg.compact_pairs_result_fn,
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
        compact_pairs_result_fn=cfg.compact_pairs_result_fn,
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
    safe_index_fn = _resolve_safe_index_fn(cfg)

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
    safe_index_fn = _resolve_safe_index_fn(cfg)

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


@cached_jit
def _compact_active_pairs_result_jit(cfg: ICGraphConfig):
    safe_index_fn = _resolve_safe_index_fn(cfg)

    def _impl(state):
        return ic_compact_active_pairs_result(
            state,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
            compact_cfg=cfg.compact_cfg,
        )

    return _impl


def compact_active_pairs_result_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted compact_active_pairs_result entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _compact_active_pairs_result_jit(cfg)


def compact_active_pairs_result_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for compact_active_pairs_result_jit (config-first naming)."""
    return compact_active_pairs_result_jit(cfg)


@cached_jit
def _wire_jax_jit(cfg: ICGraphConfig):
    safe_index_fn = _resolve_safe_index_fn(cfg)

    def _impl(state, node_a, port_a, node_b, port_b):
        return ic_wire_jax(
            state,
            node_a,
            port_a,
            node_b,
            port_b,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
        )

    return _impl


def wire_jax_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted ic_wire_jax entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _wire_jax_jit(cfg)


def wire_jax_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for wire_jax_jit (config-first naming)."""
    return wire_jax_jit(cfg)


@cached_jit
def _wire_jax_safe_jit(cfg: ICGraphConfig):
    safe_index_fn = _resolve_safe_index_fn(cfg)

    def _impl(state, node_a, port_a, node_b, port_b):
        return ic_wire_jax_safe(
            state,
            node_a,
            port_a,
            node_b,
            port_b,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
        )

    return _impl


def wire_jax_safe_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted ic_wire_jax_safe entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _wire_jax_safe_jit(cfg)


def wire_jax_safe_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for wire_jax_safe_jit (config-first naming)."""
    return wire_jax_safe_jit(cfg)


@cached_jit
def _wire_ptrs_jit(cfg: ICGraphConfig):
    safe_index_fn = _resolve_safe_index_fn(cfg)

    def _impl(state, ptr_a, ptr_b):
        return ic_wire_ptrs_jax(
            state,
            ptr_a,
            ptr_b,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
        )

    return _impl


def wire_ptrs_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted ic_wire_ptrs_jax entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _wire_ptrs_jit(cfg)


def wire_ptrs_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for wire_ptrs_jit (config-first naming)."""
    return wire_ptrs_jit(cfg)


@cached_jit
def _wire_pairs_jit(cfg: ICGraphConfig):
    safe_index_fn = _resolve_safe_index_fn(cfg)

    def _impl(state, node_a, port_a, node_b, port_b):
        return ic_wire_pairs_jax(
            state,
            node_a,
            port_a,
            node_b,
            port_b,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
        )

    return _impl


def wire_pairs_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted ic_wire_pairs_jax entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _wire_pairs_jit(cfg)


def wire_pairs_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for wire_pairs_jit (config-first naming)."""
    return wire_pairs_jit(cfg)


@cached_jit
def _wire_ptr_pairs_jit(cfg: ICGraphConfig):
    safe_index_fn = _resolve_safe_index_fn(cfg)

    def _impl(state, ptr_a, ptr_b):
        return ic_wire_ptr_pairs_jax(
            state,
            ptr_a,
            ptr_b,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
        )

    return _impl


def wire_ptr_pairs_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted ic_wire_ptr_pairs_jax entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _wire_ptr_pairs_jit(cfg)


def wire_ptr_pairs_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for wire_ptr_pairs_jit (config-first naming)."""
    return wire_ptr_pairs_jit(cfg)


@cached_jit
def _wire_star_jit(cfg: ICGraphConfig):
    safe_index_fn = _resolve_safe_index_fn(cfg)

    def _impl(state, center_node, center_port, leaf_nodes, leaf_ports):
        return ic_wire_star_jax(
            state,
            center_node,
            center_port,
            leaf_nodes,
            leaf_ports,
            safety_policy=cfg.safety_policy,
            safe_index_fn=safe_index_fn,
        )

    return _impl


def wire_star_jit(cfg: ICGraphConfig | None = None):
    """Return a jitted ic_wire_star_jax entrypoint for fixed DI."""
    if cfg is None:
        cfg = DEFAULT_GRAPH_CONFIG
    return _wire_star_jit(cfg)


def wire_star_jit_cfg(cfg: ICGraphConfig | None = None):
    """Alias for wire_star_jit (config-first naming)."""
    return wire_star_jit(cfg)


__all__ = [
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
]
