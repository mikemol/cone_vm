from __future__ import annotations

from prism_core.di import cached_jit
from ic_core.config import (
    ICGraphConfig,
    ICGraphResolved,
    ICEngineConfig,
    ICEngineResolved,
    ICExecutionResolved,
    ICRuntimeResolved,
    DEFAULT_GRAPH_CONFIG,
    DEFAULT_GRAPH_RESOLVED,
    resolve_engine_config,
    resolve_graph_config,
)
from ic_core.engine import (
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_ENGINE_RESOLVED,
)
from ic_core.engine import ic_apply_active_pairs, ic_reduce
from ic_core.bundles import WireEndpoints, WirePtrPair, WireStarEndpoints
from ic_core.graph import ICState
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


@cached_jit
def _apply_active_pairs_jit(cfg: ICEngineConfig):
    resolved = resolve_engine_config(cfg)

    def _impl(state: ICState):
        return ic_apply_active_pairs(state, cfg=resolved)

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
def _apply_active_pairs_resolved_jit(cfg: ICEngineResolved):
    def _impl(state: ICState):
        return ic_apply_active_pairs(state, cfg=cfg)

    return _impl


def apply_active_pairs_jit_resolved(
    cfg: ICEngineResolved = DEFAULT_ENGINE_RESOLVED,
):
    """Return a jitted apply_active_pairs entrypoint for resolved DI."""
    return _apply_active_pairs_resolved_jit(cfg)


@cached_jit
def _apply_active_pairs_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState):
        return ic_apply_active_pairs(state, cfg=cfg.engine)

    return _impl


def apply_active_pairs_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted apply_active_pairs entrypoint for execution bundle."""
    return _apply_active_pairs_exec_jit(cfg)


@cached_jit
def _apply_active_pairs_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState):
        return ic_apply_active_pairs(state, cfg=cfg.engine)

    return _impl


def apply_active_pairs_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted apply_active_pairs entrypoint for runtime bundle."""
    return _apply_active_pairs_runtime_jit(cfg)


@cached_jit
def _reduce_jit(cfg: ICEngineConfig):
    resolved = resolve_engine_config(cfg)

    def _impl(state: ICState, max_steps):
        return ic_reduce(state, max_steps, cfg=resolved)

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
def _reduce_resolved_jit(cfg: ICEngineResolved):
    def _impl(state: ICState, max_steps):
        return ic_reduce(state, max_steps, cfg=cfg)

    return _impl


def reduce_jit_resolved(cfg: ICEngineResolved = DEFAULT_ENGINE_RESOLVED):
    """Return a jitted reduce entrypoint for resolved DI."""
    return _reduce_resolved_jit(cfg)


@cached_jit
def _reduce_jit_exec(cfg: ICExecutionResolved):
    def _impl(state: ICState, max_steps):
        return ic_reduce(state, max_steps, cfg=cfg.engine)

    return _impl


def reduce_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted reduce entrypoint for execution bundle."""
    return _reduce_jit_exec(cfg)


@cached_jit
def _reduce_jit_runtime(cfg: ICRuntimeResolved):
    def _impl(state: ICState, max_steps):
        return ic_reduce(state, max_steps, cfg=cfg.engine)

    return _impl


def reduce_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted reduce entrypoint for runtime bundle."""
    return _reduce_jit_runtime(cfg)


@cached_jit
def _find_active_pairs_jit(cfg: ICGraphConfig):
    scan_cfg = resolve_graph_config(cfg).scan

    def _impl(state: ICState):
        return ic_find_active_pairs(state, cfg=scan_cfg)

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
def _find_active_pairs_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState):
        return ic_find_active_pairs(state, cfg=cfg.scan)

    return _impl


def find_active_pairs_jit_resolved(
    cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED,
):
    """Return a jitted find_active_pairs entrypoint for resolved DI."""
    return _find_active_pairs_resolved_jit(cfg)


@cached_jit
def _find_active_pairs_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState):
        return ic_find_active_pairs(state, cfg=cfg.graph.scan)

    return _impl


def find_active_pairs_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted find_active_pairs entrypoint for execution bundle."""
    return _find_active_pairs_exec_jit(cfg)


@cached_jit
def _find_active_pairs_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState):
        return ic_find_active_pairs(state, cfg=cfg.graph.scan)

    return _impl


def find_active_pairs_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted find_active_pairs entrypoint for runtime bundle."""
    return _find_active_pairs_runtime_jit(cfg)


@cached_jit
def _compact_active_pairs_jit(cfg: ICGraphConfig):
    scan_cfg = resolve_graph_config(cfg).scan

    def _impl(state: ICState):
        return ic_compact_active_pairs(state, cfg=scan_cfg)

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
def _compact_active_pairs_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState):
        return ic_compact_active_pairs(state, cfg=cfg.scan)

    return _impl


def compact_active_pairs_jit_resolved(
    cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED,
):
    """Return a jitted compact_active_pairs entrypoint for resolved DI."""
    return _compact_active_pairs_resolved_jit(cfg)


@cached_jit
def _compact_active_pairs_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState):
        return ic_compact_active_pairs(state, cfg=cfg.graph.scan)

    return _impl


def compact_active_pairs_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted compact_active_pairs entrypoint for execution bundle."""
    return _compact_active_pairs_exec_jit(cfg)


@cached_jit
def _compact_active_pairs_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState):
        return ic_compact_active_pairs(state, cfg=cfg.graph.scan)

    return _impl


def compact_active_pairs_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted compact_active_pairs entrypoint for runtime bundle."""
    return _compact_active_pairs_runtime_jit(cfg)


@cached_jit
def _compact_active_pairs_result_jit(cfg: ICGraphConfig):
    scan_cfg = resolve_graph_config(cfg).scan

    def _impl(state: ICState):
        return ic_compact_active_pairs_result(state, cfg=scan_cfg)

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
def _compact_active_pairs_result_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState):
        return ic_compact_active_pairs_result(state, cfg=cfg.scan)

    return _impl


def compact_active_pairs_result_jit_resolved(
    cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED,
):
    """Return a jitted compact_active_pairs_result entrypoint for resolved DI."""
    return _compact_active_pairs_result_resolved_jit(cfg)


@cached_jit
def _compact_active_pairs_result_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState):
        return ic_compact_active_pairs_result(state, cfg=cfg.graph.scan)

    return _impl


def compact_active_pairs_result_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted compact_active_pairs_result entrypoint for execution bundle."""
    return _compact_active_pairs_result_exec_jit(cfg)


@cached_jit
def _compact_active_pairs_result_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState):
        return ic_compact_active_pairs_result(state, cfg=cfg.graph.scan)

    return _impl


def compact_active_pairs_result_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted compact_active_pairs_result entrypoint for runtime bundle."""
    return _compact_active_pairs_result_runtime_jit(cfg)


@cached_jit
def _wire_jax_jit(cfg: ICGraphConfig):
    wire_cfg = resolve_graph_config(cfg).wire

    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax(
            state,
            endpoints,
            cfg=wire_cfg,
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
def _wire_jax_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax(state, endpoints, cfg=cfg.wire)

    return _impl


def wire_jax_jit_resolved(cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED):
    """Return a jitted ic_wire_jax entrypoint for resolved DI."""
    return _wire_jax_resolved_jit(cfg)


@cached_jit
def _wire_jax_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_jax_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted ic_wire_jax entrypoint for execution bundle."""
    return _wire_jax_exec_jit(cfg)


@cached_jit
def _wire_jax_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_jax_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted ic_wire_jax entrypoint for runtime bundle."""
    return _wire_jax_runtime_jit(cfg)


@cached_jit
def _wire_jax_safe_jit(cfg: ICGraphConfig):
    wire_cfg = resolve_graph_config(cfg).wire

    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax_safe(
            state,
            endpoints,
            cfg=wire_cfg,
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
def _wire_jax_safe_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax_safe(state, endpoints, cfg=cfg.wire)

    return _impl


def wire_jax_safe_jit_resolved(cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED):
    """Return a jitted ic_wire_jax_safe entrypoint for resolved DI."""
    return _wire_jax_safe_resolved_jit(cfg)


@cached_jit
def _wire_jax_safe_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax_safe(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_jax_safe_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted ic_wire_jax_safe entrypoint for execution bundle."""
    return _wire_jax_safe_exec_jit(cfg)


@cached_jit
def _wire_jax_safe_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_jax_safe(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_jax_safe_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted ic_wire_jax_safe entrypoint for runtime bundle."""
    return _wire_jax_safe_runtime_jit(cfg)


@cached_jit
def _wire_ptrs_jit(cfg: ICGraphConfig):
    wire_cfg = resolve_graph_config(cfg).wire

    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptrs_jax(
            state,
            ptrs,
            cfg=wire_cfg,
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
def _wire_ptrs_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptrs_jax(state, ptrs, cfg=cfg.wire)

    return _impl


def wire_ptrs_jit_resolved(cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED):
    """Return a jitted ic_wire_ptrs_jax entrypoint for resolved DI."""
    return _wire_ptrs_resolved_jit(cfg)


@cached_jit
def _wire_ptrs_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptrs_jax(state, ptrs, cfg=cfg.graph.wire)

    return _impl


def wire_ptrs_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted ic_wire_ptrs_jax entrypoint for execution bundle."""
    return _wire_ptrs_exec_jit(cfg)


@cached_jit
def _wire_ptrs_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptrs_jax(state, ptrs, cfg=cfg.graph.wire)

    return _impl


def wire_ptrs_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted ic_wire_ptrs_jax entrypoint for runtime bundle."""
    return _wire_ptrs_runtime_jit(cfg)


@cached_jit
def _wire_pairs_jit(cfg: ICGraphConfig):
    wire_cfg = resolve_graph_config(cfg).wire

    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_pairs_jax(
            state,
            endpoints,
            cfg=wire_cfg,
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
def _wire_pairs_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_pairs_jax(state, endpoints, cfg=cfg.wire)

    return _impl


def wire_pairs_jit_resolved(cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED):
    """Return a jitted ic_wire_pairs_jax entrypoint for resolved DI."""
    return _wire_pairs_resolved_jit(cfg)


@cached_jit
def _wire_pairs_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_pairs_jax(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_pairs_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted ic_wire_pairs_jax entrypoint for execution bundle."""
    return _wire_pairs_exec_jit(cfg)


@cached_jit
def _wire_pairs_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState, endpoints: WireEndpoints):
        return ic_wire_pairs_jax(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_pairs_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted ic_wire_pairs_jax entrypoint for runtime bundle."""
    return _wire_pairs_runtime_jit(cfg)


@cached_jit
def _wire_ptr_pairs_jit(cfg: ICGraphConfig):
    wire_cfg = resolve_graph_config(cfg).wire

    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptr_pairs_jax(
            state,
            ptrs,
            cfg=wire_cfg,
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
def _wire_ptr_pairs_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptr_pairs_jax(state, ptrs, cfg=cfg.wire)

    return _impl


def wire_ptr_pairs_jit_resolved(cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED):
    """Return a jitted ic_wire_ptr_pairs_jax entrypoint for resolved DI."""
    return _wire_ptr_pairs_resolved_jit(cfg)


@cached_jit
def _wire_ptr_pairs_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptr_pairs_jax(state, ptrs, cfg=cfg.graph.wire)

    return _impl


def wire_ptr_pairs_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted ic_wire_ptr_pairs_jax entrypoint for execution bundle."""
    return _wire_ptr_pairs_exec_jit(cfg)


@cached_jit
def _wire_ptr_pairs_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState, ptrs: WirePtrPair):
        return ic_wire_ptr_pairs_jax(state, ptrs, cfg=cfg.graph.wire)

    return _impl


def wire_ptr_pairs_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted ic_wire_ptr_pairs_jax entrypoint for runtime bundle."""
    return _wire_ptr_pairs_runtime_jit(cfg)


@cached_jit
def _wire_star_jit(cfg: ICGraphConfig):
    wire_cfg = resolve_graph_config(cfg).wire

    def _impl(state: ICState, endpoints: WireStarEndpoints):
        return ic_wire_star_jax(
            state,
            endpoints,
            cfg=wire_cfg,
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


@cached_jit
def _wire_star_resolved_jit(cfg: ICGraphResolved):
    def _impl(state: ICState, endpoints: WireStarEndpoints):
        return ic_wire_star_jax(state, endpoints, cfg=cfg.wire)

    return _impl


def wire_star_jit_resolved(cfg: ICGraphResolved = DEFAULT_GRAPH_RESOLVED):
    """Return a jitted ic_wire_star_jax entrypoint for resolved DI."""
    return _wire_star_resolved_jit(cfg)


@cached_jit
def _wire_star_exec_jit(cfg: ICExecutionResolved):
    def _impl(state: ICState, endpoints: WireStarEndpoints):
        return ic_wire_star_jax(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_star_jit_exec(cfg: ICExecutionResolved):
    """Return a jitted ic_wire_star_jax entrypoint for execution bundle."""
    return _wire_star_exec_jit(cfg)


@cached_jit
def _wire_star_runtime_jit(cfg: ICRuntimeResolved):
    def _impl(state: ICState, endpoints: WireStarEndpoints):
        return ic_wire_star_jax(state, endpoints, cfg=cfg.graph.wire)

    return _impl


def wire_star_jit_runtime(cfg: ICRuntimeResolved):
    """Return a jitted ic_wire_star_jax entrypoint for runtime bundle."""
    return _wire_star_runtime_jit(cfg)


__all__ = [
    "apply_active_pairs_jit",
    "apply_active_pairs_jit_cfg",
    "apply_active_pairs_jit_resolved",
    "apply_active_pairs_jit_exec",
    "apply_active_pairs_jit_runtime",
    "reduce_jit",
    "reduce_jit_cfg",
    "reduce_jit_resolved",
    "reduce_jit_exec",
    "reduce_jit_runtime",
    "find_active_pairs_jit",
    "find_active_pairs_jit_cfg",
    "find_active_pairs_jit_resolved",
    "find_active_pairs_jit_exec",
    "find_active_pairs_jit_runtime",
    "compact_active_pairs_jit",
    "compact_active_pairs_jit_cfg",
    "compact_active_pairs_jit_resolved",
    "compact_active_pairs_jit_exec",
    "compact_active_pairs_jit_runtime",
    "compact_active_pairs_result_jit",
    "compact_active_pairs_result_jit_cfg",
    "compact_active_pairs_result_jit_resolved",
    "compact_active_pairs_result_jit_exec",
    "compact_active_pairs_result_jit_runtime",
    "wire_jax_jit",
    "wire_jax_jit_cfg",
    "wire_jax_jit_resolved",
    "wire_jax_jit_exec",
    "wire_jax_jit_runtime",
    "wire_jax_safe_jit",
    "wire_jax_safe_jit_cfg",
    "wire_jax_safe_jit_resolved",
    "wire_jax_safe_jit_exec",
    "wire_jax_safe_jit_runtime",
    "wire_ptrs_jit",
    "wire_ptrs_jit_cfg",
    "wire_ptrs_jit_resolved",
    "wire_ptrs_jit_exec",
    "wire_ptrs_jit_runtime",
    "wire_pairs_jit",
    "wire_pairs_jit_cfg",
    "wire_pairs_jit_resolved",
    "wire_pairs_jit_exec",
    "wire_pairs_jit_runtime",
    "wire_ptr_pairs_jit",
    "wire_ptr_pairs_jit_cfg",
    "wire_ptr_pairs_jit_resolved",
    "wire_ptr_pairs_jit_exec",
    "wire_ptr_pairs_jit_runtime",
    "wire_star_jit",
    "wire_star_jit_cfg",
    "wire_star_jit_resolved",
    "wire_star_jit_exec",
    "wire_star_jit_runtime",
]
