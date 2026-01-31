"""Facade wrappers with explicit DI and glossary contracts for IC.

Axis: Interface/Control (host-visible). Wrappers should commute with q and be
erased by q; device kernels remain in ic_core.engine / ic_core.rules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

from prism_core.safety import (
    PolicyBinding,
    SafetyPolicy,
)
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
    resolve_safe_index_fn,
    safe_index_1d_cfg,
)

from ic_core.config import (
    ICEngineConfig,
    ICEngineResolved,
    ICGraphConfig,
    ICGraphResolved,
    ICExecutionConfig,
    ICExecutionResolved,
    ICRuntimeConfig,
    ICRuntimeResolved,
    ICRuleConfig,
    DEFAULT_GRAPH_CONFIG,
    DEFAULT_GRAPH_RESOLVED,
    DEFAULT_WIRE_CONFIG,
    ICWireConfig,
    DEFAULT_SCAN_CONFIG,
    ICScanConfig,
    resolve_wire_config,
    resolve_scan_config,
    resolve_graph_config,
    resolve_engine_config,
    resolve_execution_config,
    resolve_runtime_config,
)
from ic_core.bundles import WireEndpoints, WirePtrPair, WireStarEndpoints
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
    DecodePortFn,
    HaltedFn,
    RuleForTypesFn,
    ScanCorruptFn,
)
from ic_core.engine import (
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_ENGINE_RESOLVED,
    ICRewriteStats,
    ic_apply_active_pairs as _ic_apply_active_pairs_core,
    ic_reduce as _ic_reduce_core,
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
    apply_active_pairs_jit_exec,
    apply_active_pairs_jit_resolved,
    apply_active_pairs_jit_runtime,
    reduce_jit,
    reduce_jit_cfg,
    reduce_jit_exec,
    reduce_jit_resolved,
    reduce_jit_runtime,
    find_active_pairs_jit,
    find_active_pairs_jit_cfg,
    find_active_pairs_jit_exec,
    find_active_pairs_jit_resolved,
    find_active_pairs_jit_runtime,
    compact_active_pairs_jit,
    compact_active_pairs_jit_cfg,
    compact_active_pairs_jit_exec,
    compact_active_pairs_jit_resolved,
    compact_active_pairs_jit_runtime,
    compact_active_pairs_result_jit,
    compact_active_pairs_result_jit_cfg,
    compact_active_pairs_result_jit_exec,
    compact_active_pairs_result_jit_resolved,
    compact_active_pairs_result_jit_runtime,
    wire_jax_jit,
    wire_jax_jit_cfg,
    wire_jax_jit_exec,
    wire_jax_jit_resolved,
    wire_jax_jit_runtime,
    wire_jax_safe_jit,
    wire_jax_safe_jit_cfg,
    wire_jax_safe_jit_exec,
    wire_jax_safe_jit_resolved,
    wire_jax_safe_jit_runtime,
    wire_ptrs_jit,
    wire_ptrs_jit_cfg,
    wire_ptrs_jit_exec,
    wire_ptrs_jit_resolved,
    wire_ptrs_jit_runtime,
    wire_pairs_jit,
    wire_pairs_jit_cfg,
    wire_pairs_jit_exec,
    wire_pairs_jit_resolved,
    wire_pairs_jit_runtime,
    wire_ptr_pairs_jit,
    wire_ptr_pairs_jit_cfg,
    wire_ptr_pairs_jit_exec,
    wire_ptr_pairs_jit_resolved,
    wire_ptr_pairs_jit_runtime,
    wire_star_jit,
    wire_star_jit_cfg,
    wire_star_jit_exec,
    wire_star_jit_resolved,
    wire_star_jit_runtime,
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
    return replace(cfg, safety_policy=safety_policy, policy_binding=None)


def graph_config_with_policy_binding(
    policy_binding: PolicyBinding | None,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICGraphConfig:
    """Return a graph config with policy_binding set (clears safety_policy)."""
    return replace(cfg, policy_binding=policy_binding, safety_policy=None)


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


def engine_config_with_compact_result_fn(
    compact_pairs_result_fn,
    *,
    cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG,
) -> ICEngineConfig:
    """Return an engine config with compact_pairs_result_fn set."""
    return replace(cfg, compact_pairs_result_fn=compact_pairs_result_fn)


def engine_config_with_scan_cfg(
    scan_cfg: ICScanConfig,
    *,
    cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG,
) -> ICEngineConfig:
    """Return an engine config with compact_pairs_result_fn bound to scan_cfg."""
    bound = partial(ic_compact_active_pairs_result, cfg=scan_cfg)
    return replace(cfg, compact_pairs_result_fn=bound)


def engine_config_with_graph_cfg(
    graph_cfg: ICGraphConfig,
    *,
    cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG,
) -> ICEngineConfig:
    """Return an engine config with compact pairs bound to graph scan cfg."""
    scan_cfg = resolve_graph_config(graph_cfg).scan
    return engine_config_with_scan_cfg(scan_cfg, cfg=cfg)


def engine_resolved_with_graph_cfg(
    graph_cfg: ICGraphConfig,
    *,
    cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG,
) -> ICEngineResolved:
    """Return a resolved engine bundle wired to the graph scan cfg."""
    return resolve_engine_config(engine_config_with_graph_cfg(graph_cfg, cfg=cfg))


def resolve_engine_cfg(
    cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG,
) -> ICEngineResolved:
    """Resolve an engine config into a fully bound bundle."""
    return resolve_engine_config(cfg)


def execution_config_from_graph_engine(
    graph_cfg: ICGraphConfig,
    *,
    engine_cfg: ICEngineConfig = DEFAULT_ENGINE_CONFIG,
) -> ICExecutionConfig:
    """Build an execution config from graph + engine configs."""
    return ICExecutionConfig(graph_cfg=graph_cfg, engine_cfg=engine_cfg)


DEFAULT_EXECUTION_CONFIG = execution_config_from_graph_engine(
    DEFAULT_GRAPH_CONFIG, engine_cfg=DEFAULT_ENGINE_CONFIG
)
DEFAULT_EXECUTION_RESOLVED = resolve_execution_config(DEFAULT_EXECUTION_CONFIG)


def resolve_execution_cfg(
    cfg: ICExecutionConfig,
) -> ICExecutionResolved:
    """Resolve an execution config into a bound graph + engine bundle."""
    return resolve_execution_config(cfg)


def resolve_execution_cfg_default() -> ICExecutionResolved:
    """Resolve the default execution config into a bound bundle."""
    return DEFAULT_EXECUTION_RESOLVED


def runtime_config_from_graph_rule(
    graph_cfg: ICGraphConfig,
    rule_cfg: ICRuleConfig,
    *,
    engine_cfg: ICEngineConfig | None = None,
) -> ICRuntimeConfig:
    """Build a runtime config from graph + rule configs."""
    if engine_cfg is None:
        engine_cfg = engine_config_from_rules(rule_cfg)
    engine_cfg = engine_config_with_graph_cfg(graph_cfg, cfg=engine_cfg)
    return ICRuntimeConfig(
        graph_cfg=graph_cfg,
        rule_cfg=rule_cfg,
        engine_cfg=engine_cfg,
    )


def resolve_runtime_cfg(cfg: ICRuntimeConfig) -> ICRuntimeResolved:
    """Resolve a runtime config into a bound bundle."""
    return resolve_runtime_config(cfg)


def resolve_runtime_cfg_default() -> ICRuntimeResolved:
    """Resolve the default runtime config into a bound bundle."""
    return DEFAULT_RUNTIME_RESOLVED


@dataclass(frozen=True, slots=True)
class ICRuntimeOps:
    """Bound runtime operations for a resolved IC runtime bundle."""

    cfg: ICRuntimeResolved
    wire_jax: Callable
    wire_jax_safe: Callable
    wire_ptrs: Callable
    wire_pairs: Callable
    wire_ptr_pairs: Callable
    wire_star: Callable
    find_active_pairs: Callable
    compact_active_pairs: Callable
    compact_active_pairs_result: Callable
    apply_active_pairs: Callable
    reduce: Callable


def make_runtime_ops(cfg: ICRuntimeResolved) -> ICRuntimeOps:
    """Bind runtime operations to a resolved bundle."""
    return ICRuntimeOps(
        cfg=cfg,
        wire_jax=partial(ic_wire_jax, cfg=cfg.graph.wire),
        wire_jax_safe=partial(ic_wire_jax_safe, cfg=cfg.graph.wire),
        wire_ptrs=partial(ic_wire_ptrs_jax, cfg=cfg.graph.wire),
        wire_pairs=partial(ic_wire_pairs_jax, cfg=cfg.graph.wire),
        wire_ptr_pairs=partial(ic_wire_ptr_pairs_jax, cfg=cfg.graph.wire),
        wire_star=partial(ic_wire_star_jax, cfg=cfg.graph.wire),
        find_active_pairs=partial(ic_find_active_pairs, cfg=cfg.graph.scan),
        compact_active_pairs=partial(ic_compact_active_pairs, cfg=cfg.graph.scan),
        compact_active_pairs_result=partial(
            ic_compact_active_pairs_result, cfg=cfg.graph.scan
        ),
        apply_active_pairs=partial(_ic_apply_active_pairs_core, cfg=cfg.engine),
        reduce=partial(_ic_reduce_core, cfg=cfg.engine),
    )


def make_runtime_ops_from_cfg(cfg: ICRuntimeConfig) -> ICRuntimeOps:
    """Resolve and bind runtime operations from a runtime config."""
    return make_runtime_ops(resolve_runtime_config(cfg))



def ic_apply_active_pairs_resolved(
    state: ICState, *, cfg: ICEngineResolved = DEFAULT_ENGINE_RESOLVED
):
    """Interface/Control wrapper for apply_active_pairs with resolved DI."""
    return _ic_apply_active_pairs_core(state, cfg=cfg)


def ic_reduce_resolved(
    state: ICState, max_steps: int, *, cfg: ICEngineResolved = DEFAULT_ENGINE_RESOLVED
):
    """Interface/Control wrapper for reduce with resolved DI."""
    return _ic_reduce_core(state, max_steps, cfg=cfg)


def ic_apply_active_pairs_exec(
    state: ICState, *, cfg: ICExecutionResolved
):
    """Interface/Control wrapper for apply_active_pairs with execution bundle."""
    return _ic_apply_active_pairs_core(state, cfg=cfg.engine)


def ic_reduce_exec(
    state: ICState, max_steps: int, *, cfg: ICExecutionResolved
):
    """Interface/Control wrapper for reduce with execution bundle."""
    return _ic_reduce_core(state, max_steps, cfg=cfg.engine)


def ic_apply_active_pairs_runtime(
    state: ICState, *, cfg: ICRuntimeResolved
):
    """Interface/Control wrapper for apply_active_pairs with runtime bundle."""
    return _ic_apply_active_pairs_core(state, cfg=cfg.engine)


def ic_reduce_runtime(
    state: ICState, max_steps: int, *, cfg: ICRuntimeResolved
):
    """Interface/Control wrapper for reduce with runtime bundle."""
    return _ic_reduce_core(state, max_steps, cfg=cfg.engine)


def graph_config_with_guard(
    *,
    safety_policy: SafetyPolicy | None = None,
    guard_cfg: ICGuardConfig = DEFAULT_IC_GUARD_CONFIG,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
    compact_cfg=None,
) -> ICGraphConfig:
    """Return a graph config using safe_index_1d_cfg with guard config."""
    cfg = replace(cfg, guard_cfg=guard_cfg)
    if safety_policy is not None:
        cfg = replace(cfg, safety_policy=safety_policy)
    if compact_cfg is not None:
        cfg = replace(cfg, compact_cfg=compact_cfg)
    return cfg


def resolve_graph_cfg(
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICGraphResolved:
    """Resolve a graph config into bound wire/scan bundles."""
    return resolve_graph_config(cfg)


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
    endpoints: WireEndpoints,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax with safety policy."""
    wire_cfg = resolve_graph_config(cfg).wire
    return ic_wire_jax(
        state,
        endpoints,
        cfg=wire_cfg,
    )


def ic_wire_jax_resolved(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICGraphResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax with resolved graph cfg."""
    return ic_wire_jax(state, endpoints, cfg=cfg.wire)


def ic_wire_jax_exec(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICExecutionResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax with execution bundle."""
    return ic_wire_jax(state, endpoints, cfg=cfg.graph.wire)


def ic_wire_jax_runtime(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICRuntimeResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax with runtime bundle."""
    return ic_wire_jax(state, endpoints, cfg=cfg.graph.wire)


def ic_wire_jax_safe_cfg(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax_safe with safety policy."""
    wire_cfg = resolve_graph_config(cfg).wire
    return ic_wire_jax_safe(
        state,
        endpoints,
        cfg=wire_cfg,
    )


def ic_wire_jax_safe_resolved(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICGraphResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax_safe with resolved graph cfg."""
    return ic_wire_jax_safe(state, endpoints, cfg=cfg.wire)


def ic_wire_jax_safe_exec(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICExecutionResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax_safe with execution bundle."""
    return ic_wire_jax_safe(state, endpoints, cfg=cfg.graph.wire)


def ic_wire_jax_safe_runtime(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICRuntimeResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_jax_safe with runtime bundle."""
    return ic_wire_jax_safe(state, endpoints, cfg=cfg.graph.wire)


def ic_wire_ptrs_jax_cfg(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptrs_jax with safety policy."""
    wire_cfg = resolve_graph_config(cfg).wire
    return ic_wire_ptrs_jax(
        state,
        ptrs,
        cfg=wire_cfg,
    )


def ic_wire_ptrs_jax_resolved(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICGraphResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptrs_jax with resolved graph cfg."""
    return ic_wire_ptrs_jax(state, ptrs, cfg=cfg.wire)


def ic_wire_ptrs_jax_exec(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICExecutionResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptrs_jax with execution bundle."""
    return ic_wire_ptrs_jax(state, ptrs, cfg=cfg.graph.wire)


def ic_wire_ptrs_jax_runtime(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICRuntimeResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptrs_jax with runtime bundle."""
    return ic_wire_ptrs_jax(state, ptrs, cfg=cfg.graph.wire)


def ic_wire_pairs_jax_cfg(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_pairs_jax with safety policy."""
    wire_cfg = resolve_graph_config(cfg).wire
    return ic_wire_pairs_jax(
        state,
        endpoints,
        cfg=wire_cfg,
    )


def ic_wire_pairs_jax_resolved(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICGraphResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_pairs_jax with resolved graph cfg."""
    return ic_wire_pairs_jax(state, endpoints, cfg=cfg.wire)


def ic_wire_pairs_jax_exec(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICExecutionResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_pairs_jax with execution bundle."""
    return ic_wire_pairs_jax(state, endpoints, cfg=cfg.graph.wire)


def ic_wire_pairs_jax_runtime(
    state: ICState,
    endpoints: WireEndpoints,
    *,
    cfg: ICRuntimeResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_pairs_jax with runtime bundle."""
    return ic_wire_pairs_jax(state, endpoints, cfg=cfg.graph.wire)


def ic_wire_ptr_pairs_jax_cfg(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptr_pairs_jax with safety policy."""
    wire_cfg = resolve_graph_config(cfg).wire
    return ic_wire_ptr_pairs_jax(
        state,
        ptrs,
        cfg=wire_cfg,
    )


def ic_wire_ptr_pairs_jax_resolved(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICGraphResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptr_pairs_jax with resolved graph cfg."""
    return ic_wire_ptr_pairs_jax(state, ptrs, cfg=cfg.wire)


def ic_wire_ptr_pairs_jax_exec(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICExecutionResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptr_pairs_jax with execution bundle."""
    return ic_wire_ptr_pairs_jax(state, ptrs, cfg=cfg.graph.wire)


def ic_wire_ptr_pairs_jax_runtime(
    state: ICState,
    ptrs: WirePtrPair,
    *,
    cfg: ICRuntimeResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_ptr_pairs_jax with runtime bundle."""
    return ic_wire_ptr_pairs_jax(state, ptrs, cfg=cfg.graph.wire)


def ic_wire_star_jax_cfg(
    state: ICState,
    endpoints: WireStarEndpoints,
    *,
    cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG,
) -> ICState:
    """Interface/Control wrapper for ic_wire_star_jax with safety policy."""
    wire_cfg = resolve_graph_config(cfg).wire
    return ic_wire_star_jax(
        state,
        endpoints,
        cfg=wire_cfg,
    )


def ic_wire_star_jax_resolved(
    state: ICState,
    endpoints: WireStarEndpoints,
    *,
    cfg: ICGraphResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_star_jax with resolved graph cfg."""
    return ic_wire_star_jax(state, endpoints, cfg=cfg.wire)


def ic_wire_star_jax_exec(
    state: ICState,
    endpoints: WireStarEndpoints,
    *,
    cfg: ICExecutionResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_star_jax with execution bundle."""
    return ic_wire_star_jax(state, endpoints, cfg=cfg.graph.wire)


def ic_wire_star_jax_runtime(
    state: ICState,
    endpoints: WireStarEndpoints,
    *,
    cfg: ICRuntimeResolved,
) -> ICState:
    """Interface/Control wrapper for ic_wire_star_jax with runtime bundle."""
    return ic_wire_star_jax(state, endpoints, cfg=cfg.graph.wire)


def ic_find_active_pairs_cfg(
    state: ICState, *, cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for active pair detection with DI bundle."""
    scan_cfg = resolve_graph_config(cfg).scan
    return ic_find_active_pairs(state, cfg=scan_cfg)


def ic_find_active_pairs_resolved(
    state: ICState,
    *,
    cfg: ICGraphResolved,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for active pairs with resolved graph cfg."""
    return ic_find_active_pairs(state, cfg=cfg.scan)


def ic_find_active_pairs_exec(
    state: ICState,
    *,
    cfg: ICExecutionResolved,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for active pairs with execution bundle."""
    return ic_find_active_pairs(state, cfg=cfg.graph.scan)


def ic_find_active_pairs_runtime(
    state: ICState,
    *,
    cfg: ICRuntimeResolved,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for active pairs with runtime bundle."""
    return ic_find_active_pairs(state, cfg=cfg.graph.scan)


def ic_compact_active_pairs_cfg(
    state: ICState, *, cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for compact active pairs with DI bundle."""
    scan_cfg = resolve_graph_config(cfg).scan
    return ic_compact_active_pairs(state, cfg=scan_cfg)


def ic_compact_active_pairs_resolved(
    state: ICState,
    *,
    cfg: ICGraphResolved,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for compact pairs with resolved graph cfg."""
    return ic_compact_active_pairs(state, cfg=cfg.scan)


def ic_compact_active_pairs_exec(
    state: ICState,
    *,
    cfg: ICExecutionResolved,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for compact pairs with execution bundle."""
    return ic_compact_active_pairs(state, cfg=cfg.graph.scan)


def ic_compact_active_pairs_runtime(
    state: ICState,
    *,
    cfg: ICRuntimeResolved,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Interface/Control wrapper for compact pairs with runtime bundle."""
    return ic_compact_active_pairs(state, cfg=cfg.graph.scan)


def ic_compact_active_pairs_result_cfg(
    state: ICState, *, cfg: ICGraphConfig = DEFAULT_GRAPH_CONFIG
):
    """Interface/Control wrapper for CompactResult active pairs with DI bundle."""
    scan_cfg = resolve_graph_config(cfg).scan
    return ic_compact_active_pairs_result(state, cfg=scan_cfg)


def ic_compact_active_pairs_result_resolved(
    state: ICState,
    *,
    cfg: ICGraphResolved,
):
    """Interface/Control wrapper for CompactResult pairs with resolved graph cfg."""
    return ic_compact_active_pairs_result(state, cfg=cfg.scan)


def ic_compact_active_pairs_result_exec(
    state: ICState,
    *,
    cfg: ICExecutionResolved,
):
    """Interface/Control wrapper for CompactResult pairs with execution bundle."""
    return ic_compact_active_pairs_result(state, cfg=cfg.graph.scan)


def ic_compact_active_pairs_result_runtime(
    state: ICState,
    *,
    cfg: ICRuntimeResolved,
):
    """Interface/Control wrapper for CompactResult pairs with runtime bundle."""
    return ic_compact_active_pairs_result(state, cfg=cfg.graph.scan)


def engine_config_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_result_fn=ic_compact_active_pairs_result,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
) -> ICEngineConfig:
    """Build an engine config from a rule config (Interface/Control)."""
    return ICEngineConfig(
        compact_pairs_result_fn=compact_pairs_result_fn,
        decode_port_fn=decode_port_fn,
        alloc_plan_fn=rule_cfg.alloc_plan_fn,
        apply_template_planned_fn=rule_cfg.apply_template_planned_fn,
        halted_fn=halted_fn,
        scan_corrupt_fn=scan_corrupt_fn,
    )


DEFAULT_RUNTIME_CONFIG = runtime_config_from_graph_rule(
    DEFAULT_GRAPH_CONFIG, DEFAULT_RULE_CONFIG
)
DEFAULT_RUNTIME_RESOLVED = resolve_runtime_config(DEFAULT_RUNTIME_CONFIG)
DEFAULT_RUNTIME_OPS = make_runtime_ops(DEFAULT_RUNTIME_RESOLVED)


def apply_active_pairs_jit_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_result_fn=ic_compact_active_pairs_result,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
):
    """Return jitted apply_active_pairs using a rule config."""
    cfg = engine_config_from_rules(
        rule_cfg,
        compact_pairs_result_fn=compact_pairs_result_fn,
        decode_port_fn=decode_port_fn,
        halted_fn=halted_fn,
        scan_corrupt_fn=scan_corrupt_fn,
    )
    return apply_active_pairs_jit(cfg)


def reduce_jit_from_rules(
    rule_cfg: ICRuleConfig,
    *,
    compact_pairs_result_fn=ic_compact_active_pairs_result,
    decode_port_fn=decode_port,
    halted_fn=_halted,
    scan_corrupt_fn=_scan_corrupt_ports,
):
    """Return jitted reduce using a rule config."""
    cfg = engine_config_from_rules(
        rule_cfg,
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
    "ICEngineResolved",
    "ICExecutionConfig",
    "ICExecutionResolved",
    "ICRuntimeConfig",
    "ICRuntimeResolved",
    "ICGraphConfig",
    "ICWireConfig",
    "ICScanConfig",
    "AllocConfig",
    "DEFAULT_ALLOC_CONFIG",
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
    "DEFAULT_ENGINE_RESOLVED",
    "DEFAULT_GRAPH_CONFIG",
    "DEFAULT_GRAPH_RESOLVED",
    "DEFAULT_EXECUTION_CONFIG",
    "DEFAULT_EXECUTION_RESOLVED",
    "DEFAULT_RUNTIME_CONFIG",
    "DEFAULT_RUNTIME_RESOLVED",
    "DEFAULT_RUNTIME_OPS",
    "DEFAULT_WIRE_CONFIG",
    "DEFAULT_SCAN_CONFIG",
    "resolve_wire_config",
    "resolve_scan_config",
    "resolve_graph_config",
    "resolve_graph_cfg",
    "resolve_engine_config",
    "resolve_engine_cfg",
    "resolve_execution_config",
    "resolve_execution_cfg",
    "resolve_execution_cfg_default",
    "resolve_runtime_config",
    "resolve_runtime_cfg",
    "resolve_runtime_cfg_default",
    "make_runtime_ops",
    "make_runtime_ops_from_cfg",
    "ICRuntimeOps",
    "graph_config_with_policy",
    "graph_config_with_policy_binding",
    "graph_config_with_index_fn",
    "graph_config_with_compact_cfg",
    "graph_config_with_guard",
    "execution_config_from_graph_engine",
    "engine_config_with_compact_result_fn",
    "engine_config_with_scan_cfg",
    "engine_config_with_graph_cfg",
    "engine_resolved_with_graph_cfg",
    "runtime_config_from_graph_rule",
    "rule_config_with_alloc",
    "ICGuardConfig",
    "DEFAULT_IC_GUARD_CONFIG",
    "safe_index_1d_cfg",
    "resolve_safe_index_fn",
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
    "ic_wire_jax_resolved",
    "ic_wire_jax_safe_resolved",
    "ic_wire_ptrs_jax_resolved",
    "ic_wire_pairs_jax_resolved",
    "ic_wire_ptr_pairs_jax_resolved",
    "ic_wire_star_jax_resolved",
    "ic_wire_jax_exec",
    "ic_wire_jax_safe_exec",
    "ic_wire_ptrs_jax_exec",
    "ic_wire_pairs_jax_exec",
    "ic_wire_ptr_pairs_jax_exec",
    "ic_wire_star_jax_exec",
    "ic_wire_jax_runtime",
    "ic_wire_jax_safe_runtime",
    "ic_wire_ptrs_jax_runtime",
    "ic_wire_pairs_jax_runtime",
    "ic_wire_ptr_pairs_jax_runtime",
    "ic_wire_star_jax_runtime",
    "ic_find_active_pairs",
    "ic_find_active_pairs_cfg",
    "ic_find_active_pairs_resolved",
    "ic_find_active_pairs_exec",
    "ic_compact_active_pairs",
    "ic_compact_active_pairs_cfg",
    "ic_compact_active_pairs_resolved",
    "ic_compact_active_pairs_exec",
    "ic_compact_active_pairs_result",
    "ic_compact_active_pairs_result_cfg",
    "ic_compact_active_pairs_result_resolved",
    "ic_compact_active_pairs_result_exec",
    "ic_find_active_pairs_runtime",
    "ic_compact_active_pairs_runtime",
    "ic_compact_active_pairs_result_runtime",
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
    "ic_apply_active_pairs_resolved",
    "ic_reduce_resolved",
    "ic_apply_active_pairs_exec",
    "ic_reduce_exec",
    "ic_apply_active_pairs_runtime",
    "ic_reduce_runtime",
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
    "apply_active_pairs_jit_resolved",
    "apply_active_pairs_jit_exec",
    "apply_active_pairs_jit_runtime",
    "reduce_jit_resolved",
    "reduce_jit_exec",
    "reduce_jit_runtime",
    "find_active_pairs_jit_resolved",
    "find_active_pairs_jit_exec",
    "find_active_pairs_jit_runtime",
    "compact_active_pairs_jit_resolved",
    "compact_active_pairs_jit_exec",
    "compact_active_pairs_jit_runtime",
    "compact_active_pairs_result_jit_resolved",
    "compact_active_pairs_result_jit_exec",
    "compact_active_pairs_result_jit_runtime",
    "wire_jax_jit_resolved",
    "wire_jax_jit_exec",
    "wire_jax_jit_runtime",
    "wire_jax_safe_jit_resolved",
    "wire_jax_safe_jit_exec",
    "wire_jax_safe_jit_runtime",
    "wire_ptrs_jit_resolved",
    "wire_ptrs_jit_exec",
    "wire_ptrs_jit_runtime",
    "wire_pairs_jit_resolved",
    "wire_pairs_jit_exec",
    "wire_pairs_jit_runtime",
    "wire_ptr_pairs_jit_resolved",
    "wire_ptr_pairs_jit_exec",
    "wire_ptr_pairs_jit_runtime",
    "wire_star_jit_resolved",
    "wire_star_jit_exec",
    "wire_star_jit_runtime",
    "ic_alloc",
]
