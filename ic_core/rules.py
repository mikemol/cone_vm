import jax
import jax.numpy as jnp
from typing import Tuple

from ic_core.graph import (
    TYPE_FREE,
    TYPE_ERA,
    TYPE_CON,
    TYPE_DUP,
    PORT_PRINCIPAL,
    PORT_AUX_LEFT,
    PORT_AUX_RIGHT,
    ICState,
    _connect_ptrs,
    _alloc2,
    _alloc4,
    _free2,
    _init_nodes,
    _safe_uint32,
    _halted,
    encode_port,
    decode_port,
)
from ic_core.config import ICRuleConfig

RULE_ALLOC_ANNIHILATE = jnp.uint32(0)
RULE_ALLOC_ERASE = jnp.uint32(2)
RULE_ALLOC_COMMUTE = jnp.uint32(4)

TEMPLATE_NONE = jnp.uint32(0)
TEMPLATE_ANNIHILATE = jnp.uint32(1)
TEMPLATE_ERASE = jnp.uint32(2)
TEMPLATE_COMMUTE = jnp.uint32(3)

RULE_TABLE = jnp.array(
    [
        # FREE with anything is no-op.
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        # ERA interactions.
        [[0, 0], [0, 1], [2, 2], [2, 2]],
        # CON interactions.
        [[0, 0], [2, 2], [0, 1], [4, 3]],
        # DUP interactions.
        [[0, 0], [2, 2], [4, 3], [0, 1]],
    ],
    dtype=jnp.uint32,
)


def ic_rule_for_types(
    type_a: jnp.ndarray, type_b: jnp.ndarray, *, rule_table=RULE_TABLE
) -> jnp.ndarray:
    """Lookup rule vector [alloc_count, template_id] for a type pair."""
    a = type_a.astype(jnp.uint32)
    b = type_b.astype(jnp.uint32)
    return rule_table[a, b]


def ic_rule_for_types_cfg(
    type_a: jnp.ndarray,
    type_b: jnp.ndarray,
    *,
    cfg: ICRuleConfig | None = None,
) -> jnp.ndarray:
    """Interface/Control wrapper for rule lookup with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    return cfg.rule_for_types_fn(type_a, type_b)


def ic_select_template(
    state: ICState,
    node_a: int,
    node_b: int,
    *,
    rule_for_types_fn=ic_rule_for_types,
) -> jnp.ndarray:
    type_a = state.node_type[node_a]
    type_b = state.node_type[node_b]
    return rule_for_types_fn(type_a, type_b)[1]


def ic_select_template_cfg(
    state: ICState,
    node_a: int,
    node_b: int,
    *,
    cfg: ICRuleConfig | None = None,
) -> jnp.ndarray:
    """Interface/Control wrapper for template selection with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    type_a = state.node_type[node_a]
    type_b = state.node_type[node_b]
    return cfg.rule_for_types_fn(type_a, type_b)[1]


def ic_apply_annihilate(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    *,
    connect_ptrs_fn=_connect_ptrs,
) -> ICState:
    ports = state.ports
    a_left = ports[node_a, 1]
    a_right = ports[node_a, 2]
    b_left = ports[node_b, 1]
    b_right = ports[node_b, 2]
    ports = connect_ptrs_fn(ports, a_left, b_left)
    ports = connect_ptrs_fn(ports, a_right, b_right)
    ports = ports.at[node_a].set(jnp.uint32(0))
    ports = ports.at[node_b].set(jnp.uint32(0))
    node_type = state.node_type
    node_type = node_type.at[node_a].set(TYPE_FREE)
    node_type = node_type.at[node_b].set(TYPE_FREE)
    return state._replace(ports=ports, node_type=node_type)


def ic_apply_annihilate_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    *,
    cfg: ICRuleConfig | None = None,
) -> ICState:
    """Interface/Control wrapper for annihilate with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    return cfg.apply_annihilate_fn(state, node_a, node_b)


def ic_apply_erase(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    *,
    alloc2_fn=_alloc2,
    init_nodes_fn=_init_nodes,
    free2_fn=_free2,
    connect_ptrs_fn=_connect_ptrs,
    encode_port_fn=encode_port,
) -> ICState:
    type_a = state.node_type[node_a]
    is_era_a = type_a == TYPE_ERA
    era = jnp.where(is_era_a, node_a, node_b).astype(jnp.uint32)
    target = jnp.where(is_era_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    aux_left = ports[target, 1]
    aux_right = ports[target, 2]
    state2, eras = alloc2_fn(state)

    def _do(s):
        s = init_nodes_fn(s, eras, TYPE_ERA)
        ports = s.ports
        ports = connect_ptrs_fn(
            ports, encode_port_fn(eras[0], PORT_PRINCIPAL), aux_left
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(eras[1], PORT_PRINCIPAL), aux_right
        )
        ports = ports.at[era].set(jnp.uint32(0))
        ports = ports.at[target].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[era].set(TYPE_FREE)
        node_type = node_type.at[target].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return free2_fn(s, jnp.stack([era, target]).astype(jnp.uint32))

    return jax.lax.cond(state2.oom | state2.corrupt, lambda s: s, _do, state2)


def ic_apply_commute_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    *,
    cfg: ICRuleConfig | None = None,
) -> ICState:
    """Interface/Control wrapper for commute with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    return cfg.apply_commute_fn(state, node_a, node_b)


def ic_apply_erase_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    *,
    cfg: ICRuleConfig | None = None,
) -> ICState:
    """Interface/Control wrapper for erase with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    return cfg.apply_erase_fn(state, node_a, node_b)


def _ic_apply_erase_with_ids(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    eras: jnp.ndarray,
    *,
    init_nodes_fn=_init_nodes,
    free2_fn=_free2,
    connect_ptrs_fn=_connect_ptrs,
    encode_port_fn=encode_port,
) -> ICState:
    type_a = state.node_type[node_a]
    is_era_a = type_a == TYPE_ERA
    era = jnp.where(is_era_a, node_a, node_b).astype(jnp.uint32)
    target = jnp.where(is_era_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    aux_left = ports[target, 1]
    aux_right = ports[target, 2]

    def _do(s):
        s = init_nodes_fn(s, eras, TYPE_ERA)
        ports = s.ports
        ports = connect_ptrs_fn(
            ports, encode_port_fn(eras[0], PORT_PRINCIPAL), aux_left
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(eras[1], PORT_PRINCIPAL), aux_right
        )
        ports = ports.at[era].set(jnp.uint32(0))
        ports = ports.at[target].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[era].set(TYPE_FREE)
        node_type = node_type.at[target].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return free2_fn(s, jnp.stack([era, target]).astype(jnp.uint32))

    return jax.lax.cond(state.oom | state.corrupt, lambda s: s, _do, state)


def ic_apply_commute(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    *,
    alloc4_fn=_alloc4,
    init_nodes_fn=_init_nodes,
    free2_fn=_free2,
    connect_ptrs_fn=_connect_ptrs,
    encode_port_fn=encode_port,
) -> ICState:
    type_a = state.node_type[node_a]
    is_con_a = type_a == TYPE_CON
    con = jnp.where(is_con_a, node_a, node_b).astype(jnp.uint32)
    dup = jnp.where(is_con_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    con_left = ports[con, 1]
    con_right = ports[con, 2]
    dup_left = ports[dup, 1]
    dup_right = ports[dup, 2]
    state2, ids4 = alloc4_fn(state)

    def _do(s):
        dup_nodes = ids4[:2]
        con_nodes = ids4[2:]
        s = init_nodes_fn(s, con_nodes, TYPE_CON)
        s = init_nodes_fn(s, dup_nodes, TYPE_DUP)
        c0, c1 = con_nodes
        d0, d1 = dup_nodes
        ports = s.ports
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c0, PORT_PRINCIPAL),
            encode_port_fn(d0, PORT_PRINCIPAL),
        )
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c1, PORT_PRINCIPAL),
            encode_port_fn(d1, PORT_PRINCIPAL),
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(c0, PORT_AUX_LEFT), dup_left
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(c1, PORT_AUX_LEFT), dup_right
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(d0, PORT_AUX_LEFT), con_left
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(d1, PORT_AUX_LEFT), con_right
        )
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c0, PORT_AUX_RIGHT),
            encode_port_fn(d0, PORT_AUX_RIGHT),
        )
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c1, PORT_AUX_RIGHT),
            encode_port_fn(d1, PORT_AUX_RIGHT),
        )
        ports = ports.at[con].set(jnp.uint32(0))
        ports = ports.at[dup].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[con].set(TYPE_FREE)
        node_type = node_type.at[dup].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return free2_fn(s, jnp.stack([con, dup]).astype(jnp.uint32))

    return jax.lax.cond(state2.oom | state2.corrupt, lambda s: s, _do, state2)


def _ic_apply_commute_with_ids(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    ids4: jnp.ndarray,
    *,
    init_nodes_fn=_init_nodes,
    free2_fn=_free2,
    connect_ptrs_fn=_connect_ptrs,
    encode_port_fn=encode_port,
) -> ICState:
    type_a = state.node_type[node_a]
    is_con_a = type_a == TYPE_CON
    con = jnp.where(is_con_a, node_a, node_b).astype(jnp.uint32)
    dup = jnp.where(is_con_a, node_b, node_a).astype(jnp.uint32)
    ports = state.ports
    con_left = ports[con, 1]
    con_right = ports[con, 2]
    dup_left = ports[dup, 1]
    dup_right = ports[dup, 2]

    def _do(s):
        dup_nodes = ids4[:2]
        con_nodes = ids4[2:]
        s = init_nodes_fn(s, con_nodes, TYPE_CON)
        s = init_nodes_fn(s, dup_nodes, TYPE_DUP)
        c0, c1 = con_nodes
        d0, d1 = dup_nodes
        ports = s.ports
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c0, PORT_PRINCIPAL),
            encode_port_fn(d0, PORT_PRINCIPAL),
        )
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c1, PORT_PRINCIPAL),
            encode_port_fn(d1, PORT_PRINCIPAL),
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(c0, PORT_AUX_LEFT), dup_left
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(c1, PORT_AUX_LEFT), dup_right
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(d0, PORT_AUX_LEFT), con_left
        )
        ports = connect_ptrs_fn(
            ports, encode_port_fn(d1, PORT_AUX_LEFT), con_right
        )
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c0, PORT_AUX_RIGHT),
            encode_port_fn(d0, PORT_AUX_RIGHT),
        )
        ports = connect_ptrs_fn(
            ports,
            encode_port_fn(c1, PORT_AUX_RIGHT),
            encode_port_fn(d1, PORT_AUX_RIGHT),
        )
        ports = ports.at[con].set(jnp.uint32(0))
        ports = ports.at[dup].set(jnp.uint32(0))
        node_type = s.node_type
        node_type = node_type.at[con].set(TYPE_FREE)
        node_type = node_type.at[dup].set(TYPE_FREE)
        s = s._replace(node_type=node_type, ports=ports)
        return free2_fn(s, jnp.stack([con, dup]).astype(jnp.uint32))

    return jax.lax.cond(state.oom | state.corrupt, lambda s: s, _do, state)


def ic_apply_template(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    template_id: jnp.ndarray,
    *,
    apply_annihilate_fn=ic_apply_annihilate,
    apply_erase_fn=ic_apply_erase,
    apply_commute_fn=ic_apply_commute,
) -> ICState:
    template_id = template_id.astype(jnp.int32)

    def _noop(s):
        return s

    def _ann(s):
        return apply_annihilate_fn(s, node_a, node_b)

    def _erase(s):
        return apply_erase_fn(s, node_a, node_b)

    def _comm(s):
        return apply_commute_fn(s, node_a, node_b)

    def _apply(s):
        return jax.lax.switch(template_id, (_noop, _ann, _erase, _comm), s)

    return jax.lax.cond(state.oom | state.corrupt, _noop, _apply, state)


def ic_apply_template_planned_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    template_id: jnp.ndarray,
    alloc_ids: jnp.ndarray,
    *,
    cfg: ICRuleConfig | None = None,
):
    """Interface/Control wrapper for planned template apply with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    return cfg.apply_template_planned_fn(
        state, node_a, node_b, template_id, alloc_ids
    )


def ic_apply_template_cfg(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    template_id: jnp.ndarray,
    *,
    cfg: ICRuleConfig | None = None,
) -> ICState:
    """Interface/Control wrapper for apply_template with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    return cfg.apply_template_fn(state, node_a, node_b, template_id)


def _alloc_plan(
    state: ICState,
    pairs: jnp.ndarray,
    count: jnp.ndarray,
    *,
    decode_port_fn=decode_port,
    rule_table=RULE_TABLE,
    safe_uint32_fn=_safe_uint32,
    halted_fn=_halted,
) -> Tuple[ICState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = pairs.shape[0]
    zeros4 = jnp.zeros((4,), dtype=jnp.uint32)
    zeros_ids = jnp.zeros((n, 4), dtype=jnp.uint32)
    zeros_counts = jnp.zeros((n,), dtype=jnp.uint32)
    zeros_templates = jnp.full((n,), TEMPLATE_NONE, dtype=jnp.uint32)
    zeros_valid = jnp.zeros((n,), dtype=jnp.bool_)

    def _halt(s):
        return s, zeros_templates, zeros_counts, zeros_ids, zeros_valid

    def _run(s):
        idx = jnp.arange(n, dtype=jnp.uint32)
        valid = idx < count
        node_a = jnp.where(valid, pairs, jnp.uint32(0))
        node_b = decode_port_fn(s.ports[node_a, PORT_PRINCIPAL])[0]
        rule = rule_table[
            s.node_type[node_a].astype(jnp.uint32),
            s.node_type[node_b].astype(jnp.uint32),
        ]
        alloc_counts = jnp.where(valid, rule[:, 0], jnp.uint32(0))
        template_ids = jnp.where(valid, rule[:, 1], TEMPLATE_NONE)
        offsets = jnp.cumsum(alloc_counts) - alloc_counts
        total_alloc = jnp.sum(alloc_counts).astype(jnp.int32)
        free_top = s.free_top.astype(jnp.int32)
        ok = (free_top >= total_alloc) & (~halted_fn(s))
        base = free_top - total_alloc
        base_safe = jnp.where(ok, base, jnp.int32(0))
        free_top_new = jnp.where(ok, safe_uint32_fn(base), s.free_top)
        oom_new = jnp.where(ok, s.oom, jnp.bool_(True))
        state2 = s._replace(free_top=free_top_new, oom=oom_new)

        alloc_ids = jnp.zeros((n, 4), dtype=jnp.uint32)

        def build(i, buf):
            count_i = alloc_counts[i]
            offset = offsets[i].astype(jnp.int32)
            start = base_safe + offset

            def take2(_):
                ids2 = jax.lax.dynamic_slice(s.free_stack, (start,), (2,))
                return jnp.concatenate([ids2, zeros4[:2]], axis=0)

            def take4(_):
                return jax.lax.dynamic_slice(s.free_stack, (start,), (4,))

            ids = jax.lax.cond(
                count_i == jnp.uint32(2),
                take2,
                lambda _: jax.lax.cond(
                    count_i == jnp.uint32(4),
                    take4,
                    lambda __: zeros4,
                    operand=None,
                ),
                operand=None,
            )
            return buf.at[i].set(ids)

        alloc_ids = jax.lax.cond(
            ok,
            lambda _: jax.lax.fori_loop(0, n, build, alloc_ids),
            lambda _: alloc_ids,
            operand=None,
        )
        return state2, template_ids, alloc_counts, alloc_ids, valid

    return jax.lax.cond(halted_fn(state), _halt, _run, state)


def ic_alloc_plan_cfg(
    state: ICState,
    pairs: jnp.ndarray,
    count: jnp.ndarray,
    *,
    cfg: ICRuleConfig | None = None,
):
    """Interface/Control wrapper for alloc plan with DI bundle."""
    if cfg is None:
        cfg = DEFAULT_RULE_CONFIG
    return cfg.alloc_plan_fn(state, pairs, count)


def _apply_template_planned(
    state: ICState,
    node_a: jnp.ndarray,
    node_b: jnp.ndarray,
    template_id: jnp.ndarray,
    alloc_ids: jnp.ndarray,
    *,
    apply_annihilate_fn=ic_apply_annihilate,
    apply_erase_with_ids_fn=_ic_apply_erase_with_ids,
    apply_commute_with_ids_fn=_ic_apply_commute_with_ids,
) -> ICState:
    template_id = template_id.astype(jnp.int32)

    def _noop(s):
        return s

    def _ann(s):
        return apply_annihilate_fn(s, node_a, node_b)

    def _erase(s):
        return apply_erase_with_ids_fn(s, node_a, node_b, alloc_ids[:2])

    def _comm(s):
        return apply_commute_with_ids_fn(s, node_a, node_b, alloc_ids)

    def _apply(s):
        return jax.lax.switch(template_id, (_noop, _ann, _erase, _comm), s)

    return jax.lax.cond(state.oom | state.corrupt, _noop, _apply, state)


DEFAULT_RULE_CONFIG = ICRuleConfig(
    rule_for_types_fn=ic_rule_for_types,
    apply_annihilate_fn=ic_apply_annihilate,
    apply_erase_fn=ic_apply_erase,
    apply_commute_fn=ic_apply_commute,
    apply_template_fn=ic_apply_template,
    alloc_plan_fn=_alloc_plan,
    apply_template_planned_fn=_apply_template_planned,
)


__all__ = [
    "RULE_ALLOC_ANNIHILATE",
    "RULE_ALLOC_ERASE",
    "RULE_ALLOC_COMMUTE",
    "TEMPLATE_NONE",
    "TEMPLATE_ANNIHILATE",
    "TEMPLATE_ERASE",
    "TEMPLATE_COMMUTE",
    "RULE_TABLE",
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
    "_alloc_plan",
    "ic_alloc_plan_cfg",
    "_apply_template_planned",
    "ic_apply_template_planned_cfg",
    "DEFAULT_RULE_CONFIG",
]
