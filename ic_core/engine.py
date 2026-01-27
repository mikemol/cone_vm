import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from ic_core.graph import (
    ICState,
    PORT_PRINCIPAL,
    decode_port,
    ic_compact_active_pairs,
    _halted,
    _scan_corrupt_ports,
)
from ic_core.rules import TEMPLATE_NONE, _alloc_plan, _apply_template_planned


class ICRewriteStats(NamedTuple):
    active_pairs: jnp.ndarray
    alloc_nodes: jnp.ndarray
    freed_nodes: jnp.ndarray
    template_counts: jnp.ndarray


@jax.jit
def ic_apply_active_pairs(state: ICState) -> Tuple[ICState, ICRewriteStats]:
    state = _scan_corrupt_ports(state)
    zero_stats = ICRewriteStats(
        active_pairs=jnp.uint32(0),
        alloc_nodes=jnp.uint32(0),
        freed_nodes=jnp.uint32(0),
        template_counts=jnp.zeros((4,), dtype=jnp.uint32),
    )

    def _halt(s):
        return s, zero_stats

    def _run(s):
        pairs, count, _ = ic_compact_active_pairs(s)
        count_i = count.astype(jnp.int32)

        def body(i, carry):
            s_in, alloc, freed, tmpl_counts, tmpl_ids, alloc_counts, alloc_ids = carry
            node_a = pairs[i]
            node_b = decode_port(s_in.ports[node_a, PORT_PRINCIPAL])[0]
            tmpl = tmpl_ids[i]
            s2 = _apply_template_planned(s_in, node_a, node_b, tmpl, alloc_ids[i])
            ok = (~s_in.oom) & (~s_in.corrupt) & (~s2.oom) & (~s2.corrupt)
            tmpl_i = tmpl.astype(jnp.int32)
            tmpl_counts = tmpl_counts.at[tmpl_i].add(ok.astype(jnp.uint32))
            alloc_delta = jnp.where(ok, alloc_counts[i], jnp.uint32(0))
            freed_delta = jnp.where(
                (tmpl != TEMPLATE_NONE) & ok, jnp.uint32(2), jnp.uint32(0)
            )
            return (
                s2,
                alloc + alloc_delta,
                freed + freed_delta,
                tmpl_counts,
                tmpl_ids,
                alloc_counts,
                alloc_ids,
            )

        def _apply(s_in):
            s2, tmpl_ids, alloc_counts, alloc_ids, _ = _alloc_plan(s_in, pairs, count)

            def _run_plan(s_plan):
                init = (
                    s_plan,
                    jnp.uint32(0),
                    jnp.uint32(0),
                    zero_stats.template_counts,
                    tmpl_ids,
                    alloc_counts,
                    alloc_ids,
                )
                s_out, alloc, freed, tmpl_counts, _, _, _ = jax.lax.fori_loop(
                    0, count_i, body, init
                )
                stats = ICRewriteStats(
                    active_pairs=count,
                    alloc_nodes=alloc,
                    freed_nodes=freed,
                    template_counts=tmpl_counts,
                )
                return s_out, stats

            return jax.lax.cond(
                s2.oom | s2.corrupt,
                lambda s_plan: (s_plan, zero_stats),
                _run_plan,
                s2,
            )

        return jax.lax.cond(
            count_i == 0, lambda s_in: (s_in, zero_stats), _apply, s
        )

    return jax.lax.cond(_halted(state), _halt, _run, state)


@jax.jit
def ic_reduce(
    state: ICState, max_steps: int
) -> Tuple[ICState, ICRewriteStats, jnp.ndarray]:
    state = _scan_corrupt_ports(state)
    max_steps_i = jnp.int32(max_steps)
    zero_stats = ICRewriteStats(
        active_pairs=jnp.uint32(0),
        alloc_nodes=jnp.uint32(0),
        freed_nodes=jnp.uint32(0),
        template_counts=jnp.zeros((4,), dtype=jnp.uint32),
    )

    def cond(carry):
        s, stats, steps, last_active = carry
        return (
            (steps < max_steps_i)
            & (last_active > 0)
            & (~s.oom)
            & (~s.corrupt)
        )

    def body(carry):
        s, stats, steps, _ = carry
        s2, batch = ic_apply_active_pairs(s)
        stats = ICRewriteStats(
            active_pairs=stats.active_pairs + batch.active_pairs,
            alloc_nodes=stats.alloc_nodes + batch.alloc_nodes,
            freed_nodes=stats.freed_nodes + batch.freed_nodes,
            template_counts=stats.template_counts + batch.template_counts,
        )
        return s2, stats, steps + 1, batch.active_pairs

    init = (state, zero_stats, jnp.int32(0), jnp.int32(1))
    s_out, stats_out, steps_out, _ = jax.lax.while_loop(cond, body, init)
    return s_out, stats_out, steps_out


__all__ = ["ICRewriteStats", "ic_apply_active_pairs", "ic_reduce"]
