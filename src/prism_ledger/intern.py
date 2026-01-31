from dataclasses import dataclass

# dataflow-bundle: a0, a1, a2, a3, b0, b1, b2, b3
# dataflow-bundle: a4, b4
# dataflow-bundle: coord_norm_probe_assert_fn, coord_norm_probe_reset_cb_fn
# dataflow-bundle: new_ids, new_keys
# dataflow-bundle: t_b0, t_b1, t_b2, t_b3

import jax
import jax.numpy as jnp
from jax import lax

from prism_core import jax_safe as _jax_safe
from prism_core.permutation import _invert_perm
from prism_ledger.config import DEFAULT_INTERN_CONFIG, InternConfig
from prism_metrics.probes import (
    _coord_norm_probe_assert,
    _coord_norm_probe_enabled,
    _coord_norm_probe_reset_cb,
    _coord_norm_probe_tick,
)
from prism_vm_core.candidates import _candidate_indices
from prism_vm_core.constants import MAX_COORD_STEPS, MAX_COUNT
from prism_vm_core.guards import (
    _guard_max,
    _guard_zero_args,
    _guard_zero_row,
    _guards_enabled,
)
from prism_vm_core.ledger_keys import _checked_pack_key, _pack_key
from prism_vm_core.ontology import (
    OP_ADD,
    OP_COORD_ONE,
    OP_COORD_PAIR,
    OP_COORD_ZERO,
    OP_MUL,
    OP_NULL,
)
from prism_vm_core.structures import Ledger, NodeBatch

_scatter_drop = _jax_safe.scatter_drop


@dataclass(frozen=True, slots=True)
class KeyColumns:
    """Bundle of sorted key columns."""

    b0: jnp.ndarray
    b1: jnp.ndarray
    b2: jnp.ndarray
    b3: jnp.ndarray
    b4: jnp.ndarray

def _coord_norm_id_jax_core(ledger, coord_id, *, lookup_node_id_fn=None):
    # CDᵣ + Normalize𝚌 (core, no probe)
    # NOTE: repeated lookups per step are an m1/m4 tradeoff; batching is
    # tracked in IMPLEMENTATION_PLAN.md.
    lookup_node_id_fn = lookup_node_id_fn or _lookup_node_id
    leaf_zero_id, leaf_zero_found = lookup_node_id_fn(
        ledger,
        jnp.int32(OP_COORD_ZERO),
        jnp.int32(0),
        jnp.int32(0),
    )
    leaf_one_id, leaf_one_found = lookup_node_id_fn(
        ledger,
        jnp.int32(OP_COORD_ONE),
        jnp.int32(0),
        jnp.int32(0),
    )

    def body(_, cid):
        op = ledger.opcode[cid]
        is_zero = op == OP_COORD_ZERO
        is_one = op == OP_COORD_ONE
        is_pair = op == OP_COORD_PAIR
        cid = jnp.where(is_zero & leaf_zero_found, leaf_zero_id, cid)
        cid = jnp.where(is_one & leaf_one_found, leaf_one_id, cid)

        left = ledger.arg1[cid]
        right = ledger.arg2[cid]
        left_op = ledger.opcode[left]
        right_op = ledger.opcode[right]
        left = jnp.where(
            (left_op == OP_COORD_ZERO) & leaf_zero_found, leaf_zero_id, left
        )
        left = jnp.where(
            (left_op == OP_COORD_ONE) & leaf_one_found, leaf_one_id, left
        )
        right = jnp.where(
            (right_op == OP_COORD_ZERO) & leaf_zero_found, leaf_zero_id, right
        )
        right = jnp.where(
            (right_op == OP_COORD_ONE) & leaf_one_found, leaf_one_id, right
        )

        pair_id, pair_found = lookup_node_id_fn(
            ledger, jnp.int32(OP_COORD_PAIR), left, right
        )
        cid = jnp.where(is_pair & pair_found, pair_id, cid)
        return cid

    return lax.fori_loop(0, MAX_COORD_STEPS, body, coord_id)


def _coord_norm_id_jax(
    ledger,
    coord_id,
    *,
    coord_norm_id_core_fn=None,
    lookup_node_id_fn=None,
    guards_enabled_fn=_guards_enabled,
    probe_tick_fn=_coord_norm_probe_tick,
):
    # CDᵣ + Normalize𝚌
    # Debug-only probe to detect normalization scope; see tests/test_coord_norm_probe.py.
    if guards_enabled_fn():
        jax.debug.callback(probe_tick_fn, jnp.int32(1), ordered=True)
    coord_norm_id_core_fn = coord_norm_id_core_fn or _coord_norm_id_jax_core
    return coord_norm_id_core_fn(
        ledger, coord_id, lookup_node_id_fn=lookup_node_id_fn
    )


def _coord_norm_id_jax_noprobe(
    ledger, coord_id, *, coord_norm_id_core_fn=None, lookup_node_id_fn=None
):
    coord_norm_id_core_fn = coord_norm_id_core_fn or _coord_norm_id_jax_core
    return coord_norm_id_core_fn(
        ledger, coord_id, lookup_node_id_fn=lookup_node_id_fn
    )


def _lookup_node_id(ledger, op, a1, a2, *, pack_key_fn=_pack_key):
    k0, k1, k2, k3, k4 = pack_key_fn(op, a1, a2)
    L_b0 = ledger.keys_b0_sorted
    L_b1 = ledger.keys_b1_sorted
    L_b2 = ledger.keys_b2_sorted
    L_b3 = ledger.keys_b3_sorted
    L_b4 = ledger.keys_b4_sorted
    L_ids = ledger.ids_sorted
    count = ledger.count.astype(jnp.int32)

    def _lex_less(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4):
        return jnp.logical_or(
            a0 < b0,
            jnp.logical_and(
                a0 == b0,
                jnp.logical_or(
                    a1 < b1,
                    jnp.logical_and(
                        a1 == b1,
                        jnp.logical_or(
                            a2 < b2,
                            jnp.logical_and(
                                a2 == b2,
                                jnp.logical_or(
                                    a3 < b3, jnp.logical_and(a3 == b3, a4 < b4)
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def _do_search(_):
        lo = jnp.int32(0)
        hi = count

        def cond(state):
            lo_i, hi_i = state
            return lo_i < hi_i

        def body(state):
            lo_i, hi_i = state
            mid = (lo_i + hi_i) // 2
            mid_b0 = L_b0[mid]
            mid_b1 = L_b1[mid]
            mid_b2 = L_b2[mid]
            mid_b3 = L_b3[mid]
            mid_b4 = L_b4[mid]
            go_right = _lex_less(
                mid_b0,
                mid_b1,
                mid_b2,
                mid_b3,
                mid_b4,
                k0,
                k1,
                k2,
                k3,
                k4,
            )
            lo_i = jnp.where(go_right, mid + 1, lo_i)
            hi_i = jnp.where(go_right, hi_i, mid)
            return (lo_i, hi_i)

        # NOTE: lax.while_loop returns (lo, hi); pos is the final lo bound.
        # hi_final is unused but kept explicit for readability.
        pos, hi_final = lax.while_loop(cond, body, (lo, hi))
        _ = hi_final
        safe_pos = jnp.minimum(pos, count - 1)
        # safe_pos is bounds-only; treat as valid only when pos < count.
        # count > 0 is enforced by the outer cond; keep that guard if refactoring.
        # See IMPLEMENTATION_PLAN.md.
        found = (
            (pos < count)
            & (L_b0[safe_pos] == k0)
            & (L_b1[safe_pos] == k1)
            & (L_b2[safe_pos] == k2)
            & (L_b3[safe_pos] == k3)
            & (L_b4[safe_pos] == k4)
        )
        out_id = jnp.where(found, L_ids[safe_pos], jnp.int32(0))
        return out_id, found

    return lax.cond(
        count > 0,
        _do_search,
        lambda _: (jnp.int32(0), jnp.bool_(False)),
        operand=None,
    )


def _key_safe_normalize_nodes(
    ops, a1, a2, *, guard_zero_args_fn=_guard_zero_args
):
    is_null = ops == OP_NULL
    is_coord_leaf = (ops == OP_COORD_ZERO) | (ops == OP_COORD_ONE)
    zero_mask = is_null | is_coord_leaf
    guard_zero_args_fn(
        is_coord_leaf, a1, a2, "key_safe_normalize.coord_leaf_args"
    )
    a1 = jnp.where(zero_mask, jnp.int32(0), a1)
    a2 = jnp.where(zero_mask, jnp.int32(0), a2)
    # NOTE: OP_COORD_PAIR is treated as ordered; no commutative swap here.
    swap = (ops == OP_MUL) | (ops == OP_ADD)
    swap = swap & (a2 < a1)
    a1_swapped = jnp.where(swap, a2, a1)
    a2_swapped = jnp.where(swap, a1, a2)
    return ops, a1_swapped, a2_swapped


def _intern_nodes_impl_core(
    ledger,
    proposed_ops,
    proposed_a1,
    proposed_a2,
    *,
    cfg: InternConfig,
    key_safe_normalize_fn=_key_safe_normalize_nodes,
    pack_key_fn=_pack_key,
    candidate_indices_fn=_candidate_indices,
    guard_max_fn=_guard_max,
    guards_enabled_fn=_guards_enabled,
    coord_norm_id_jax_fn=_coord_norm_id_jax,
    coord_norm_id_jax_noprobe_fn=_coord_norm_id_jax_noprobe,
    coord_norm_probe_enabled_fn=_coord_norm_probe_enabled,
    coord_norm_probe_reset_cb_fn=_coord_norm_probe_reset_cb,
    coord_norm_probe_assert_fn=_coord_norm_probe_assert,
    scatter_drop_fn=_scatter_drop,
):
    max_key = jnp.uint8(0xFF)
    # Interning pipeline (vectorized):
    # - Key-safe normalization only (coord pairs); no semantic rewrites.
    # - Pack/sort keys to dedupe proposals and batch-search ledger buckets.
    # - Allocate new ids and merge sorted key arrays into the ledger.
    # Performance note: interning runs on fixed-shape buffers for JIT stability,
    # so some passes touch LEDGER_CAPACITY even when count is small (m1 tradeoff).
    # See IMPLEMENTATION_PLAN.md (m4) for the mitigation roadmap.
    # CORRUPT is semantic (alias risk); OOM is capacity.
    base_corrupt = ledger.corrupt
    base_oom = ledger.oom
    proposed_ops = jnp.where(base_corrupt, jnp.int32(0), proposed_ops)
    proposed_a1 = jnp.where(base_corrupt, jnp.int32(0), proposed_a1)
    proposed_a2 = jnp.where(base_corrupt, jnp.int32(0), proposed_a2)
    is_coord_pair = proposed_ops == OP_COORD_PAIR

    has_coord = jnp.any(is_coord_pair)
    if guards_enabled_fn() and coord_norm_probe_enabled_fn():
        jax.debug.callback(coord_norm_probe_reset_cb_fn, jnp.int32(0), ordered=True)
    # CD_r/CD_a: normalize coord pairs before packing keys for stable lookup.

    def _norm(args):
        proposed_a1, proposed_a2 = args
        coord_enabled = is_coord_pair.astype(jnp.int32)
        coord_idx, coord_valid, coord_count = candidate_indices_fn(coord_enabled)

        def _norm_body(i, state):
            a1_arr, a2_arr = state
            idx = coord_idx[i]
            a1_norm = coord_norm_id_jax_fn(ledger, a1_arr[idx])
            a2_norm = coord_norm_id_jax_noprobe_fn(ledger, a2_arr[idx])
            a1_arr = a1_arr.at[idx].set(a1_norm)
            a2_arr = a2_arr.at[idx].set(a2_norm)
            return a1_arr, a2_arr

        proposed_a1, proposed_a2 = lax.fori_loop(
            0, coord_count, _norm_body, (proposed_a1, proposed_a2)
        )
        return proposed_a1, proposed_a2

    def _no_norm(args):
        return args

    proposed_a1, proposed_a2 = lax.cond(
        has_coord, _norm, _no_norm, (proposed_a1, proposed_a2)
    )
    if guards_enabled_fn() and coord_norm_probe_enabled_fn():
        jax.debug.callback(coord_norm_probe_assert_fn, has_coord, ordered=True)

    # Key-safety: Normalize𝚌 before packing.
    # Sort proposals by packed key so duplicates collapse deterministically.
    P_b0, P_b1, P_b2, P_b3, P_b4 = pack_key_fn(
        proposed_ops, proposed_a1, proposed_a2
    )
    perm = jnp.lexsort((P_b4, P_b3, P_b2, P_b1, P_b0)).astype(jnp.int32)

    s_b0 = P_b0[perm]
    s_b1 = P_b1[perm]
    s_b2 = P_b2[perm]
    s_b3 = P_b3[perm]
    s_b4 = P_b4[perm]
    s_ops = proposed_ops[perm]
    s_a1 = proposed_a1[perm]
    s_a2 = proposed_a2[perm]
    new_entry_len = s_b0.shape[0]

    # Mark leader entries for each unique key in sorted order.
    is_diff = jnp.concatenate(
        [
            jnp.array([True]),
            (s_b0[1:] != s_b0[:-1])
            | (s_b1[1:] != s_b1[:-1])
            | (s_b2[1:] != s_b2[:-1])
            | (s_b3[1:] != s_b3[:-1])
            | (s_b4[1:] != s_b4[:-1]),
        ]
    )

    idx = jnp.arange(s_b0.shape[0], dtype=jnp.int32)

    def scan_fn(carry, x):
        is_leader, i = x
        new_carry = jnp.where(is_leader, i, carry)
        return new_carry, new_carry

    _, leader_idx = lax.scan(scan_fn, jnp.int32(0), (is_diff, idx))

    L_b0 = ledger.keys_b0_sorted
    L_b1 = ledger.keys_b1_sorted
    L_b2 = ledger.keys_b2_sorted
    L_b3 = ledger.keys_b3_sorted
    L_b4 = ledger.keys_b4_sorted
    L_ids = ledger.ids_sorted

    count = ledger.count.astype(jnp.int32)
    max_count = jnp.int32(MAX_COUNT)
    available = jnp.maximum(max_count - count, 0)
    available = jnp.where(base_oom | base_corrupt, jnp.int32(0), available)
    if cfg.op_buckets_full_range:
        op_start = jnp.zeros(256, dtype=jnp.int32)
        op_end = jnp.full((256,), count, dtype=jnp.int32)
    else:
        # Bucket existing keys by opcode byte to narrow search ranges.
        # Use searchsorted on the sorted opcode column to avoid full scans.
        op_values = jnp.arange(256, dtype=jnp.uint8)
        op_start = jnp.searchsorted(L_b0, op_values, side="left").astype(jnp.int32)
        op_end = jnp.searchsorted(L_b0, op_values, side="right").astype(jnp.int32)
        op_start = jnp.minimum(op_start, count)
        op_end = jnp.minimum(op_end, count)
    # NOTE: opcode buckets are a precursor to per-op merges; full-array merge
    # remains an m1 tradeoff (see IMPLEMENTATION_PLAN.md).

    def _lex_less(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4):
        return jnp.logical_or(
            a0 < b0,
            jnp.logical_and(
                a0 == b0,
                jnp.logical_or(
                    a1 < b1,
                    jnp.logical_and(
                        a1 == b1,
                        jnp.logical_or(
                            a2 < b2,
                            jnp.logical_and(
                                a2 == b2,
                                jnp.logical_or(
                                    a3 < b3, jnp.logical_and(a3 == b3, a4 < b4)
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def _search_one(t_b0, t_b1, t_b2, t_b3, t_b4, start, end):
        lo = start
        hi = end

        def cond(state):
            lo_i, hi_i = state
            return lo_i < hi_i

        def body(state):
            lo_i, hi_i = state
            mid = (lo_i + hi_i) // 2
            mid_b0 = L_b0[mid]
            mid_b1 = L_b1[mid]
            mid_b2 = L_b2[mid]
            mid_b3 = L_b3[mid]
            mid_b4 = L_b4[mid]
            go_right = _lex_less(
                mid_b0,
                mid_b1,
                mid_b2,
                mid_b3,
                mid_b4,
                t_b0,
                t_b1,
                t_b2,
                t_b3,
                t_b4,
            )
            lo_i = jnp.where(go_right, mid + 1, lo_i)
            hi_i = jnp.where(go_right, hi_i, mid)
            return (lo_i, hi_i)

        lo, _ = lax.while_loop(cond, body, (lo, hi))
        return lo

    op_idx = s_b0.astype(jnp.int32)
    op_lo = op_start[op_idx]
    op_hi = op_end[op_idx]
    insert_pos = jax.vmap(_search_one)(s_b0, s_b1, s_b2, s_b3, s_b4, op_lo, op_hi)
    safe_pos = jnp.minimum(insert_pos, count - 1)
    # Assumes ledger.count > 0 (seeded). If that invariant changes, guard
    # count==0 here; see IMPLEMENTATION_PLAN.md.

    found_match = (
        (insert_pos < count)
        & (L_b0[safe_pos] == s_b0)
        & (L_b1[safe_pos] == s_b1)
        & (L_b2[safe_pos] == s_b2)
        & (L_b3[safe_pos] == s_b3)
        & (L_b4[safe_pos] == s_b4)
    )
    matched_ids = L_ids[safe_pos].astype(jnp.int32)

    is_new = is_diff & (~found_match) & (~(base_oom | base_corrupt))
    requested_new = jnp.sum(is_new.astype(jnp.int32))
    overflow = (count + requested_new) > max_count
    # NOTE: overflow relies on requested_new being accurate; add a secondary
    # guard on num_new if is_new logic changes (see IMPLEMENTATION_PLAN.md).
    if cfg.force_spawn_clip and _jax_safe.TEST_GUARDS:
        # Test-only hook: force a spawn/available mismatch to exercise guardrails.
        available = jnp.maximum(requested_new - jnp.int32(1), 0)

    def _overflow(_):
        zero_ids = jnp.zeros_like(proposed_ops)
        # NOTE: overflow is treated as CORRUPT in m1 because the semantic id
        # cap matches capacity; a distinct OOM path is deferred to the plan.
        # NOTE(m1): capacity == semantic id hard-cap; overflow is CORRUPT.
        # Planned(m?): split OOM vs CORRUPT once semantic cap decouples from storage.
        new_ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))
        return zero_ids, new_ledger

    # Helper defined before _allocate to avoid JAX tracing scoping ambiguity.
    # NOTE: global merge is an m1 tradeoff; performance roadmap is tracked in
    # IMPLEMENTATION_PLAN.md.
    def _merge_sorted_keys(
        old_keys: KeyColumns,
        old_ids,
        old_count,
        new_keys: KeyColumns,
        new_ids,
        new_items,
    ):
        out_b0 = jnp.full_like(old_keys.b0, max_key)
        out_b1 = jnp.full_like(old_keys.b1, max_key)
        out_b2 = jnp.full_like(old_keys.b2, max_key)
        out_b3 = jnp.full_like(old_keys.b3, max_key)
        out_b4 = jnp.full_like(old_keys.b4, max_key)
        out_ids = jnp.zeros_like(old_ids)
        total = old_count + new_items
        guard_max_fn(total, jnp.int32(old_keys.b0.shape[0]), "merge.total")

        def body(k, state):
            i, j, b0, b1, b2, b3, b4, ids = state
            old_valid = i < old_count
            new_valid = j < new_items
            safe_i = jnp.where(old_valid, i, 0)
            safe_j = jnp.where(new_valid, j, 0)

            old0 = jnp.where(old_valid, old_keys.b0[safe_i], max_key)
            old1 = jnp.where(old_valid, old_keys.b1[safe_i], max_key)
            old2 = jnp.where(old_valid, old_keys.b2[safe_i], max_key)
            old3 = jnp.where(old_valid, old_keys.b3[safe_i], max_key)
            old4 = jnp.where(old_valid, old_keys.b4[safe_i], max_key)

            new0 = jnp.where(new_valid, new_keys.b0[safe_j], max_key)
            new1 = jnp.where(new_valid, new_keys.b1[safe_j], max_key)
            new2 = jnp.where(new_valid, new_keys.b2[safe_j], max_key)
            new3 = jnp.where(new_valid, new_keys.b3[safe_j], max_key)
            new4 = jnp.where(new_valid, new_keys.b4[safe_j], max_key)

            new_less = _lex_less(
                new0,
                new1,
                new2,
                new3,
                new4,
                old0,
                old1,
                old2,
                old3,
                old4,
            )
            take_new = jnp.where(old_valid & new_valid, new_less, new_valid)

            picked0 = jnp.where(take_new, new0, old0)
            picked1 = jnp.where(take_new, new1, old1)
            picked2 = jnp.where(take_new, new2, old2)
            picked3 = jnp.where(take_new, new3, old3)
            picked4 = jnp.where(take_new, new4, old4)

            old_id = jnp.where(old_valid, old_ids[safe_i], jnp.int32(0))
            new_id = jnp.where(new_valid, new_ids[safe_j], jnp.int32(0))
            picked_id = jnp.where(take_new, new_id, old_id)

            b0 = b0.at[k].set(picked0)
            b1 = b1.at[k].set(picked1)
            b2 = b2.at[k].set(picked2)
            b3 = b3.at[k].set(picked3)
            b4 = b4.at[k].set(picked4)
            ids = ids.at[k].set(picked_id)

            i = jnp.where(take_new, i, i + 1)
            j = jnp.where(take_new, j + 1, j)
            return (i, j, b0, b1, b2, b3, b4, ids)

        init_state = (
            jnp.int32(0),
            jnp.int32(0),
            out_b0,
            out_b1,
            out_b2,
            out_b3,
            out_b4,
            out_ids,
        )
        _, _, out_b0, out_b1, out_b2, out_b3, out_b4, out_ids = lax.fori_loop(
            0, total, body, init_state
        )
        return out_b0, out_b1, out_b2, out_b3, out_b4, out_ids

    def _merge_sorted_keys_bucketed(
        old_keys: KeyColumns,
        old_ids,
        old_count,
        new_keys: KeyColumns,
        new_ids,
        new_items,
        op_start,
        op_end,
    ):
        out_b0 = jnp.full_like(old_keys.b0, max_key)
        out_b1 = jnp.full_like(old_keys.b1, max_key)
        out_b2 = jnp.full_like(old_keys.b2, max_key)
        out_b3 = jnp.full_like(old_keys.b3, max_key)
        out_b4 = jnp.full_like(old_keys.b4, max_key)
        out_ids = jnp.zeros_like(old_ids)
        total = old_count + new_items
        guard_max_fn(total, jnp.int32(old_keys.b0.shape[0]), "merge.total")

        op_values = jnp.arange(256, dtype=jnp.uint8)
        new_op_start = jnp.searchsorted(new_keys.b0, op_values, side="left").astype(
            jnp.int32
        )
        new_op_end = jnp.searchsorted(new_keys.b0, op_values, side="right").astype(
            jnp.int32
        )
        new_op_start = jnp.minimum(new_op_start, new_items)
        new_op_end = jnp.minimum(new_op_end, new_items)
        new_counts = new_op_end - new_op_start
        prefix_new = jnp.cumsum(new_counts) - new_counts

        def _merge_op(op_idx, state):
            b0, b1, b2, b3, b4, ids = state
            old_lo = op_start[op_idx]
            old_hi = op_end[op_idx]
            new_lo = new_op_start[op_idx]
            new_hi = new_op_end[op_idx]
            old_len = old_hi - old_lo
            new_len = new_hi - new_lo
            total_len = old_len + new_len
            dest_lo = old_lo + prefix_new[op_idx]

            def _merge_body(k, carry):
                i, j, b0, b1, b2, b3, b4, ids = carry
                old_valid = i < old_len
                new_valid = j < new_len
                old_idx = old_lo + i
                new_idx = new_lo + j

                old0 = jnp.where(old_valid, old_keys.b0[old_idx], max_key)
                old1 = jnp.where(old_valid, old_keys.b1[old_idx], max_key)
                old2 = jnp.where(old_valid, old_keys.b2[old_idx], max_key)
                old3 = jnp.where(old_valid, old_keys.b3[old_idx], max_key)
                old4 = jnp.where(old_valid, old_keys.b4[old_idx], max_key)

                new0 = jnp.where(new_valid, new_keys.b0[new_idx], max_key)
                new1 = jnp.where(new_valid, new_keys.b1[new_idx], max_key)
                new2 = jnp.where(new_valid, new_keys.b2[new_idx], max_key)
                new3 = jnp.where(new_valid, new_keys.b3[new_idx], max_key)
                new4 = jnp.where(new_valid, new_keys.b4[new_idx], max_key)

                new_less = _lex_less(
                    new0,
                    new1,
                    new2,
                    new3,
                    new4,
                    old0,
                    old1,
                    old2,
                    old3,
                    old4,
                )
                take_new = jnp.where(old_valid & new_valid, new_less, new_valid)

                picked0 = jnp.where(take_new, new0, old0)
                picked1 = jnp.where(take_new, new1, old1)
                picked2 = jnp.where(take_new, new2, old2)
                picked3 = jnp.where(take_new, new3, old3)
                picked4 = jnp.where(take_new, new4, old4)

                old_id = jnp.where(old_valid, old_ids[old_idx], jnp.int32(0))
                new_id = jnp.where(new_valid, new_ids[new_idx], jnp.int32(0))
                picked_id = jnp.where(take_new, new_id, old_id)

                out_idx = dest_lo + k
                b0 = b0.at[out_idx].set(picked0)
                b1 = b1.at[out_idx].set(picked1)
                b2 = b2.at[out_idx].set(picked2)
                b3 = b3.at[out_idx].set(picked3)
                b4 = b4.at[out_idx].set(picked4)
                ids = ids.at[out_idx].set(picked_id)

                i = jnp.where(take_new, i, i + 1)
                j = jnp.where(take_new, j + 1, j)
                return (i, j, b0, b1, b2, b3, b4, ids)

            def _run_merge(state):
                b0_in, b1_in, b2_in, b3_in, b4_in, ids_in = state
                init = (
                    jnp.int32(0),
                    jnp.int32(0),
                    b0_in,
                    b1_in,
                    b2_in,
                    b3_in,
                    b4_in,
                    ids_in,
                )
                _, _, b0_out, b1_out, b2_out, b3_out, b4_out, ids_out = (
                    lax.fori_loop(0, total_len, _merge_body, init)
                )
                return (b0_out, b1_out, b2_out, b3_out, b4_out, ids_out)

            return lax.cond(total_len > 0, _run_merge, lambda s: s, state)

        init_state = (out_b0, out_b1, out_b2, out_b3, out_b4, out_ids)
        out_b0, out_b1, out_b2, out_b3, out_b4, out_ids = lax.fori_loop(
            0, jnp.int32(256), _merge_op, init_state
        )
        return out_b0, out_b1, out_b2, out_b3, out_b4, out_ids

    def _allocate(_):
        # Allocate new ids (subject to capacity) and write node payloads.
        spawn = is_new.astype(jnp.int32)
        prefix = jnp.cumsum(spawn)
        spawn = spawn * (prefix <= available).astype(jnp.int32)
        is_new_mask = spawn.astype(jnp.bool_)
        offsets = jnp.cumsum(spawn) - spawn
        num_new = jnp.sum(spawn).astype(jnp.int32)
        spawn_mismatch = num_new != requested_new

        def _partial_alloc(_):
            zero_ids = jnp.zeros_like(proposed_ops)
            new_ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))
            return zero_ids, new_ledger

        def _write_alloc(_):
            write_start = ledger.count.astype(jnp.int32)
            new_ids_for_sorted = jnp.where(
                found_match,
                matched_ids,
                jnp.where(is_new_mask, write_start + offsets, jnp.int32(0)),
            )

            leader_ids = jnp.where(is_diff, new_ids_for_sorted, jnp.int32(0))
            ids_sorted_order = leader_ids[leader_idx]

            inv_perm = _invert_perm(perm)
            final_ids = ids_sorted_order[inv_perm]

            new_opcode = ledger.opcode
            new_arg1 = ledger.arg1
            new_arg2 = ledger.arg2

            valid_w = is_new_mask
            safe_w = jnp.where(
                valid_w, write_start + offsets, jnp.int32(new_opcode.shape[0])
            )

            new_opcode = scatter_drop_fn(
                new_opcode,
                safe_w,
                jnp.where(valid_w, s_ops, new_opcode[0]),
                "intern_nodes.new_opcode",
            )
            new_arg1 = scatter_drop_fn(
                new_arg1,
                safe_w,
                jnp.where(valid_w, s_a1, new_arg1[0]),
                "intern_nodes.new_arg1",
            )
            new_arg2 = scatter_drop_fn(
                new_arg2,
                safe_w,
                jnp.where(valid_w, s_a2, new_arg2[0]),
                "intern_nodes.new_arg2",
            )

            new_count = ledger.count + num_new
            new_oom = base_oom
            new_corrupt = base_corrupt
            guard_max_fn(new_count, max_count, "ledger.count")
            guard_max_fn(
                new_count,
                jnp.int32(new_opcode.shape[0]),
                "ledger.backing_array_length",
            )

            valid_new = is_new_mask
            safe_new = jnp.where(valid_new, offsets, jnp.int32(new_entry_len))

            new_entry_b0_sorted = jnp.full_like(s_b0, max_key)
            new_entry_b1_sorted = jnp.full_like(s_b1, max_key)
            new_entry_b2_sorted = jnp.full_like(s_b2, max_key)
            new_entry_b3_sorted = jnp.full_like(s_b3, max_key)
            new_entry_b4_sorted = jnp.full_like(s_b4, max_key)
            new_entry_ids_sorted = jnp.zeros(new_entry_len, dtype=jnp.int32)

            new_entry_b0_sorted = scatter_drop_fn(
                new_entry_b0_sorted,
                safe_new,
                jnp.where(valid_new, s_b0, new_entry_b0_sorted[0]),
                "intern_nodes.new_entry_b0",
            )
            new_entry_b1_sorted = scatter_drop_fn(
                new_entry_b1_sorted,
                safe_new,
                jnp.where(valid_new, s_b1, new_entry_b1_sorted[0]),
                "intern_nodes.new_entry_b1",
            )
            new_entry_b2_sorted = scatter_drop_fn(
                new_entry_b2_sorted,
                safe_new,
                jnp.where(valid_new, s_b2, new_entry_b2_sorted[0]),
                "intern_nodes.new_entry_b2",
            )
            new_entry_b3_sorted = scatter_drop_fn(
                new_entry_b3_sorted,
                safe_new,
                jnp.where(valid_new, s_b3, new_entry_b3_sorted[0]),
                "intern_nodes.new_entry_b3",
            )
            new_entry_b4_sorted = scatter_drop_fn(
                new_entry_b4_sorted,
                safe_new,
                jnp.where(valid_new, s_b4, new_entry_b4_sorted[0]),
                "intern_nodes.new_entry_b4",
            )
            new_entry_ids_sorted = scatter_drop_fn(
                new_entry_ids_sorted,
                safe_new,
                jnp.where(valid_new, new_ids_for_sorted, new_entry_ids_sorted[0]),
                "intern_nodes.new_entry_ids",
            )

            # Merge sorted new keys into the ledger's sorted key arrays.
            old_keys = KeyColumns(L_b0, L_b1, L_b2, L_b3, L_b4)
            new_keys = KeyColumns(
                new_entry_b0_sorted,
                new_entry_b1_sorted,
                new_entry_b2_sorted,
                new_entry_b3_sorted,
                new_entry_b4_sorted,
            )
            (
                new_keys_b0_sorted,
                new_keys_b1_sorted,
                new_keys_b2_sorted,
                new_keys_b3_sorted,
                new_keys_b4_sorted,
                new_ids_sorted,
            ) = _merge_sorted_keys_bucketed(
                old_keys,
                L_ids,
                count,
                new_keys,
                new_entry_ids_sorted,
                num_new,
                op_start,
                op_end,
            )

            new_ledger = Ledger(
                opcode=new_opcode,
                arg1=new_arg1,
                arg2=new_arg2,
                keys_b0_sorted=new_keys_b0_sorted,
                keys_b1_sorted=new_keys_b1_sorted,
                keys_b2_sorted=new_keys_b2_sorted,
                keys_b3_sorted=new_keys_b3_sorted,
                keys_b4_sorted=new_keys_b4_sorted,
                ids_sorted=new_ids_sorted,
                count=new_count,
                oom=new_oom,
                corrupt=new_corrupt,
            )
            return final_ids, new_ledger

        return lax.cond(spawn_mismatch, _partial_alloc, _write_alloc, operand=None)

    return lax.cond(overflow, _overflow, _allocate, operand=None)


def _intern_nodes_impl(
    ledger,
    batch: NodeBatch,
    *,
    cfg: InternConfig,
    intern_core_fn=_intern_nodes_impl_core,
    key_safe_normalize_fn=_key_safe_normalize_nodes,
    guard_zero_args_fn=_guard_zero_args,
    checked_pack_key_fn=_checked_pack_key,
    pack_key_fn=_pack_key,
    candidate_indices_fn=_candidate_indices,
    guard_max_fn=_guard_max,
    guards_enabled_fn=_guards_enabled,
    coord_norm_id_jax_fn=_coord_norm_id_jax,
    coord_norm_id_jax_noprobe_fn=_coord_norm_id_jax_noprobe,
    coord_norm_probe_enabled_fn=_coord_norm_probe_enabled,
    coord_norm_probe_reset_cb_fn=_coord_norm_probe_reset_cb,
    coord_norm_probe_assert_fn=_coord_norm_probe_assert,
    scatter_drop_fn=_scatter_drop,
):
    # Canonical_i: full key equality; only key-safe normalization belongs here.
    proposed_ops, proposed_a1, proposed_a2 = batch
    coord_leaf_mask = (proposed_ops == OP_COORD_ZERO) | (proposed_ops == OP_COORD_ONE)
    coord_leaf_nonzero = jnp.any(
        coord_leaf_mask & ((proposed_a1 != 0) | (proposed_a2 != 0))
    )
    proposed_ops, proposed_a1, proposed_a2 = key_safe_normalize_fn(
        proposed_ops,
        proposed_a1,
        proposed_a2,
        guard_zero_args_fn=guard_zero_args_fn,
    )
    is_null = proposed_ops == OP_NULL
    bad_key, _ = checked_pack_key_fn(
        proposed_ops, proposed_a1, proposed_a2, ledger.count
    )
    bounds_corrupt = bad_key | coord_leaf_nonzero

    def _corrupt_return(_):
        # Bounds violations are semantic CORRUPT; return zero ids (flag only).
        zero_ids = jnp.zeros_like(proposed_ops)
        new_ledger = ledger._replace(corrupt=jnp.array(True, dtype=jnp.bool_))
        return zero_ids, new_ledger

    def _do(_):
        return intern_core_fn(
            ledger,
            proposed_ops,
            proposed_a1,
            proposed_a2,
            cfg=cfg,
            key_safe_normalize_fn=key_safe_normalize_fn,
            pack_key_fn=pack_key_fn,
            candidate_indices_fn=candidate_indices_fn,
            guard_max_fn=guard_max_fn,
            guards_enabled_fn=guards_enabled_fn,
            coord_norm_id_jax_fn=coord_norm_id_jax_fn,
            coord_norm_id_jax_noprobe_fn=coord_norm_id_jax_noprobe_fn,
            coord_norm_probe_enabled_fn=coord_norm_probe_enabled_fn,
            coord_norm_probe_reset_cb_fn=coord_norm_probe_reset_cb_fn,
            coord_norm_probe_assert_fn=coord_norm_probe_assert_fn,
            scatter_drop_fn=scatter_drop_fn,
        )

    ids, new_ledger = lax.cond(bounds_corrupt, _corrupt_return, _do, operand=None)
    ids = jnp.where(is_null, jnp.int32(0), ids)
    return ids, new_ledger


def intern_nodes(
    ledger,
    batch_or_ops,
    a1=None,
    a2=None,
    *,
    cfg: InternConfig = DEFAULT_INTERN_CONFIG,
    intern_impl_fn=_intern_nodes_impl,
    lookup_node_id_fn=_lookup_node_id,
    key_safe_normalize_fn=_key_safe_normalize_nodes,
    guard_zero_args_fn=_guard_zero_args,
    checked_pack_key_fn=_checked_pack_key,
    guard_zero_row_fn=_guard_zero_row,
    pack_key_fn=_pack_key,
    candidate_indices_fn=_candidate_indices,
    guard_max_fn=_guard_max,
    guards_enabled_fn=_guards_enabled,
    coord_norm_id_jax_fn=_coord_norm_id_jax,
    coord_norm_id_jax_noprobe_fn=_coord_norm_id_jax_noprobe,
    coord_norm_probe_enabled_fn=_coord_norm_probe_enabled,
    coord_norm_probe_reset_cb_fn=_coord_norm_probe_reset_cb,
    coord_norm_probe_assert_fn=_coord_norm_probe_assert,
    scatter_drop_fn=_scatter_drop,
):
    """
    Batch-intern a list of proposed (op,a1,a2) nodes into the canonical Ledger.
    Canonical identity is via full key-byte equality (Canonicalᵢ).

    Args:
      ledger: Ledger
      batch_or_ops: NodeBatch or ops array
      a1/a2: optional arg arrays when passing raw ops

    Returns:
      final_ids: int32 array, shape [N], canonical ids for each proposal
      new_ledger: Ledger, updated
    """
    if a1 is None and a2 is None:
        if not isinstance(batch_or_ops, NodeBatch):
            raise TypeError("intern_nodes expects a NodeBatch or (ops, a1, a2)")
        batch = batch_or_ops
    else:
        if a1 is None or a2 is None:
            raise TypeError("intern_nodes expects both a1 and a2 arrays")
        batch = NodeBatch(batch_or_ops, a1, a2)
    proposed_ops, proposed_a1, proposed_a2 = batch
    if proposed_ops.shape[0] == 0:
        return jnp.zeros_like(proposed_ops), ledger
    stop = ledger.oom | ledger.corrupt
    # NOTE: stop path returns zeros today; read-only lookup fallback is deferred.

    # Once invalid, interning must not allocate or mutate state (m1 guardrail).
    # Stop path performs read-only lookup for existing keys (m1).
    # See IMPLEMENTATION_PLAN.md (m1 guardrails).
    def _lookup_existing(_):
        ops, a1, a2 = batch
        ops, a1, a2 = key_safe_normalize_fn(
            ops,
            a1,
            a2,
            guard_zero_args_fn=guard_zero_args_fn,
        )
        bad_key, _ = checked_pack_key_fn(ops, a1, a2, ledger.count)
        ops = jnp.where(bad_key, jnp.int32(0), ops)
        a1 = jnp.where(bad_key, jnp.int32(0), a1)
        a2 = jnp.where(bad_key, jnp.int32(0), a2)

        def _lookup_one(op, a1_val, a2_val):
            return lookup_node_id_fn(ledger, op, a1_val, a2_val)

        ids, found = jax.vmap(_lookup_one)(ops, a1, a2)
        ids = jnp.where(found & (~bad_key), ids, jnp.int32(0))
        ids = jnp.where(ops == OP_NULL, jnp.int32(0), ids)
        return ids, ledger

    def _do(_):
        return intern_impl_fn(
            ledger,
            batch,
            cfg=cfg,
            key_safe_normalize_fn=key_safe_normalize_fn,
            guard_zero_args_fn=guard_zero_args_fn,
            checked_pack_key_fn=checked_pack_key_fn,
            pack_key_fn=pack_key_fn,
            candidate_indices_fn=candidate_indices_fn,
            guard_max_fn=guard_max_fn,
            guards_enabled_fn=guards_enabled_fn,
            coord_norm_id_jax_fn=coord_norm_id_jax_fn,
            coord_norm_id_jax_noprobe_fn=coord_norm_id_jax_noprobe_fn,
            coord_norm_probe_enabled_fn=coord_norm_probe_enabled_fn,
            coord_norm_probe_reset_cb_fn=coord_norm_probe_reset_cb_fn,
            coord_norm_probe_assert_fn=coord_norm_probe_assert_fn,
            scatter_drop_fn=scatter_drop_fn,
        )

    ids, new_ledger = lax.cond(stop, _lookup_existing, _do, operand=None)
    guard_zero_row_fn(
        new_ledger.opcode, new_ledger.arg1, new_ledger.arg2, "intern_nodes.row1"
    )
    return ids, new_ledger


__all__ = [
    "_coord_norm_id_jax",
    "_coord_norm_id_jax_noprobe",
    "_lookup_node_id",
    "KeyColumns",
    "InternConfig",
    "DEFAULT_INTERN_CONFIG",
    "intern_nodes",
]
