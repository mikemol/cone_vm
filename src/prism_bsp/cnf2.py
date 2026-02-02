from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from prism_core import jax_safe as _jax_safe
from prism_core.di import call_with_optional_kwargs
from prism_core.guards import (
    GuardConfig,
    resolve_safe_gather_ok_value_fn,
)
from prism_core.compact import scatter_compacted_ids
from prism_core.safety import (
    PolicyMode,
    DEFAULT_SAFETY_POLICY,
    POLICY_VALUE_DEFAULT,
    PolicyValue,
    SafetyPolicy,
    require_static_policy,
    require_value_policy,
)
from prism_core.errors import PrismPolicyBindingError
from prism_core.modes import ValidateMode
from prism_coord.coord import coord_xor_batch
from prism_ledger.intern import intern_nodes, intern_nodes_state
from prism_ledger.config import InternConfig, DEFAULT_INTERN_CONFIG
from prism_ledger.index import LedgerState, derive_ledger_state
from prism_bsp.config import (
    Cnf2Config,
    Cnf2BoundConfig,
    Cnf2StaticBoundConfig,
    Cnf2ValueBoundConfig,
    Cnf2CommitInputs,
    Cnf2InternInputs,
    Cnf2RuntimeFns,
    DEFAULT_CNF2_RUNTIME_FNS,
    resolve_cnf2_inputs,
    resolve_cnf2_candidate_inputs,
    resolve_cnf2_intern_inputs,
    resolve_validate_mode,
    make_cnf2_post_q_handler_static,
    make_cnf2_post_q_handler_value,
)
from prism_semantics.commit import (
    _identity_q,
    apply_q,
    apply_q_ok,
    commit_stratum,
    commit_stratum_bound,
    commit_stratum_static,
    commit_stratum_value,
)
from prism_vm_core.candidates import _candidate_indices
from prism_vm_core.domains import (
    _committed_ids,
    _host_bool_value,
    _host_int_value,
    _provisional_ids,
)
from prism_vm_core.guards import _guards_enabled
from prism_vm_core.hashes import _ledger_roots_hash_host
from prism_vm_core.ontology import (
    OP_ADD,
    OP_COORD_ONE,
    OP_COORD_PAIR,
    OP_COORD_ZERO,
    OP_MUL,
    OP_SUC,
    OP_ZERO,
    ZERO_PTR,
)
from prism_vm_core.structures import CandidateBuffer, Ledger, Stratum, NodeBatch
from prism_vm_core.protocols import (
    ApplyQFn,
    CandidateIndicesFn,
    CommitStratumFn,
    CoordXorBatchFn,
    EmitCandidatesFn,
    GuardsEnabledFn,
    HostBoolValueFn,
    HostIntValueFn,
    IdentityQFn,
    InternFn,
    InternStateFn,
    LedgerRootsHashFn,
    NodeBatchFn,
    ScatterDropFn,
    SafeGatherOkFn,
    SafeGatherOkValueFn,
)

EMPTY_COMMIT_OPTIONAL: dict = {}


# dataflow-bundle: commit_stratum_fn, intern_fn
# CNF-2 commit/intern pair forwarded through DI resolution.
# dataflow-bundle: _frontier, _ledger, _post_ids
# root-assertion guard hook bundle (debug-only)
# dataflow-bundle: next_frontier, post_ids
# root-assertion bundle (debug-only)
@dataclass(frozen=True)
class _RootAssertNoopArgs:
    ledger: object
    frontier: object
    post_ids: object


@dataclass(frozen=True)
class _RootAssertArgs:
    next_frontier: object
    post_ids: object


def _assert_roots_noop(_ledger, _frontier, _post_ids):
    _ = _RootAssertNoopArgs(_ledger, _frontier, _post_ids)
    return None


_TEST_GUARDS = _jax_safe.TEST_GUARDS
_scatter_drop = _jax_safe.scatter_drop


def _node_batch(op, a1, a2) -> NodeBatch:
    return NodeBatch(op=op, a1=a1, a2=a2)


def _resolve_intern_cfg(
    cfg: Cnf2Config | None, intern_cfg: InternConfig | None
) -> InternConfig:
    if intern_cfg is None and cfg is not None and cfg.intern_cfg is not None:
        intern_cfg = cfg.intern_cfg
    if intern_cfg is None:
        intern_cfg = DEFAULT_INTERN_CONFIG
    return intern_cfg


def _state_with_ledger(state: LedgerState, ledger) -> LedgerState:
    if ledger is state.ledger:
        return state
    return LedgerState(
        ledger=ledger,
        index=state.index,
        op_buckets_full_range=state.op_buckets_full_range,
    )


def _coord_xor_batch_state(
    state: LedgerState,
    left_ids,
    right_ids,
    *,
    coord_xor_batch_fn: CoordXorBatchFn,
    intern_cfg: InternConfig,
) -> tuple[jnp.ndarray, LedgerState]:
    ids, ledger = coord_xor_batch_fn(state.ledger, left_ids, right_ids)
    if ledger is state.ledger:
        return ids, state
    return ids, derive_ledger_state(
        ledger, op_buckets_full_range=intern_cfg.op_buckets_full_range
    )


def _commit_stratum_state(
    state: LedgerState,
    stratum: Stratum,
    *,
    commit_fns: Cnf2CommitInputs,
    commit_optional: dict,
    intern_cfg: InternConfig,
    **kwargs,
):
    commit_stratum_fn = commit_fns.commit_stratum_fn
    intern_fn = commit_fns.intern_fn

    def _intern_with_index(ledger, batch_or_ops, a1=None, a2=None):
        return call_with_optional_kwargs(
            intern_fn,
            {"ledger_index": state.index},
            ledger,
            batch_or_ops,
            a1,
            a2,
        )

    ledger, canon_ids, q_map = call_with_optional_kwargs(
        commit_stratum_fn,
        commit_optional,
        state.ledger,
        stratum,
        intern_fn=_intern_with_index,
        **kwargs,
    )
    if ledger is state.ledger:
        return state, canon_ids, q_map
    new_state = derive_ledger_state(
        ledger, op_buckets_full_range=intern_cfg.op_buckets_full_range
    )
    return new_state, canon_ids, q_map


def emit_candidates(ledger, frontier_ids):
    num_frontier = frontier_ids.shape[0]
    size = num_frontier * 2
    enabled = jnp.zeros(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = jnp.zeros(size, dtype=jnp.int32)
    arg2 = jnp.zeros(size, dtype=jnp.int32)

    f_ops = ledger.opcode[frontier_ids]
    f_a1 = ledger.arg1[frontier_ids]
    f_a2 = ledger.arg2[frontier_ids]
    op_a1 = ledger.opcode[f_a1]
    op_a2 = ledger.opcode[f_a2]

    is_add = f_ops == OP_ADD
    is_mul = f_ops == OP_MUL
    is_zero_a1 = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    is_suc_a1 = op_a1 == OP_SUC
    is_suc_a2 = op_a2 == OP_SUC

    is_add_zero = is_add & (is_zero_a1 | is_zero_a2)
    is_mul_zero = is_mul & (is_zero_a1 | is_zero_a2)
    is_add_suc = is_add & (is_suc_a1 | is_suc_a2) & (~is_add_zero)
    is_mul_suc = is_mul & (is_suc_a1 | is_suc_a2) & (~is_mul_zero)
    enable0 = is_add_zero | is_mul_zero | is_add_suc | is_mul_suc
    zero_on_a1 = is_zero_a1
    zero_on_a2 = (~is_zero_a1) & is_zero_a2
    zero_other = jnp.where(zero_on_a1, f_a2, f_a1)
    y_id = jnp.where(is_add_zero, zero_other, jnp.int32(ZERO_PTR))

    suc_on_a1 = is_suc_a1
    suc_on_a2 = (~is_suc_a1) & is_suc_a2
    suc_node = jnp.where(suc_on_a1, f_a1, f_a2)
    other_node = jnp.where(suc_on_a1, f_a2, f_a1)
    val_x = ledger.arg1[suc_node]
    val_y = other_node

    cand0_op = jnp.zeros_like(f_ops)
    cand0_a1 = jnp.zeros_like(f_a1)
    cand0_a2 = jnp.zeros_like(f_a2)

    id_mask = is_add_zero | is_mul_zero
    cand0_op = jnp.where(id_mask, ledger.opcode[y_id], cand0_op)
    cand0_a1 = jnp.where(id_mask, ledger.arg1[y_id], cand0_a1)
    cand0_a2 = jnp.where(id_mask, ledger.arg2[y_id], cand0_a2)

    cand0_op = jnp.where(is_add_suc, jnp.int32(OP_ADD), cand0_op)
    cand0_a1 = jnp.where(is_add_suc, val_x, cand0_a1)
    cand0_a2 = jnp.where(is_add_suc, val_y, cand0_a2)

    cand0_op = jnp.where(is_mul_suc, jnp.int32(OP_MUL), cand0_op)
    cand0_a1 = jnp.where(is_mul_suc, val_x, cand0_a1)
    cand0_a2 = jnp.where(is_mul_suc, val_y, cand0_a2)

    cand0_op = jnp.where(enable0, cand0_op, jnp.int32(0))
    cand0_a1 = jnp.where(enable0, cand0_a1, jnp.int32(0))
    cand0_a2 = jnp.where(enable0, cand0_a2, jnp.int32(0))

    idx0 = jnp.arange(num_frontier, dtype=jnp.int32) * 2
    enabled = enabled.at[idx0].set(enable0.astype(jnp.int32))
    opcode = opcode.at[idx0].set(cand0_op)
    arg1 = arg1.at[idx0].set(cand0_a1)
    arg2 = arg2.at[idx0].set(cand0_a2)

    return CandidateBuffer(enabled=enabled, opcode=opcode, arg1=arg1, arg2=arg2)


def emit_candidates_cfg(
    ledger,
    frontier_ids,
    *,
    cfg: Cnf2Config | None = None,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
):
    """Interface/Control wrapper for emit_candidates with DI bundle."""
    resolved = resolve_cnf2_candidate_inputs(
        cfg,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=_candidate_indices,
        candidate_indices_default=_candidate_indices,
        scatter_drop_fn=_scatter_drop,
    )
    return resolved.emit_candidates_fn(ledger, frontier_ids)


def compact_candidates_result(
    candidates, *, candidate_indices_fn=_candidate_indices
):
    enabled = candidates.enabled.astype(jnp.int32)
    result = candidate_indices_fn(enabled)
    idx = result.idx
    valid = result.valid
    safe_idx = jnp.where(valid, idx, 0)

    compacted = CandidateBuffer(
        enabled=valid.astype(jnp.int32),
        opcode=candidates.opcode[safe_idx],
        arg1=candidates.arg1[safe_idx],
        arg2=candidates.arg2[safe_idx],
    )
    return compacted, result


def compact_candidates(
    candidates, *, candidate_indices_fn=_candidate_indices
):
    compacted, result = compact_candidates_result(
        candidates, candidate_indices_fn=candidate_indices_fn
    )
    return compacted, result.count


def compact_candidates_cfg(
    candidates, *, cfg: Cnf2Config | None = None
):
    """Interface/Control wrapper for compact_candidates with DI bundle."""
    resolved = resolve_cnf2_candidate_inputs(
        cfg,
        emit_candidates_fn=emit_candidates,
        candidate_indices_fn=_candidate_indices,
        candidate_indices_default=_candidate_indices,
        scatter_drop_fn=_scatter_drop,
    )
    return compact_candidates(candidates, candidate_indices_fn=resolved.candidate_indices_fn)


def compact_candidates_with_index_result(
    candidates, *, candidate_indices_fn=_candidate_indices
):
    compacted, result = compact_candidates_result(
        candidates, candidate_indices_fn=candidate_indices_fn
    )
    return compacted, result, result.idx


def compact_candidates_with_index(
    candidates, *, candidate_indices_fn=_candidate_indices
):
    compacted, result, idx = compact_candidates_with_index_result(
        candidates, candidate_indices_fn=candidate_indices_fn
    )
    return compacted, result.count, idx


def compact_candidates_with_index_cfg(
    candidates, *, cfg: Cnf2Config | None = None
):
    """Interface/Control wrapper for compact_candidates_with_index with DI bundle."""
    resolved = resolve_cnf2_candidate_inputs(
        cfg,
        emit_candidates_fn=emit_candidates,
        candidate_indices_fn=_candidate_indices,
        candidate_indices_default=_candidate_indices,
        scatter_drop_fn=_scatter_drop,
    )
    return compact_candidates_with_index(
        candidates, candidate_indices_fn=resolved.candidate_indices_fn
    )


def _scatter_compacted_ids(
    comp_idx,
    ids_compact,
    count,
    size,
    *,
    scatter_drop_fn=_scatter_drop,
):
    return scatter_compacted_ids(
        comp_idx,
        ids_compact,
        count,
        size,
        scatter_drop_fn=scatter_drop_fn,
        index_dtype=jnp.int32,
    )


def scatter_compacted_ids_cfg(
    comp_idx,
    ids_compact,
    count,
    size,
    *,
    cfg: Cnf2Config | None = None,
):
    """Interface/Control wrapper for _scatter_compacted_ids with DI bundle."""
    resolved = resolve_cnf2_candidate_inputs(
        cfg,
        emit_candidates_fn=emit_candidates,
        candidate_indices_fn=_candidate_indices,
        candidate_indices_default=_candidate_indices,
        scatter_drop_fn=_scatter_drop,
    )
    return _scatter_compacted_ids(
        comp_idx,
        ids_compact,
        count,
        size,
        scatter_drop_fn=resolved.scatter_drop_fn,
    )


def _ledger_index_is_bound(fn: Callable[..., object]) -> bool:
    if getattr(fn, "_prism_ledger_index_bound", False):
        return True
    if isinstance(fn, partial):
        keywords = fn.keywords or {}
        return "ledger_index" in keywords
    return False


def _bind_intern_with_index(
    ledger: Ledger,
    intern_inputs: Cnf2InternInputs,
    *,
    intern_cfg: InternConfig | None,
) -> Cnf2InternInputs:
    if _ledger_index_is_bound(intern_inputs.intern_fn):
        return intern_inputs
    cfg = intern_cfg or DEFAULT_INTERN_CONFIG
    ledger_index = derive_ledger_state(
        ledger, op_buckets_full_range=cfg.op_buckets_full_range
    ).index

    def _intern_with_index(ledger, batch_or_ops, a1=None, a2=None):
        return call_with_optional_kwargs(
            intern_inputs.intern_fn,
            {"ledger_index": ledger_index},
            ledger,
            batch_or_ops,
            a1,
            a2,
        )

    setattr(_intern_with_index, "_prism_ledger_index_bound", True)
    return replace(intern_inputs, intern_fn=_intern_with_index)


def _intern_candidates_core(
    ledger,
    candidates,
    *,
    intern_inputs: Cnf2InternInputs,
):
    compacted, count = intern_inputs.compact_candidates_fn(candidates)
    enabled = compacted.enabled.astype(jnp.int32)
    ops = jnp.where(enabled, compacted.opcode, jnp.int32(0))
    a1 = jnp.where(enabled, compacted.arg1, jnp.int32(0))
    a2 = jnp.where(enabled, compacted.arg2, jnp.int32(0))
    ids, new_ledger = intern_inputs.intern_fn(
        ledger, intern_inputs.node_batch_fn(ops, a1, a2)
    )
    return ids, new_ledger, count


def intern_candidates(
    ledger: Ledger,
    candidates,
    *,
    compact_candidates_fn: Callable[..., tuple] = compact_candidates,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
):
    resolved = resolve_cnf2_intern_inputs(
        None,
        compact_candidates_fn=compact_candidates_fn,
        compact_candidates_default=compact_candidates,
        candidate_indices_default=_candidate_indices,
        intern_fn=intern_fn,
        intern_fn_default=intern_nodes,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        node_batch_default=_node_batch,
    )
    resolved = _bind_intern_with_index(
        ledger,
        resolved,
        intern_cfg=intern_cfg,
    )
    return _intern_candidates_core(
        ledger,
        candidates,
        intern_inputs=resolved,
    )


def intern_candidates_cfg(
    ledger: Ledger,
    candidates,
    *,
    cfg: Cnf2Config | None = None,
    compact_candidates_fn: Callable[..., tuple] = compact_candidates,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
):
    """Interface/Control wrapper for intern_candidates with DI bundle."""
    if cfg is not None and intern_cfg is None:
        intern_cfg = cfg.intern_cfg
    resolved = resolve_cnf2_intern_inputs(
        cfg,
        compact_candidates_fn=compact_candidates_fn,
        compact_candidates_default=compact_candidates,
        candidate_indices_default=_candidate_indices,
        intern_fn=intern_fn,
        intern_fn_default=intern_nodes,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        node_batch_default=_node_batch,
    )
    resolved = _bind_intern_with_index(
        ledger,
        resolved,
        intern_cfg=intern_cfg,
    )
    return _intern_candidates_core(
        ledger,
        candidates,
        intern_inputs=resolved,
    )

def _cycle_candidates_core_impl_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    commit_optional: dict = EMPTY_COMMIT_OPTIONAL,
    post_q_handler,
    guard_cfg: GuardConfig | None = None,
    commit_fns: Cnf2CommitInputs,
    intern_state_fn: InternStateFn = intern_nodes_state,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    assert_roots_fn=_assert_roots_noop,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    resolved = resolve_cnf2_inputs(
        cfg,
        guard_cfg=guard_cfg,
        intern_cfg=intern_cfg,
        commit_fns=commit_fns,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=safe_gather_ok_fn,
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=_guards_enabled,
        ledger_roots_hash_host_fn=_ledger_roots_hash_host,
        runtime_fns=runtime_fns,
    )
    guard_cfg = resolved.guard_cfg
    intern_cfg = resolved.intern_cfg
    commit_fns = resolved.commit_fns
    safe_gather_ok_fn = resolved.safe_gather_ok_fn
    safe_gather_ok_value_fn = resolved.safe_gather_ok_value_fn
    node_batch_fn = resolved.node_batch_fn
    coord_xor_batch_fn = resolved.coord_xor_batch_fn
    emit_candidates_fn = resolved.emit_candidates_fn
    candidate_indices_fn = resolved.candidate_indices_fn
    scatter_drop_fn = resolved.scatter_drop_fn
    apply_q_fn = resolved.apply_q_fn
    identity_q_fn = resolved.identity_q_fn
    host_bool_value_fn = resolved.host_bool_value_fn
    host_int_value_fn = resolved.host_int_value_fn
    runtime_fns = resolved.runtime_fns
    if commit_optional:
        commit_optional = dict(commit_optional)
        if "safe_gather_ok_fn" in commit_optional:
            commit_optional["safe_gather_ok_fn"] = safe_gather_ok_fn
        if "safe_gather_ok_value_fn" in commit_optional:
            commit_optional["safe_gather_ok_value_fn"] = safe_gather_ok_value_fn
        if "guard_cfg" in commit_optional:
            commit_optional["guard_cfg"] = guard_cfg
    intern_cfg = _resolve_intern_cfg(cfg, intern_cfg)
    if commit_fns.intern_fn is intern_nodes:
        commit_fns = Cnf2CommitInputs(
            intern_fn=partial(intern_nodes, cfg=intern_cfg),
            commit_stratum_fn=commit_fns.commit_stratum_fn,
        )
    if intern_state_fn is intern_nodes_state:
        def _intern_state_from_commit_fn(state_in, batch_or_ops, a1=None, a2=None, *, cfg=None):
            optional = {"cfg": cfg, "ledger_index": state_in.index}
            ids, new_ledger = call_with_optional_kwargs(
                commit_fns.intern_fn,
                optional,
                state_in.ledger,
                batch_or_ops,
                a1,
                a2,
            )
            if new_ledger is state_in.ledger:
                return ids, state_in
            op_buckets_full_range = (
                cfg.op_buckets_full_range
                if cfg is not None
                else DEFAULT_INTERN_CONFIG.op_buckets_full_range
            )
            new_state = derive_ledger_state(
                new_ledger, op_buckets_full_range=op_buckets_full_range
            )
            return ids, new_state

        intern_state_fn = _intern_state_from_commit_fn
    cnf2_metrics_update_fn = runtime_fns.cnf2_metrics_update_fn
    ledger = state.ledger
    # BSPᵗ: temporal superstep / barrier semantics.
    frontier_ids = _committed_ids(frontier_ids)
    frontier_arr = jnp.atleast_1d(frontier_ids.a)
    frontier_ids = _committed_ids(frontier_arr)
    num_frontier = frontier_arr.shape[0]

    def _peel_one(ptr):
        def cond(state):
            curr, _ = state
            return ledger.opcode[curr] == OP_SUC

        def body(state):
            curr, depth = state
            return ledger.arg1[curr], depth + 1

        return lax.while_loop(cond, body, (ptr, jnp.int32(0)))

    rewrite_ids, depths = jax.vmap(_peel_one)(frontier_arr)

    r_ops = ledger.opcode[rewrite_ids]
    r_a1 = ledger.arg1[rewrite_ids]
    r_a2 = ledger.arg2[rewrite_ids]
    op_a1 = ledger.opcode[r_a1]
    op_a2 = ledger.opcode[r_a2]
    is_coord_a1 = (
        (op_a1 == OP_COORD_ZERO)
        | (op_a1 == OP_COORD_ONE)
        | (op_a1 == OP_COORD_PAIR)
    )
    is_coord_a2 = (
        (op_a2 == OP_COORD_ZERO)
        | (op_a2 == OP_COORD_ONE)
        | (op_a2 == OP_COORD_PAIR)
    )
    is_coord_add = (r_ops == OP_ADD) & is_coord_a1 & is_coord_a2

    # Coordinate aggregation (Aggregate𝚌) runs before stratum0 to preserve
    # strict strata while canonicalizing coord-add payloads.
    coord_ids = jnp.zeros_like(rewrite_ids)
    coord_enabled = is_coord_add.astype(jnp.int32)
    coord_result = candidate_indices_fn(coord_enabled)
    coord_idx = coord_result.idx
    coord_valid = coord_result.valid
    coord_count = coord_result.count
    coord_count_i = host_int_value_fn(coord_count)
    coord_idx_safe = jnp.where(coord_valid, coord_idx, 0)
    coord_left = r_a1[coord_idx_safe][:coord_count_i]
    coord_right = r_a2[coord_idx_safe][:coord_count_i]
    coord_ids_compact, state = _coord_xor_batch_state(
        state,
        coord_left,
        coord_right,
        coord_xor_batch_fn=coord_xor_batch_fn,
        intern_cfg=intern_cfg,
    )
    ledger = state.ledger
    coord_ids_full = jnp.zeros_like(coord_idx_safe)
    coord_ids_full = coord_ids_full.at[:coord_count_i].set(coord_ids_compact)
    coord_ids = _scatter_compacted_ids(
        coord_idx,
        coord_ids_full,
        coord_count,
        num_frontier,
        scatter_drop_fn=scatter_drop_fn,
    )

    start0 = ledger.count.astype(jnp.int32)
    candidates = emit_candidates_fn(ledger, rewrite_ids)
    compacted0, count0, comp_idx0 = compact_candidates_with_index(
        candidates, candidate_indices_fn=candidate_indices_fn
    )
    enabled0 = compacted0.enabled.astype(jnp.int32)
    ops0 = jnp.where(enabled0, compacted0.opcode, jnp.int32(0))
    a1_0 = jnp.where(enabled0, compacted0.arg1, jnp.int32(0))
    a2_0 = jnp.where(enabled0, compacted0.arg2, jnp.int32(0))
    ids_compact, state0 = call_with_optional_kwargs(
        intern_state_fn,
        {"cfg": intern_cfg},
        state,
        node_batch_fn(ops0, a1_0, a2_0),
    )
    ledger0 = state0.ledger
    size0 = candidates.enabled.shape[0]
    ids_full0 = _scatter_compacted_ids(
        comp_idx0,
        ids_compact,
        count0,
        size0,
        scatter_drop_fn=scatter_drop_fn,
    )
    # Candidate buffer layout invariant: slot0 at 2*i, slot1 at 2*i+1.
    # cycle_candidates relies on this; see IMPLEMENTATION_PLAN.md.
    idx0 = jnp.arange(num_frontier, dtype=jnp.int32) * 2
    slot0_ids = ids_full0[idx0]
    slot0_ids = jnp.where(is_coord_add, coord_ids, slot0_ids)
    is_add = r_ops == OP_ADD
    is_mul = r_ops == OP_MUL
    is_suc_a1 = op_a1 == OP_SUC
    is_suc_a2 = op_a2 == OP_SUC
    is_add_suc = is_add & (is_suc_a1 | is_suc_a2)
    is_mul_suc = is_mul & (is_suc_a1 | is_suc_a2)
    is_zero_a1 = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    is_add_zero = is_add & (is_zero_a1 | is_zero_a2)
    is_mul_zero = is_mul & (is_zero_a1 | is_zero_a2)
    is_add_suc = is_add_suc & (~is_add_zero)
    is_mul_suc = is_mul_suc & (~is_mul_zero)
    suc_on_a1 = is_suc_a1
    suc_on_a2 = (~is_suc_a1) & is_suc_a2
    suc_node = jnp.where(suc_on_a1, r_a1, r_a2)
    val_x = ledger.arg1[suc_node]
    val_y = jnp.where(suc_on_a1, r_a2, r_a1)

    # Slot1 is the continuation slot in CNF-2; enabled starting in m2. Under
    # test guards (m3 normative), hyperstrata visibility is enforced so slot1
    # reads only from slot0 + pre-step.
    # See IMPLEMENTATION_PLAN.md (CNF-2 continuation slot).
    # M2 commit: slot1 is always enabled; no runtime gate.
    slot1_add = is_add_suc
    slot1_mul = is_mul_suc
    slot1_enabled = slot1_add | slot1_mul
    slot1_ops = jnp.zeros_like(r_ops)
    slot1_a1 = jnp.zeros_like(r_a1)
    slot1_a2 = jnp.zeros_like(r_a2)
    slot1_ops = jnp.where(slot1_add, jnp.int32(OP_SUC), slot1_ops)
    slot1_a1 = jnp.where(slot1_add, slot0_ids, slot1_a1)
    slot1_ops = jnp.where(slot1_mul, jnp.int32(OP_ADD), slot1_ops)
    slot1_a1 = jnp.where(slot1_mul, val_y, slot1_a1)
    slot1_a2 = jnp.where(slot1_mul, slot0_ids, slot1_a2)

    slot1_ops = jnp.where(slot1_enabled, slot1_ops, jnp.int32(0))
    slot1_a1 = jnp.where(slot1_enabled, slot1_a1, jnp.int32(0))
    slot1_a2 = jnp.where(slot1_enabled, slot1_a2, jnp.int32(0))
    slot1_ids, state1 = call_with_optional_kwargs(
        intern_state_fn,
        {"cfg": intern_cfg},
        state0,
        node_batch_fn(slot1_ops, slot1_a1, slot1_a2),
    )
    ledger1 = state1.ledger
    zero_on_a1 = is_zero_a1
    zero_on_a2 = (~is_zero_a1) & is_zero_a2
    zero_other = jnp.where(zero_on_a1, r_a2, r_a1)
    base_next = rewrite_ids
    base_next = jnp.where(is_add_zero, zero_other, base_next)
    base_next = jnp.where(is_mul_zero, jnp.int32(ZERO_PTR), base_next)
    base_next = jnp.where(slot1_add, slot1_ids, base_next)
    base_next = jnp.where(slot1_mul, slot1_ids, base_next)
    base_next = jnp.where(is_coord_add, coord_ids, base_next)
    changed_mask = base_next != rewrite_ids

    wrap_strata = [(host_int_value_fn(ledger1.count), 0)]
    wrap_depths = depths
    next_frontier = base_next
    state2 = state1
    ledger2 = state2.ledger
    while host_bool_value_fn(jnp.any((wrap_depths > 0) & (~ledger2.oom))):
        to_wrap = (wrap_depths > 0) & (~ledger2.oom)
        ops = jnp.where(to_wrap, jnp.int32(OP_SUC), jnp.int32(0))
        a1 = jnp.where(to_wrap, next_frontier, jnp.int32(0))
        a2 = jnp.zeros_like(a1)
        start = host_int_value_fn(ledger2.count)
        new_ids, state2 = call_with_optional_kwargs(
            intern_state_fn,
            {"cfg": intern_cfg},
            state2,
            node_batch_fn(ops, a1, a2),
        )
        ledger2 = state2.ledger
        end = host_int_value_fn(ledger2.count)
        wrap_strata.append((start, end - start))
        next_frontier = jnp.where(to_wrap, new_ids, next_frontier)
        wrap_depths = wrap_depths - to_wrap.astype(jnp.int32)

    # Strata counts track appended id ranges (ledger.count deltas), not
    # proposal counts; keep validators/q-map aligned (see IMPLEMENTATION_PLAN.md).
    stratum0 = Stratum(
        start=start0, count=(ledger0.count - start0).astype(jnp.int32)
    )
    start1 = ledger0.count.astype(jnp.int32)
    stratum1 = Stratum(
        start=start1, count=(ledger1.count - start1).astype(jnp.int32)
    )
    start2_i = wrap_strata[0][0]
    count2_i = sum(count for _, count in wrap_strata)
    stratum2 = Stratum(
        start=jnp.int32(start2_i), count=jnp.int32(count2_i)
    )
    rewrite_child = host_int_value_fn(count0)
    changed_count = host_int_value_fn(jnp.sum(changed_mask.astype(jnp.int32)))
    cnf2_metrics_update_fn(rewrite_child, changed_count, int(count2_i))
    state2, _, q_map = _commit_stratum_state(
        state2,
        stratum0,
        commit_fns=commit_fns,
        commit_optional=commit_optional,
        intern_cfg=intern_cfg,
        validate_mode=validate_mode,
    )
    state2, _, q_map = _commit_stratum_state(
        state2,
        stratum1,
        commit_fns=commit_fns,
        commit_optional=commit_optional,
        intern_cfg=intern_cfg,
        prior_q=q_map,
        validate_mode=validate_mode,
    )
    # Wrapper strata are micro-strata in s=2; commit in order for hyperstrata visibility.
    for start_i, count_i in wrap_strata:
        micro_stratum = Stratum(
            start=jnp.int32(start_i), count=jnp.int32(count_i)
        )
        state2, _, q_map = _commit_stratum_state(
            state2,
            micro_stratum,
            commit_fns=commit_fns,
            commit_optional=commit_optional,
            intern_cfg=intern_cfg,
            prior_q=q_map,
            validate_mode=validate_mode,
        )
    next_frontier = _provisional_ids(next_frontier)
    ledger2, post_ids = post_q_handler(state2.ledger, q_map, next_frontier)
    state2 = _state_with_ledger(state2, ledger2)
    assert_roots_fn(state2.ledger, next_frontier, post_ids)
    return state2, next_frontier, (stratum0, stratum1, stratum2), q_map


def _cycle_candidates_core_static_bound(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy: SafetyPolicy,
    guard_cfg: GuardConfig | None = None,
    commit_fns: Cnf2CommitInputs,
    intern_state_fn: InternStateFn = intern_nodes_state,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """CNF-2 core (static policy) with policy decisions pre-bound at the edge.

    This entrypoint assumes:
    - safe_gather_policy is already resolved (non-optional),
    - safe_gather_ok_fn is already bound if policy-bound behavior is required,
    - commit_fns.commit_stratum_fn is already the correct bound/static variant.

    All policy binding / guard binding / config resolution must happen
    outside this function so the core remains branch-free with respect to
    policy composition. Only algorithmic guards remain below.
    """
    mode = resolve_validate_mode(
        validate_mode, guards_enabled_fn=guards_enabled_fn
    )
    intern_cfg = _resolve_intern_cfg(cfg, intern_cfg)
    if isinstance(ledger, LedgerState):
        state = ledger
        ledger = state.ledger
    else:
        state = derive_ledger_state(
            ledger, op_buckets_full_range=intern_cfg.op_buckets_full_range
        )
    commit_optional = {
        "safe_gather_policy": safe_gather_policy,
        "safe_gather_ok_fn": safe_gather_ok_fn,
        "guard_cfg": guard_cfg,
    }

    frontier_ids = _committed_ids(frontier_ids)
    # --- Algorithmic guards (explicit, non-policy) ---
    # Corrupt ledger short-circuit: no rewrite on invalid state.
    # SYNC: host read to short-circuit on corrupt ledgers (m1).
    if host_bool_value_fn(ledger.corrupt):
        empty = Stratum(start=ledger.count.astype(jnp.int32), count=jnp.int32(0))
        return (
            state,
            _provisional_ids(frontier_ids.a),
            (empty, empty, empty),
            identity_q_fn,
        )
    frontier_arr = jnp.atleast_1d(frontier_ids.a)
    frontier_ids = _committed_ids(frontier_arr)
    # Empty frontier short-circuit: no rewrite or allocation.
    if frontier_arr.shape[0] == 0:
        empty = Stratum(start=ledger.count.astype(jnp.int32), count=jnp.int32(0))
        return (
            state,
            _provisional_ids(frontier_ids.a),
            (empty, empty, empty),
            identity_q_fn,
        )

    def _assert_roots(ledger2, next_frontier, post_ids):
        # Test guard: denotation invariance under q projection.
        if not _TEST_GUARDS:
            return
        args = _RootAssertArgs(next_frontier, post_ids)
        pre_hash = ledger_roots_hash_host_fn(ledger2, args.next_frontier.a)
        post_hash = ledger_roots_hash_host_fn(ledger2, args.post_ids.a)
        if pre_hash != post_hash:
            raise RuntimeError("BSPᵗ projection changed root structure")

    _post_q_handler = make_cnf2_post_q_handler_static(apply_q_fn)

    return _cycle_candidates_core_impl_state(
        state,
        frontier_ids,
        validate_mode=mode,
        cfg=cfg,
        commit_optional=commit_optional,
        post_q_handler=_post_q_handler,
        guard_cfg=guard_cfg,
        commit_fns=commit_fns,
        intern_state_fn=intern_state_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        assert_roots_fn=_assert_roots,
        safe_gather_ok_fn=safe_gather_ok_fn,
        safe_gather_ok_value_fn=_jax_safe.safe_gather_1d_ok_value,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        runtime_fns=runtime_fns,
    )


def _cycle_candidates_core_value_bound(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy_value: PolicyValue,
    guard_cfg: GuardConfig | None = None,
    commit_fns: Cnf2CommitInputs,
    intern_state_fn: InternStateFn = intern_nodes_state,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """CNF-2 core (policy as JAX value) with policy decisions pre-bound at the edge.

    This entrypoint assumes:
    - safe_gather_policy_value is already resolved (non-optional),
    - safe_gather_ok_value_fn is already guard-bound,
    - commit_fns.commit_stratum_fn is already the correct value-policy variant.

    All policy binding / guard binding / config resolution must happen
    outside this function so the core remains branch-free with respect to
    policy composition. Only algorithmic guards remain below.
    """
    mode = resolve_validate_mode(
        validate_mode, guards_enabled_fn=guards_enabled_fn
    )
    intern_cfg = _resolve_intern_cfg(cfg, intern_cfg)
    if isinstance(ledger, LedgerState):
        state = ledger
        ledger = state.ledger
    else:
        state = derive_ledger_state(
            ledger, op_buckets_full_range=intern_cfg.op_buckets_full_range
        )
    commit_optional = {
        "safe_gather_policy_value": safe_gather_policy_value,
        "safe_gather_ok_value_fn": safe_gather_ok_value_fn,
        "guard_cfg": guard_cfg,
    }
    frontier_ids = _committed_ids(frontier_ids)
    # --- Algorithmic guards (explicit, non-policy) ---
    # Corrupt ledger short-circuit: no rewrite on invalid state.
    # SYNC: host read to short-circuit on corrupt ledgers (m1).
    if host_bool_value_fn(ledger.corrupt):
        empty = Stratum(start=ledger.count.astype(jnp.int32), count=jnp.int32(0))
        return (
            state,
            _provisional_ids(frontier_ids.a),
            (empty, empty, empty),
            identity_q_fn,
        )
    frontier_arr = jnp.atleast_1d(frontier_ids.a)
    frontier_ids = _committed_ids(frontier_arr)
    # Empty frontier short-circuit: no rewrite or allocation.
    if frontier_arr.shape[0] == 0:
        empty = Stratum(start=ledger.count.astype(jnp.int32), count=jnp.int32(0))
        return (
            state,
            _provisional_ids(frontier_ids.a),
            (empty, empty, empty),
            identity_q_fn,
        )

    def _assert_roots(ledger2, next_frontier, post_ids):
        # Test guard: denotation invariance under q projection.
        if not _TEST_GUARDS:
            return
        args = _RootAssertArgs(next_frontier, post_ids)
        pre_hash = ledger_roots_hash_host_fn(ledger2, args.next_frontier.a)
        post_hash = ledger_roots_hash_host_fn(ledger2, args.post_ids.a)
        if pre_hash != post_hash:
            raise RuntimeError("BSPᵗ projection changed root structure")

    _post_q_handler = make_cnf2_post_q_handler_value(apply_q_fn)

    return _cycle_candidates_core_impl_state(
        state,
        frontier_ids,
        validate_mode=mode,
        cfg=cfg,
        commit_optional=commit_optional,
        post_q_handler=_post_q_handler,
        guard_cfg=guard_cfg,
        commit_fns=commit_fns,
        intern_state_fn=intern_state_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        assert_roots_fn=_assert_roots,
        safe_gather_ok_fn=_jax_safe.safe_gather_1d_ok,
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        runtime_fns=runtime_fns,
    )


def _apply_q_optional_ok(apply_q_fn: ApplyQFn, q_map, ids):
    if apply_q_fn is apply_q:
        return apply_q_ok(q_map, ids)
    if apply_q_fn is apply_q_ok:
        return apply_q_fn(q_map, ids)
    result = call_with_optional_kwargs(
        apply_q_fn, {"return_ok": True}, q_map, ids
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def _resolve_guard_cfg(guard_cfg: GuardConfig | None, cfg: Cnf2Config | None):
    if guard_cfg is None and cfg is not None and cfg.guard_cfg is not None:
        return cfg.guard_cfg
    return guard_cfg


def cycle_candidates_static(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn = commit_stratum_static,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    safe_gather_ok_bound_fn=None,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    if cfg is not None and runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.runtime_fns
    if cfg is not None and cfg.policy_binding is not None:
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            raise PrismPolicyBindingError(
                "cycle_candidates_static received cfg.policy_binding value-mode; "
                "use cycle_candidates_value",
                context="cycle_candidates_static",
                policy_mode="static",
            )
        if safe_gather_policy is None:
            safe_gather_policy = require_static_policy(
                cfg.policy_binding, context="cycle_candidates_static"
            )
    if cfg is not None and cfg.safe_gather_policy_value is not None:
        raise PrismPolicyBindingError(
            "cycle_candidates_static received cfg.safe_gather_policy_value; "
            "use cycle_candidates_value",
            context="cycle_candidates_static",
            policy_mode="static",
        )
    if safe_gather_policy is None and cfg is not None and cfg.safe_gather_policy is not None:
        safe_gather_policy = cfg.safe_gather_policy
    if cfg is not None and cfg.safe_gather_ok_bound_fn is not None:
        if safe_gather_ok_bound_fn is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_static received safe_gather_ok_bound_fn twice",
                context="cycle_candidates_static",
                policy_mode="static",
            )
        safe_gather_ok_bound_fn = cfg.safe_gather_ok_bound_fn
    if (
        cfg is not None
        and cfg.safe_gather_ok_fn is not None
        and safe_gather_ok_fn is _jax_safe.safe_gather_1d_ok
    ):
        safe_gather_ok_fn = cfg.safe_gather_ok_fn
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    if safe_gather_ok_bound_fn is not None:
        if safe_gather_ok_fn is not _jax_safe.safe_gather_1d_ok:
            raise PrismPolicyBindingError(
                "cycle_candidates_static received both safe_gather_ok_fn and "
                "safe_gather_ok_bound_fn",
                context="cycle_candidates_static",
                policy_mode="static",
            )
        if guard_cfg is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_static received guard_cfg with "
                "safe_gather_ok_bound_fn; bind guard config into safe_gather_ok_fn",
                context="cycle_candidates_static",
                policy_mode="static",
            )
        if commit_stratum_fn in (commit_stratum, commit_stratum_static):
            commit_stratum_fn = commit_stratum_bound
        if commit_stratum_fn is not commit_stratum_bound:
            raise PrismPolicyBindingError(
                "cycle_candidates_static received safe_gather_ok_bound_fn without commit_stratum_bound",
                context="cycle_candidates_static",
                policy_mode="static",
            )
        safe_gather_ok_fn = safe_gather_ok_bound_fn
    guard_cfg = _resolve_guard_cfg(guard_cfg, cfg)
    if commit_stratum_fn is commit_stratum:
        commit_stratum_fn = commit_stratum_static
    commit_fns = Cnf2CommitInputs(
        intern_fn=intern_fn,
        commit_stratum_fn=commit_stratum_fn,
    )
    state, frontier, strata, q_map = _cycle_candidates_core_static_bound(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
        commit_fns=commit_fns,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=safe_gather_ok_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )
    return state.ledger, frontier, strata, q_map


def cycle_candidates_static_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn = commit_stratum_static,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    safe_gather_ok_bound_fn=None,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """CNF-2 evaluation (static policy) that preserves LedgerState."""
    if cfg is not None and runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.runtime_fns
    if cfg is not None and cfg.policy_binding is not None:
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            raise PrismPolicyBindingError(
                "cycle_candidates_static_state received cfg.policy_binding value-mode; "
                "use cycle_candidates_value_state",
                context="cycle_candidates_static_state",
                policy_mode="static",
            )
        if safe_gather_policy is None:
            safe_gather_policy = require_static_policy(
                cfg.policy_binding, context="cycle_candidates_static_state"
            )
    if cfg is not None and cfg.safe_gather_policy_value is not None:
        raise PrismPolicyBindingError(
            "cycle_candidates_static_state received cfg.safe_gather_policy_value; "
            "use cycle_candidates_value_state",
            context="cycle_candidates_static_state",
            policy_mode="static",
        )
    if safe_gather_policy is None and cfg is not None and cfg.safe_gather_policy is not None:
        safe_gather_policy = cfg.safe_gather_policy
    if cfg is not None and cfg.safe_gather_ok_bound_fn is not None:
        if safe_gather_ok_bound_fn is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_static_state received safe_gather_ok_bound_fn twice",
                context="cycle_candidates_static_state",
                policy_mode="static",
            )
        safe_gather_ok_bound_fn = cfg.safe_gather_ok_bound_fn
    if (
        cfg is not None
        and cfg.safe_gather_ok_fn is not None
        and safe_gather_ok_fn is _jax_safe.safe_gather_1d_ok
    ):
        safe_gather_ok_fn = cfg.safe_gather_ok_fn
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    if safe_gather_ok_bound_fn is not None:
        if safe_gather_ok_fn is not _jax_safe.safe_gather_1d_ok:
            raise PrismPolicyBindingError(
                "cycle_candidates_static_state received both safe_gather_ok_fn and "
                "safe_gather_ok_bound_fn",
                context="cycle_candidates_static_state",
                policy_mode="static",
            )
        if guard_cfg is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_static_state received guard_cfg with "
                "safe_gather_ok_bound_fn; bind guard config into safe_gather_ok_fn",
                context="cycle_candidates_static_state",
                policy_mode="static",
            )
        if commit_stratum_fn in (commit_stratum, commit_stratum_static):
            commit_stratum_fn = commit_stratum_bound
        if commit_stratum_fn is not commit_stratum_bound:
            raise PrismPolicyBindingError(
                "cycle_candidates_static_state received safe_gather_ok_bound_fn without commit_stratum_bound",
                context="cycle_candidates_static_state",
                policy_mode="static",
            )
        safe_gather_ok_fn = safe_gather_ok_bound_fn
    guard_cfg = _resolve_guard_cfg(guard_cfg, cfg)
    if commit_stratum_fn is commit_stratum:
        commit_stratum_fn = commit_stratum_static
    commit_fns = Cnf2CommitInputs(
        intern_fn=intern_fn,
        commit_stratum_fn=commit_stratum_fn,
    )
    return _cycle_candidates_core_static_bound(
        state,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
        commit_fns=commit_fns,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=safe_gather_ok_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_value(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn = commit_stratum_value,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    safe_gather_ok_bound_fn=None,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    if cfg is not None and runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.runtime_fns
    if cfg is not None and cfg.policy_binding is not None:
        if cfg.policy_binding.mode == PolicyMode.STATIC:
            raise PrismPolicyBindingError(
                "cycle_candidates_value received cfg.policy_binding static-mode; "
                "use cycle_candidates_static",
                context="cycle_candidates_value",
                policy_mode="value",
            )
        if safe_gather_policy_value is None:
            safe_gather_policy_value = require_value_policy(
                cfg.policy_binding, context="cycle_candidates_value"
            )
    if cfg is not None and cfg.safe_gather_policy is not None:
        raise PrismPolicyBindingError(
            "cycle_candidates_value received cfg.safe_gather_policy; "
            "use cycle_candidates_static",
            context="cycle_candidates_value",
            policy_mode="value",
        )
    if safe_gather_ok_bound_fn is not None or (
        cfg is not None and cfg.safe_gather_ok_bound_fn is not None
    ):
        raise PrismPolicyBindingError(
            "cycle_candidates_value received safe_gather_ok_bound_fn; "
            "use cycle_candidates_static",
            context="cycle_candidates_value",
            policy_mode="value",
        )
    if (
        safe_gather_policy_value is None
        and cfg is not None
        and cfg.safe_gather_policy_value is not None
    ):
        safe_gather_policy_value = cfg.safe_gather_policy_value
    if safe_gather_policy_value is None:
        safe_gather_policy_value = POLICY_VALUE_DEFAULT
    guard_cfg = _resolve_guard_cfg(guard_cfg, cfg)
    safe_gather_ok_value_fn = resolve_safe_gather_ok_value_fn(
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        guard_cfg=guard_cfg,
    )
    if commit_stratum_fn is commit_stratum:
        commit_stratum_fn = commit_stratum_value
    commit_fns = Cnf2CommitInputs(
        intern_fn=intern_fn,
        commit_stratum_fn=commit_stratum_fn,
    )
    state, frontier, strata, q_map = _cycle_candidates_core_value_bound(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        safe_gather_policy_value=safe_gather_policy_value,
        guard_cfg=guard_cfg,
        commit_fns=commit_fns,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=None,
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )
    return state.ledger, frontier, strata, q_map


def cycle_candidates_value_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn = commit_stratum_value,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    safe_gather_ok_bound_fn=None,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """CNF-2 evaluation (policy as JAX value) that preserves LedgerState."""
    if cfg is not None and runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.runtime_fns
    if cfg is not None and cfg.policy_binding is not None:
        if cfg.policy_binding.mode == PolicyMode.STATIC:
            raise PrismPolicyBindingError(
                "cycle_candidates_value_state received cfg.policy_binding static-mode; "
                "use cycle_candidates_static_state",
                context="cycle_candidates_value_state",
                policy_mode="value",
            )
        if safe_gather_policy_value is None:
            safe_gather_policy_value = require_value_policy(
                cfg.policy_binding, context="cycle_candidates_value_state"
            )
    if cfg is not None and cfg.safe_gather_policy is not None:
        raise PrismPolicyBindingError(
            "cycle_candidates_value_state received cfg.safe_gather_policy; "
            "use cycle_candidates_static_state",
            context="cycle_candidates_value_state",
            policy_mode="value",
        )
    if safe_gather_ok_bound_fn is not None or (
        cfg is not None and cfg.safe_gather_ok_bound_fn is not None
    ):
        raise PrismPolicyBindingError(
            "cycle_candidates_value_state received safe_gather_ok_bound_fn; "
            "use cycle_candidates_static_state",
            context="cycle_candidates_value_state",
            policy_mode="value",
        )
    if (
        safe_gather_policy_value is None
        and cfg is not None
        and cfg.safe_gather_policy_value is not None
    ):
        safe_gather_policy_value = cfg.safe_gather_policy_value
    if safe_gather_policy_value is None:
        safe_gather_policy_value = POLICY_VALUE_DEFAULT
    guard_cfg = _resolve_guard_cfg(guard_cfg, cfg)
    safe_gather_ok_value_fn = resolve_safe_gather_ok_value_fn(
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        guard_cfg=guard_cfg,
    )
    if commit_stratum_fn is commit_stratum:
        commit_stratum_fn = commit_stratum_value
    commit_fns = Cnf2CommitInputs(
        intern_fn=intern_fn,
        commit_stratum_fn=commit_stratum_fn,
    )
    return _cycle_candidates_core_value_bound(
        state,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        safe_gather_policy_value=safe_gather_policy_value,
        guard_cfg=guard_cfg,
        commit_fns=commit_fns,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=None,
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )


def cycle_candidates(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn = commit_stratum,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    safe_gather_ok_bound_fn=None,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    if cfg is not None and runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.runtime_fns
    if cfg is not None and cfg.safe_gather_policy_value is not None:
        safe_gather_policy_value = cfg.safe_gather_policy_value
    if safe_gather_policy_value is not None:
        if safe_gather_policy is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates received both safe_gather_policy and "
                "safe_gather_policy_value",
                context="cycle_candidates",
                policy_mode="ambiguous",
            )
        return cycle_candidates_value(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cfg,
            safe_gather_policy_value=safe_gather_policy_value,
            guard_cfg=guard_cfg,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            node_batch_fn=node_batch_fn,
            coord_xor_batch_fn=coord_xor_batch_fn,
            emit_candidates_fn=emit_candidates_fn,
            candidate_indices_fn=candidate_indices_fn,
            scatter_drop_fn=scatter_drop_fn,
            commit_stratum_fn=commit_stratum_fn,
            apply_q_fn=apply_q_fn,
            identity_q_fn=identity_q_fn,
            safe_gather_ok_value_fn=safe_gather_ok_value_fn,
            host_bool_value_fn=host_bool_value_fn,
            host_int_value_fn=host_int_value_fn,
            guards_enabled_fn=guards_enabled_fn,
            ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
            runtime_fns=runtime_fns,
        )
    return cycle_candidates_static(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        commit_stratum_fn=commit_stratum_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=safe_gather_ok_fn,
        safe_gather_ok_bound_fn=safe_gather_ok_bound_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_state(
    state: LedgerState,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy: SafetyPolicy | None = None,
    safe_gather_policy_value: PolicyValue | None = None,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn = commit_stratum,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    safe_gather_ok_bound_fn=None,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    if cfg is not None and runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.runtime_fns
    if cfg is not None and cfg.safe_gather_policy_value is not None:
        safe_gather_policy_value = cfg.safe_gather_policy_value
    if safe_gather_policy_value is not None:
        if safe_gather_policy is not None:
            raise PrismPolicyBindingError(
                "cycle_candidates_state received both safe_gather_policy and "
                "safe_gather_policy_value",
                context="cycle_candidates_state",
                policy_mode="ambiguous",
            )
        return cycle_candidates_value_state(
            state,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cfg,
            safe_gather_policy_value=safe_gather_policy_value,
            guard_cfg=guard_cfg,
            intern_fn=intern_fn,
            intern_cfg=intern_cfg,
            node_batch_fn=node_batch_fn,
            coord_xor_batch_fn=coord_xor_batch_fn,
            emit_candidates_fn=emit_candidates_fn,
            candidate_indices_fn=candidate_indices_fn,
            scatter_drop_fn=scatter_drop_fn,
            commit_stratum_fn=commit_stratum_fn,
            apply_q_fn=apply_q_fn,
            identity_q_fn=identity_q_fn,
            safe_gather_ok_value_fn=safe_gather_ok_value_fn,
            host_bool_value_fn=host_bool_value_fn,
            host_int_value_fn=host_int_value_fn,
            guards_enabled_fn=guards_enabled_fn,
            ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
            runtime_fns=runtime_fns,
        )
    return cycle_candidates_static_state(
        state,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        safe_gather_policy=safe_gather_policy,
        guard_cfg=guard_cfg,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        commit_stratum_fn=commit_stratum_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=safe_gather_ok_fn,
        safe_gather_ok_bound_fn=safe_gather_ok_bound_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )


def cycle_candidates_bound(
    ledger,
    frontier_ids,
    cfg: Cnf2BoundConfig,
    *,
    validate_mode: ValidateMode = ValidateMode.NONE,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn | None = None,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """PolicyBinding-required wrapper for cycle_candidates.

    Binds policy + guard exactly once, then delegates to branch-free core.
    """
    if runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.cfg.runtime_fns
    if isinstance(cfg, Cnf2ValueBoundConfig):
        if commit_stratum_fn is None:
            commit_stratum_fn = commit_stratum_value
        cfg_resolved, policy_value = cfg.bind_cfg(
            safe_gather_ok_value_fn=safe_gather_ok_value_fn,
            guard_cfg=guard_cfg,
            commit_stratum_fn=commit_stratum_fn,
        )
        commit_stratum_value_fn = (
            cfg_resolved.commit_stratum_fn or commit_stratum_value
        )
        commit_fns = Cnf2CommitInputs(
            intern_fn=intern_fn,
            commit_stratum_fn=commit_stratum_value_fn,
        )
        state, frontier, strata, q_map = _cycle_candidates_core_value_bound(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cfg_resolved,
            safe_gather_policy_value=policy_value,
            guard_cfg=None,
            commit_fns=commit_fns,
            intern_cfg=intern_cfg,
            node_batch_fn=node_batch_fn,
            coord_xor_batch_fn=coord_xor_batch_fn,
            emit_candidates_fn=emit_candidates_fn,
            candidate_indices_fn=candidate_indices_fn,
            scatter_drop_fn=scatter_drop_fn,
            apply_q_fn=apply_q_fn,
            identity_q_fn=identity_q_fn,
            safe_gather_ok_value_fn=cfg_resolved.safe_gather_ok_value_fn
            or safe_gather_ok_value_fn,
            host_bool_value_fn=host_bool_value_fn,
            host_int_value_fn=host_int_value_fn,
            guards_enabled_fn=guards_enabled_fn,
            ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
            runtime_fns=runtime_fns,
        )
        return state.ledger, frontier, strata, q_map
    if not isinstance(cfg, Cnf2StaticBoundConfig):
        raise PrismPolicyBindingError(
            "cycle_candidates_bound expected Cnf2StaticBoundConfig or Cnf2ValueBoundConfig",
            context="cycle_candidates_bound",
            policy_mode="ambiguous",
        )
    if commit_stratum_fn is None:
        commit_stratum_fn = commit_stratum_bound
    cfg_resolved, policy = cfg.bind_cfg(
        safe_gather_ok_fn=safe_gather_ok_fn,
        guard_cfg=guard_cfg,
        commit_stratum_fn=commit_stratum_fn,
    )
    commit_stratum_static_fn = (
        cfg_resolved.commit_stratum_fn or commit_stratum_bound
    )
    commit_fns = Cnf2CommitInputs(
        intern_fn=intern_fn,
        commit_stratum_fn=commit_stratum_static_fn,
    )
    state, frontier, strata, q_map = _cycle_candidates_core_static_bound(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg_resolved,
        safe_gather_policy=policy,
        guard_cfg=None,
        commit_fns=commit_fns,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=cfg_resolved.safe_gather_ok_bound_fn or safe_gather_ok_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )
    return state.ledger, frontier, strata, q_map


def cycle_candidates_bound_state(
    state: LedgerState,
    frontier_ids,
    cfg: Cnf2BoundConfig,
    *,
    validate_mode: ValidateMode = ValidateMode.NONE,
    guard_cfg: GuardConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    coord_xor_batch_fn: CoordXorBatchFn = coord_xor_batch,
    emit_candidates_fn: EmitCandidatesFn = emit_candidates,
    candidate_indices_fn: CandidateIndicesFn = _candidate_indices,
    scatter_drop_fn: ScatterDropFn = _scatter_drop,
    commit_stratum_fn: CommitStratumFn | None = None,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn: SafeGatherOkFn = _jax_safe.safe_gather_1d_ok,
    safe_gather_ok_value_fn: SafeGatherOkValueFn = _jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    runtime_fns: Cnf2RuntimeFns = DEFAULT_CNF2_RUNTIME_FNS,
):
    """PolicyBinding-required wrapper for cycle_candidates returning LedgerState."""
    if runtime_fns is DEFAULT_CNF2_RUNTIME_FNS:
        runtime_fns = cfg.cfg.runtime_fns
    if isinstance(cfg, Cnf2ValueBoundConfig):
        if commit_stratum_fn is None:
            commit_stratum_fn = commit_stratum_value
        cfg_resolved, policy_value = cfg.bind_cfg(
            safe_gather_ok_value_fn=safe_gather_ok_value_fn,
            guard_cfg=guard_cfg,
            commit_stratum_fn=commit_stratum_fn,
        )
        commit_stratum_value_fn = (
            cfg_resolved.commit_stratum_fn or commit_stratum_value
        )
        commit_fns = Cnf2CommitInputs(
            intern_fn=intern_fn,
            commit_stratum_fn=commit_stratum_value_fn,
        )
        state, frontier, strata, q_map = _cycle_candidates_core_value_bound(
            state,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cfg_resolved,
            safe_gather_policy_value=policy_value,
            guard_cfg=None,
            commit_fns=commit_fns,
            intern_cfg=intern_cfg,
            node_batch_fn=node_batch_fn,
            coord_xor_batch_fn=coord_xor_batch_fn,
            emit_candidates_fn=emit_candidates_fn,
            candidate_indices_fn=candidate_indices_fn,
            scatter_drop_fn=scatter_drop_fn,
            apply_q_fn=apply_q_fn,
            identity_q_fn=identity_q_fn,
            safe_gather_ok_value_fn=cfg_resolved.safe_gather_ok_value_fn
            or safe_gather_ok_value_fn,
            host_bool_value_fn=host_bool_value_fn,
            host_int_value_fn=host_int_value_fn,
            guards_enabled_fn=guards_enabled_fn,
            ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
            runtime_fns=runtime_fns,
        )
        return state, frontier, strata, q_map
    if commit_stratum_fn is None:
        commit_stratum_fn = commit_stratum_bound
    cfg_resolved, policy = cfg.bind_cfg(
        safe_gather_ok_fn=safe_gather_ok_fn,
        guard_cfg=guard_cfg,
        commit_stratum_fn=commit_stratum_fn,
    )
    commit_stratum_static_fn = (
        cfg_resolved.commit_stratum_fn or commit_stratum_bound
    )
    commit_fns = Cnf2CommitInputs(
        intern_fn=intern_fn,
        commit_stratum_fn=commit_stratum_static_fn,
    )
    state, frontier, strata, q_map = _cycle_candidates_core_static_bound(
        state,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg_resolved,
        safe_gather_policy=policy,
        guard_cfg=None,
        commit_fns=commit_fns,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        coord_xor_batch_fn=coord_xor_batch_fn,
        emit_candidates_fn=emit_candidates_fn,
        candidate_indices_fn=candidate_indices_fn,
        scatter_drop_fn=scatter_drop_fn,
        apply_q_fn=apply_q_fn,
        identity_q_fn=identity_q_fn,
        safe_gather_ok_fn=cfg_resolved.safe_gather_ok_fn or safe_gather_ok_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        runtime_fns=runtime_fns,
    )
    return state, frontier, strata, q_map


__all__ = [
    "emit_candidates",
    "emit_candidates_cfg",
    "compact_candidates_result",
    "compact_candidates",
    "compact_candidates_cfg",
    "compact_candidates_with_index_result",
    "compact_candidates_with_index",
    "compact_candidates_with_index_cfg",
    "scatter_compacted_ids_cfg",
    "intern_candidates",
    "intern_candidates_cfg",
    "cycle_candidates",
    "cycle_candidates_state",
    "cycle_candidates_static",
    "cycle_candidates_static_state",
    "cycle_candidates_value",
    "cycle_candidates_value_state",
    "cycle_candidates_bound",
    "cycle_candidates_bound_state",
]
