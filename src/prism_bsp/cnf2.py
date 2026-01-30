import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from prism_core import jax_safe as _jax_safe
from prism_core.di import call_with_optional_kwargs
from prism_core.guards import (
    GuardConfig,
    resolve_safe_gather_ok_fn,
    resolve_safe_gather_ok_value_fn,
)
from prism_core.compact import scatter_compacted_ids
from prism_core.safety import (
    PolicyBinding,
    PolicyMode,
    coerce_policy_mode,
    DEFAULT_SAFETY_POLICY,
    POLICY_VALUE_DEFAULT,
    PolicyValue,
    SafetyPolicy,
    oob_any,
    oob_any_value,
)
from prism_core.errors import PrismPolicyBindingError, PrismCnf2ModeConflictError
from prism_core.modes import ValidateMode, coerce_validate_mode, Cnf2Mode, coerce_cnf2_mode
from prism_coord.coord import coord_xor_batch
from prism_ledger.intern import intern_nodes
from prism_ledger.config import InternConfig
from prism_metrics.metrics import _cnf2_metrics_enabled, _cnf2_metrics_update
from prism_bsp.config import Cnf2Config
from prism_semantics.commit import _identity_q, apply_q, commit_stratum
from prism_vm_core.candidates import _candidate_indices, candidate_indices_cfg
from prism_vm_core.domains import (
    _committed_ids,
    _host_bool_value,
    _host_int_value,
    _provisional_ids,
)
from prism_vm_core.gating import _cnf2_enabled, _cnf2_slot1_enabled
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
from prism_vm_core.structures import CandidateBuffer, Stratum, NodeBatch
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
    LedgerRootsHashFn,
    NodeBatchFn,
    ScatterDropFn,
)


_TEST_GUARDS = _jax_safe.TEST_GUARDS
_scatter_drop = _jax_safe.scatter_drop


def _node_batch(op, a1, a2) -> NodeBatch:
    return NodeBatch(op=op, a1=a1, a2=a2)


def emit_candidates(ledger, frontier_ids):
    num_frontier = frontier_ids.shape[0]
    size = num_frontier * 2
    enabled = jnp.zeros(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = jnp.zeros(size, dtype=jnp.int32)
    arg2 = jnp.zeros(size, dtype=jnp.int32)

    if num_frontier == 0:
        return CandidateBuffer(enabled=enabled, opcode=opcode, arg1=arg1, arg2=arg2)

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
    if cfg is not None and cfg.emit_candidates_fn is not None:
        emit_candidates_fn = cfg.emit_candidates_fn
    return emit_candidates_fn(ledger, frontier_ids)


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
    candidate_indices_fn = _candidate_indices
    if cfg is not None and cfg.candidate_indices_fn is not None:
        candidate_indices_fn = cfg.candidate_indices_fn
    if cfg is not None and cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
        candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
    return compact_candidates(candidates, candidate_indices_fn=candidate_indices_fn)


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
    candidate_indices_fn = _candidate_indices
    if cfg is not None and cfg.candidate_indices_fn is not None:
        candidate_indices_fn = cfg.candidate_indices_fn
    if cfg is not None and cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
        candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
    return compact_candidates_with_index(
        candidates, candidate_indices_fn=candidate_indices_fn
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
    scatter_drop_fn = _scatter_drop
    if cfg is not None and cfg.scatter_drop_fn is not None:
        scatter_drop_fn = cfg.scatter_drop_fn
    return _scatter_compacted_ids(
        comp_idx,
        ids_compact,
        count,
        size,
        scatter_drop_fn=scatter_drop_fn,
    )


def intern_candidates(
    ledger,
    candidates,
    *,
    compact_candidates_fn=compact_candidates,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
):
    if intern_cfg is not None and intern_fn is intern_nodes:
        intern_fn = partial(intern_nodes, cfg=intern_cfg)
    compacted, count = compact_candidates_fn(candidates)
    enabled = compacted.enabled.astype(jnp.int32)
    ops = jnp.where(enabled, compacted.opcode, jnp.int32(0))
    a1 = jnp.where(enabled, compacted.arg1, jnp.int32(0))
    a2 = jnp.where(enabled, compacted.arg2, jnp.int32(0))
    ids, new_ledger = intern_fn(ledger, node_batch_fn(ops, a1, a2))
    return ids, new_ledger, count


def intern_candidates_cfg(
    ledger,
    candidates,
    *,
    cfg: Cnf2Config | None = None,
    compact_candidates_fn=compact_candidates,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
):
    """Interface/Control wrapper for intern_candidates with DI bundle."""
    if cfg is not None:
        intern_cfg = intern_cfg if intern_cfg is not None else cfg.intern_cfg
        if cfg.intern_fn is not None and intern_fn is intern_nodes:
            intern_fn = cfg.intern_fn
        if cfg.node_batch_fn is not None and node_batch_fn is _node_batch:
            node_batch_fn = cfg.node_batch_fn
        if cfg.candidate_indices_fn is not None:
            compact_candidates_fn = partial(
                compact_candidates, candidate_indices_fn=cfg.candidate_indices_fn
            )
    return intern_candidates(
        ledger,
        candidates,
        compact_candidates_fn=compact_candidates_fn,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
    )

def _cycle_candidates_core_common(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    policy_mode: PolicyMode | str,
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
    safe_gather_ok_fn=_jax_safe.safe_gather_1d_ok,
    safe_gather_ok_value_fn=_jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    cnf2_enabled_fn=_cnf2_enabled,
    cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    cnf2_metrics_enabled_fn=_cnf2_metrics_enabled,
    cnf2_metrics_update_fn=_cnf2_metrics_update,
):
    def _maybe_override(current, default, override):
        if override is None:
            return current
        if current is default:
            return override
        return current

    cnf2_mode = None
    if cfg is not None:
        cnf2_mode = cfg.cnf2_mode
        guard_cfg = cfg.guard_cfg if guard_cfg is None else guard_cfg
        intern_cfg = intern_cfg if intern_cfg is not None else cfg.intern_cfg
        if cfg.coord_cfg is not None and coord_xor_batch_fn is coord_xor_batch:
            coord_xor_batch_fn = partial(coord_xor_batch, cfg=cfg.coord_cfg)
        intern_fn = _maybe_override(intern_fn, intern_nodes, cfg.intern_fn)
        node_batch_fn = _maybe_override(node_batch_fn, _node_batch, cfg.node_batch_fn)
        coord_xor_batch_fn = _maybe_override(
            coord_xor_batch_fn, coord_xor_batch, cfg.coord_xor_batch_fn
        )
        emit_candidates_fn = _maybe_override(
            emit_candidates_fn, emit_candidates, cfg.emit_candidates_fn
        )
        candidate_indices_fn = _maybe_override(
            candidate_indices_fn, _candidate_indices, cfg.candidate_indices_fn
        )
        if cfg.compact_cfg is not None and candidate_indices_fn is _candidate_indices:
            candidate_indices_fn = partial(candidate_indices_cfg, compact_cfg=cfg.compact_cfg)
        scatter_drop_fn = _maybe_override(
            scatter_drop_fn, _scatter_drop, cfg.scatter_drop_fn
        )
        commit_stratum_fn = _maybe_override(
            commit_stratum_fn, commit_stratum, cfg.commit_stratum_fn
        )
        apply_q_fn = _maybe_override(apply_q_fn, apply_q, cfg.apply_q_fn)
        identity_q_fn = _maybe_override(identity_q_fn, _identity_q, cfg.identity_q_fn)
        safe_gather_ok_fn = _maybe_override(
            safe_gather_ok_fn, _jax_safe.safe_gather_1d_ok, cfg.safe_gather_ok_fn
        )
        safe_gather_ok_value_fn = _maybe_override(
            safe_gather_ok_value_fn,
            _jax_safe.safe_gather_1d_ok_value,
            cfg.safe_gather_ok_value_fn,
        )
        host_bool_value_fn = _maybe_override(
            host_bool_value_fn, _host_bool_value, cfg.host_bool_value_fn
        )
        host_int_value_fn = _maybe_override(
            host_int_value_fn, _host_int_value, cfg.host_int_value_fn
        )
        guards_enabled_fn = _maybe_override(
            guards_enabled_fn, _guards_enabled, cfg.guards_enabled_fn
        )
        ledger_roots_hash_host_fn = _maybe_override(
            ledger_roots_hash_host_fn,
            _ledger_roots_hash_host,
            cfg.ledger_roots_hash_host_fn,
        )
        if cfg.cnf2_metrics_enabled_fn is not None and cnf2_metrics_enabled_fn is _cnf2_metrics_enabled:
            cnf2_metrics_enabled_fn = cfg.cnf2_metrics_enabled_fn
        if cfg.cnf2_metrics_update_fn is not None and cnf2_metrics_update_fn is _cnf2_metrics_update:
            cnf2_metrics_update_fn = cfg.cnf2_metrics_update_fn
        if cfg.cnf2_enabled_fn is not None and cnf2_enabled_fn is _cnf2_enabled:
            cnf2_enabled_fn = cfg.cnf2_enabled_fn
        if cfg.cnf2_slot1_enabled_fn is not None and cnf2_slot1_enabled_fn is _cnf2_slot1_enabled:
            cnf2_slot1_enabled_fn = cfg.cnf2_slot1_enabled_fn
        if cfg.flags is not None:
            if cfg.flags.enabled is not None and cnf2_enabled_fn is _cnf2_enabled:
                cnf2_enabled_fn = lambda: bool(cfg.flags.enabled)
            if cfg.flags.slot1_enabled is not None and cnf2_slot1_enabled_fn is _cnf2_slot1_enabled:
                cnf2_slot1_enabled_fn = lambda: bool(cfg.flags.slot1_enabled)
        if cfg.policy_binding is not None:
            if safe_gather_policy is not None or safe_gather_policy_value is not None:
                raise PrismPolicyBindingError(
                    "cycle_candidates_core received both policy_binding and "
                    "safe_gather_policy/safe_gather_policy_value",
                    context="cycle_candidates_core",
                    policy_mode="ambiguous",
                )
            policy_mode = cfg.policy_binding.mode
            if policy_mode == PolicyMode.VALUE:
                safe_gather_policy_value = cfg.policy_binding.policy_value
            else:
                safe_gather_policy = cfg.policy_binding.policy

    policy_mode = coerce_policy_mode(policy_mode, context="cycle_candidates_core")
    if policy_mode == PolicyMode.STATIC and safe_gather_policy is None:
        raise PrismPolicyBindingError(
            "cycle_candidates core (static) requires safe_gather_policy",
            context="cycle_candidates_core",
            policy_mode="static",
        )
    if policy_mode == PolicyMode.VALUE and safe_gather_policy_value is None:
        raise PrismPolicyBindingError(
            "cycle_candidates core (value) requires safe_gather_policy_value",
            context="cycle_candidates_core",
            policy_mode="value",
        )
    if cnf2_mode is not None:
        mode = coerce_cnf2_mode(cnf2_mode, context="cycle_candidates_core")
        if mode != Cnf2Mode.AUTO:
            if (
                cfg is not None
                and cfg.flags is not None
            ) or cnf2_enabled_fn is not _cnf2_enabled or cnf2_slot1_enabled_fn is not _cnf2_slot1_enabled:
                raise PrismCnf2ModeConflictError(
                    "cycle_candidates_core received cnf2_mode alongside cnf2_flags or cnf2_*_enabled_fn",
                    context="cycle_candidates_core",
                )
            enabled_value = mode in (Cnf2Mode.BASE, Cnf2Mode.SLOT1)
            slot1_value = mode == Cnf2Mode.SLOT1
            cnf2_enabled_fn = lambda: enabled_value
            cnf2_slot1_enabled_fn = lambda: slot1_value
    if intern_cfg is not None and intern_fn is intern_nodes:
        intern_fn = partial(intern_nodes, cfg=intern_cfg)
    if intern_cfg is not None and coord_xor_batch_fn is coord_xor_batch:
        coord_xor_batch_fn = partial(coord_xor_batch, intern_cfg=intern_cfg)
    # BSPᵗ: temporal superstep / barrier semantics.
    frontier_ids = _committed_ids(frontier_ids)
    if not cnf2_enabled_fn():
        # CNF-2 candidate pipeline is staged for m2+ (plan); guard at entry.
        # See IMPLEMENTATION_PLAN.md (m2 CNF-2 pipeline).
        raise RuntimeError("cycle_candidates disabled until m2 (set PRISM_ENABLE_CNF2=1)")
    # SYNC: host read to short-circuit on corrupt ledgers (m1).
    if host_bool_value_fn(ledger.corrupt):
        empty = Stratum(start=ledger.count.astype(jnp.int32), count=jnp.int32(0))
        return (
            ledger,
            _provisional_ids(frontier_ids.a),
            (empty, empty, empty),
            identity_q_fn,
        )
    frontier_arr = jnp.atleast_1d(frontier_ids.a)
    frontier_ids = _committed_ids(frontier_arr)
    num_frontier = frontier_arr.shape[0]
    if num_frontier == 0:
        empty = Stratum(start=ledger.count.astype(jnp.int32), count=jnp.int32(0))
        return (
            ledger,
            _provisional_ids(frontier_ids.a),
            (empty, empty, empty),
            identity_q_fn,
        )

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
    if coord_count_i > 0:
        coord_idx_safe = jnp.where(coord_valid, coord_idx, 0)
        coord_left = r_a1[coord_idx_safe][:coord_count_i]
        coord_right = r_a2[coord_idx_safe][:coord_count_i]
        coord_ids_compact, ledger = coord_xor_batch_fn(
            ledger, coord_left, coord_right
        )
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
    ids_compact, ledger0 = intern_fn(ledger, node_batch_fn(ops0, a1_0, a2_0))
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
    slot1_gate = cnf2_slot1_enabled_fn()
    slot1_add = is_add_suc & slot1_gate
    slot1_mul = is_mul_suc & slot1_gate
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
    if slot1_gate:
        slot1_ids, ledger1 = intern_fn(
            ledger0, node_batch_fn(slot1_ops, slot1_a1, slot1_a2)
        )
    else:
        slot1_ids = jnp.zeros_like(rewrite_ids)
        ledger1 = ledger0
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

    wrap_strata = []
    wrap_depths = depths
    next_frontier = base_next
    ledger2 = ledger1
    while host_bool_value_fn(jnp.any((wrap_depths > 0) & (~ledger2.oom))):
        to_wrap = (wrap_depths > 0) & (~ledger2.oom)
        ops = jnp.where(to_wrap, jnp.int32(OP_SUC), jnp.int32(0))
        a1 = jnp.where(to_wrap, next_frontier, jnp.int32(0))
        a2 = jnp.zeros_like(a1)
        start = host_int_value_fn(ledger2.count)
        new_ids, ledger2 = intern_fn(ledger2, node_batch_fn(ops, a1, a2))
        end = host_int_value_fn(ledger2.count)
        if end > start:
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
    if wrap_strata:
        start2_i = wrap_strata[0][0]
        count2_i = sum(count for _, count in wrap_strata)
    else:
        start2_i = host_int_value_fn(ledger1.count)
        count2_i = 0
    stratum2 = Stratum(
        start=jnp.int32(start2_i), count=jnp.int32(count2_i)
    )
    if cnf2_metrics_enabled_fn():
        rewrite_child = host_int_value_fn(count0)
        changed_count = host_int_value_fn(
            jnp.sum(changed_mask.astype(jnp.int32))
        )
        cnf2_metrics_update_fn(rewrite_child, changed_count, int(count2_i))
    mode = coerce_validate_mode(validate_mode, context="cycle_candidates")
    if guards_enabled_fn() and mode == ValidateMode.NONE:
        mode = ValidateMode.STRICT
    if policy_mode == PolicyMode.STATIC:
        commit_optional = {
            "safe_gather_policy": safe_gather_policy,
            "safe_gather_ok_fn": safe_gather_ok_fn,
            "guard_cfg": guard_cfg,
        }
    else:
        commit_optional = {
            "safe_gather_policy_value": safe_gather_policy_value,
            "safe_gather_ok_value_fn": safe_gather_ok_value_fn,
            "guard_cfg": guard_cfg,
        }
    ledger2, _, q_map = call_with_optional_kwargs(
        commit_stratum_fn,
        commit_optional,
        ledger2,
        stratum0,
        validate_mode=mode,
        intern_fn=intern_fn,
    )
    ledger2, _, q_map = call_with_optional_kwargs(
        commit_stratum_fn,
        commit_optional,
        ledger2,
        stratum1,
        prior_q=q_map,
        validate_mode=mode,
        intern_fn=intern_fn,
    )
    # Wrapper strata are micro-strata in s=2; commit in order for hyperstrata visibility.
    for start_i, count_i in wrap_strata:
        micro_stratum = Stratum(
            start=jnp.int32(start_i), count=jnp.int32(count_i)
        )
        ledger2, _, q_map = call_with_optional_kwargs(
            commit_stratum_fn,
            commit_optional,
            ledger2,
            micro_stratum,
            prior_q=q_map,
            validate_mode=mode,
            intern_fn=intern_fn,
        )
    next_frontier = _provisional_ids(next_frontier)
    meta = getattr(q_map, "_prism_meta", None)
    post_ids = None
    ok = None
    if meta is not None:
        if policy_mode == PolicyMode.STATIC:
            if meta.safe_gather_policy_value is not None:
                raise PrismPolicyBindingError(
                    "cycle_candidates core (static) received policy_value metadata",
                    context="cycle_candidates_core",
                    policy_mode="static",
                )
            if meta.safe_gather_policy is not None:
                post_ids, ok = _apply_q_optional_ok(apply_q_fn, q_map, next_frontier)
                if ok is not None:
                    corrupt = oob_any(ok, policy=meta.safe_gather_policy)
                    ledger2 = ledger2._replace(corrupt=ledger2.corrupt | corrupt)
        else:
            if meta.safe_gather_policy is not None:
                raise PrismPolicyBindingError(
                    "cycle_candidates core (value) received policy metadata",
                    context="cycle_candidates_core",
                    policy_mode="value",
                )
            if meta.safe_gather_policy_value is not None:
                post_ids, ok = _apply_q_optional_ok(apply_q_fn, q_map, next_frontier)
                if ok is not None:
                    corrupt = oob_any_value(
                        ok, policy_value=meta.safe_gather_policy_value
                    )
                    ledger2 = ledger2._replace(corrupt=ledger2.corrupt | corrupt)
    if _TEST_GUARDS:
        pre_hash = ledger_roots_hash_host_fn(ledger2, next_frontier.a)
        if post_ids is None:
            post_ids, ok = _apply_q_optional_ok(apply_q_fn, q_map, next_frontier)
        post_ids = post_ids.a
        post_hash = ledger_roots_hash_host_fn(ledger2, post_ids)
        if pre_hash != post_hash:
            raise RuntimeError("BSPᵗ projection changed root structure")
    return ledger2, next_frontier, (stratum0, stratum1, stratum2), q_map


def _cycle_candidates_core_static(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy: SafetyPolicy,
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
    safe_gather_ok_fn=_jax_safe.safe_gather_1d_ok,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    cnf2_enabled_fn=_cnf2_enabled,
    cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    cnf2_metrics_enabled_fn=_cnf2_metrics_enabled,
    cnf2_metrics_update_fn=_cnf2_metrics_update,
):
    return _cycle_candidates_core_common(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        policy_mode=PolicyMode.STATIC,
        safe_gather_policy=safe_gather_policy,
        safe_gather_policy_value=None,
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
        safe_gather_ok_value_fn=None,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        cnf2_enabled_fn=cnf2_enabled_fn,
        cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
        cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
        cnf2_metrics_update_fn=cnf2_metrics_update_fn,
    )


def _cycle_candidates_core_value(
    ledger,
    frontier_ids,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
    safe_gather_policy_value: PolicyValue,
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
    safe_gather_ok_value_fn=_jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    cnf2_enabled_fn=_cnf2_enabled,
    cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    cnf2_metrics_enabled_fn=_cnf2_metrics_enabled,
    cnf2_metrics_update_fn=_cnf2_metrics_update,
):
    return _cycle_candidates_core_common(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        policy_mode=PolicyMode.VALUE,
        safe_gather_policy=None,
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
        safe_gather_ok_fn=None,
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        cnf2_enabled_fn=cnf2_enabled_fn,
        cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
        cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
        cnf2_metrics_update_fn=cnf2_metrics_update_fn,
    )


def _apply_q_optional_ok(apply_q_fn, q_map, ids):
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
    commit_stratum_fn: CommitStratumFn = commit_stratum,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_fn=_jax_safe.safe_gather_1d_ok,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    cnf2_enabled_fn=_cnf2_enabled,
    cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    cnf2_metrics_enabled_fn=_cnf2_metrics_enabled,
    cnf2_metrics_update_fn=_cnf2_metrics_update,
):
    if cfg is not None and cfg.policy_binding is not None:
        if cfg.policy_binding.mode == PolicyMode.VALUE:
            raise PrismPolicyBindingError(
                "cycle_candidates_static received cfg.policy_binding value-mode; "
                "use cycle_candidates_value",
                context="cycle_candidates_static",
                policy_mode="static",
            )
        if safe_gather_policy is None:
            safe_gather_policy = cfg.policy_binding.policy
    if cfg is not None and cfg.safe_gather_policy_value is not None:
        raise PrismPolicyBindingError(
            "cycle_candidates_static received cfg.safe_gather_policy_value; "
            "use cycle_candidates_value",
            context="cycle_candidates_static",
            policy_mode="static",
        )
    if safe_gather_policy is None and cfg is not None and cfg.safe_gather_policy is not None:
        safe_gather_policy = cfg.safe_gather_policy
    if safe_gather_policy is None:
        safe_gather_policy = DEFAULT_SAFETY_POLICY
    guard_cfg = _resolve_guard_cfg(guard_cfg, cfg)
    safe_gather_ok_fn = resolve_safe_gather_ok_fn(
        safe_gather_ok_fn=safe_gather_ok_fn,
        policy=safe_gather_policy,
        guard_cfg=guard_cfg,
    )
    return _cycle_candidates_core_static(
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
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        cnf2_enabled_fn=cnf2_enabled_fn,
        cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
        cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
        cnf2_metrics_update_fn=cnf2_metrics_update_fn,
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
    commit_stratum_fn: CommitStratumFn = commit_stratum,
    apply_q_fn: ApplyQFn = apply_q,
    identity_q_fn: IdentityQFn = _identity_q,
    safe_gather_ok_value_fn=_jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    cnf2_enabled_fn=_cnf2_enabled,
    cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    cnf2_metrics_enabled_fn=_cnf2_metrics_enabled,
    cnf2_metrics_update_fn=_cnf2_metrics_update,
):
    if cfg is not None and cfg.policy_binding is not None:
        if cfg.policy_binding.mode == PolicyMode.STATIC:
            raise PrismPolicyBindingError(
                "cycle_candidates_value received cfg.policy_binding static-mode; "
                "use cycle_candidates_static",
                context="cycle_candidates_value",
                policy_mode="value",
            )
        if safe_gather_policy_value is None:
            safe_gather_policy_value = cfg.policy_binding.policy_value
    if cfg is not None and cfg.safe_gather_policy is not None:
        raise PrismPolicyBindingError(
            "cycle_candidates_value received cfg.safe_gather_policy; "
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
    return _cycle_candidates_core_value(
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
        safe_gather_ok_fn=None,
        safe_gather_ok_value_fn=safe_gather_ok_value_fn,
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        cnf2_enabled_fn=cnf2_enabled_fn,
        cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
        cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
        cnf2_metrics_update_fn=cnf2_metrics_update_fn,
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
    safe_gather_ok_fn=_jax_safe.safe_gather_1d_ok,
    safe_gather_ok_value_fn=_jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    cnf2_enabled_fn=_cnf2_enabled,
    cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    cnf2_metrics_enabled_fn=_cnf2_metrics_enabled,
    cnf2_metrics_update_fn=_cnf2_metrics_update,
):
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
            cnf2_enabled_fn=cnf2_enabled_fn,
            cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
            cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
            cnf2_metrics_update_fn=cnf2_metrics_update_fn,
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
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        cnf2_enabled_fn=cnf2_enabled_fn,
        cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
        cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
        cnf2_metrics_update_fn=cnf2_metrics_update_fn,
    )


def cycle_candidates_bound(
    ledger,
    frontier_ids,
    policy_binding: PolicyBinding,
    validate_mode: ValidateMode = ValidateMode.NONE,
    *,
    cfg: Cnf2Config | None = None,
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
    safe_gather_ok_fn=_jax_safe.safe_gather_1d_ok,
    safe_gather_ok_value_fn=_jax_safe.safe_gather_1d_ok_value,
    host_bool_value_fn: HostBoolValueFn = _host_bool_value,
    host_int_value_fn: HostIntValueFn = _host_int_value,
    guards_enabled_fn: GuardsEnabledFn = _guards_enabled,
    ledger_roots_hash_host_fn: LedgerRootsHashFn = _ledger_roots_hash_host,
    cnf2_enabled_fn=_cnf2_enabled,
    cnf2_slot1_enabled_fn=_cnf2_slot1_enabled,
    cnf2_metrics_enabled_fn=_cnf2_metrics_enabled,
    cnf2_metrics_update_fn=_cnf2_metrics_update,
):
    """PolicyBinding-required wrapper for cycle_candidates."""
    if cfg is None:
        cfg = Cnf2Config(policy_binding=policy_binding)
    else:
        cfg = replace(
            cfg,
            policy_binding=policy_binding,
            safe_gather_policy=None,
            safe_gather_policy_value=None,
        )
    if policy_binding.mode == PolicyMode.VALUE:
        return cycle_candidates_value(
            ledger,
            frontier_ids,
            validate_mode=validate_mode,
            cfg=cfg,
            safe_gather_policy_value=policy_binding.policy_value,
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
            cnf2_enabled_fn=cnf2_enabled_fn,
            cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
            cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
            cnf2_metrics_update_fn=cnf2_metrics_update_fn,
        )
    return cycle_candidates_static(
        ledger,
        frontier_ids,
        validate_mode=validate_mode,
        cfg=cfg,
        safe_gather_policy=policy_binding.policy,
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
        host_bool_value_fn=host_bool_value_fn,
        host_int_value_fn=host_int_value_fn,
        guards_enabled_fn=guards_enabled_fn,
        ledger_roots_hash_host_fn=ledger_roots_hash_host_fn,
        cnf2_enabled_fn=cnf2_enabled_fn,
        cnf2_slot1_enabled_fn=cnf2_slot1_enabled_fn,
        cnf2_metrics_enabled_fn=cnf2_metrics_enabled_fn,
        cnf2_metrics_update_fn=cnf2_metrics_update_fn,
    )


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
    "cycle_candidates_static",
    "cycle_candidates_value",
    "cycle_candidates_bound",
]
