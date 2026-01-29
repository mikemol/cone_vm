import jax
import jax.numpy as jnp
from jax import lax
from functools import lru_cache, partial

from prism_ledger.intern import intern_nodes
from prism_ledger.config import InternConfig
from prism_vm_core.domains import _host_raise_if_bad
from prism_vm_core.ontology import OP_ADD, OP_MUL, OP_SUC, OP_ZERO, ZERO_PTR
from prism_vm_core.structures import NodeBatch
from prism_vm_core.protocols import HostRaiseFn, InternFn, NodeBatchFn
from prism_bsp.config import IntrinsicConfig, DEFAULT_INTRINSIC_CONFIG


def _node_batch(op, a1, a2) -> NodeBatch:
    return NodeBatch(op=op, a1=a1, a2=a2)


def _cycle_intrinsic_impl(
    ledger,
    frontier_ids,
    *,
    intern_fn=intern_nodes,
    node_batch_fn=_node_batch,
):
    # m1 evaluator: intrinsic rewrite steps on Ledger (CNF-2 gated off).
    # See IMPLEMENTATION_PLAN.md (m1 intrinsic evaluator).
    def _skip(_):
        return ledger, frontier_ids

    def _do(_):
        ledger_local = ledger

        def _peel_one(ptr):
            def cond(state):
                curr, _ = state
                return ledger_local.opcode[curr] == OP_SUC

            def body(state):
                curr, depth = state
                return ledger_local.arg1[curr], depth + 1

            return lax.while_loop(cond, body, (ptr, jnp.int32(0)))

        base_ids, depths = jax.vmap(_peel_one)(frontier_ids)

        t_ops = ledger_local.opcode[base_ids]
        t_a1 = ledger_local.arg1[base_ids]
        t_a2 = ledger_local.arg2[base_ids]
        op_a1 = ledger_local.opcode[t_a1]
        op_a2 = ledger_local.opcode[t_a2]
        is_add = t_ops == OP_ADD
        is_mul = t_ops == OP_MUL
        is_zero_a1 = op_a1 == OP_ZERO
        is_zero_a2 = op_a2 == OP_ZERO
        is_suc_a1 = op_a1 == OP_SUC
        is_suc_a2 = op_a2 == OP_SUC
        is_add_suc = is_add & (is_suc_a1 | is_suc_a2)
        is_mul_suc = is_mul & (is_suc_a1 | is_suc_a2)
        is_add_zero = is_add & (is_zero_a1 | is_zero_a2)
        is_mul_zero = is_mul & (is_zero_a1 | is_zero_a2)
        is_add_suc = is_add_suc & (~is_add_zero)
        is_mul_suc = is_mul_suc & (~is_mul_zero)
        zero_on_a1 = is_zero_a1
        zero_on_a2 = (~is_zero_a1) & is_zero_a2
        zero_other = jnp.where(zero_on_a1, t_a2, t_a1)

        suc_on_a1 = is_suc_a1
        suc_on_a2 = (~is_suc_a1) & is_suc_a2
        suc_node = jnp.where(suc_on_a1, t_a1, t_a2)
        val_x = ledger_local.arg1[suc_node]
        val_y = jnp.where(suc_on_a1, t_a2, t_a1)

        l1_ops = jnp.zeros_like(t_ops)
        l1_a1 = jnp.zeros_like(t_a1)
        l1_a2 = jnp.zeros_like(t_a2)
        l1_ops = jnp.where(is_add_suc, OP_ADD, l1_ops)
        l1_a1 = jnp.where(is_add_suc, val_x, l1_a1)
        l1_a2 = jnp.where(is_add_suc, val_y, l1_a2)
        l1_ops = jnp.where(is_mul_suc, OP_MUL, l1_ops)
        l1_a1 = jnp.where(is_mul_suc, val_x, l1_a1)
        l1_a2 = jnp.where(is_mul_suc, val_y, l1_a2)

        l1_ids, ledger_local = intern_fn(
            ledger_local, node_batch_fn(l1_ops, l1_a1, l1_a2)
        )

        l2_ops = jnp.zeros_like(t_ops)
        l2_a1 = jnp.zeros_like(t_a1)
        l2_a2 = jnp.zeros_like(t_a2)
        l2_ops = jnp.where(is_add_suc, OP_SUC, l2_ops)
        l2_a1 = jnp.where(is_add_suc, l1_ids, l2_a1)
        l2_ops = jnp.where(is_mul_suc, OP_ADD, l2_ops)
        l2_a1 = jnp.where(is_mul_suc, val_y, l2_a1)
        l2_a2 = jnp.where(is_mul_suc, l1_ids, l2_a2)

        l2_ids, ledger_local = intern_fn(
            ledger_local, node_batch_fn(l2_ops, l2_a1, l2_a2)
        )

        base_next = base_ids
        base_next = jnp.where(is_add_zero, zero_other, base_next)
        base_next = jnp.where(is_mul_zero, jnp.int32(ZERO_PTR), base_next)
        base_next = jnp.where(is_add_suc, l2_ids, base_next)
        base_next = jnp.where(is_mul_suc, l2_ids, base_next)
        changed = base_next != base_ids
        wrap_depth = jnp.where(changed, depths, jnp.int32(0))
        wrap_child = jnp.where(changed, base_next, frontier_ids)

        def wrap_cond(state):
            depth, _, led = state
            return jnp.any((depth > 0) & (~led.oom))

        def wrap_body(state):
            depth, child, led = state
            to_wrap = (depth > 0) & (~led.oom)
            ops = jnp.where(to_wrap, jnp.int32(OP_SUC), jnp.int32(0))
            a1 = jnp.where(to_wrap, child, jnp.int32(0))
            a2 = jnp.zeros_like(a1)
            new_ids, led = intern_fn(led, node_batch_fn(ops, a1, a2))
            child = jnp.where(to_wrap, new_ids, child)
            depth = depth - to_wrap.astype(jnp.int32)
            return depth, child, led

        _, wrap_child, ledger_local = lax.while_loop(
            wrap_cond, wrap_body, (wrap_depth, wrap_child, ledger_local)
        )
        return ledger_local, wrap_child

    return lax.cond(ledger.corrupt, _skip, _do, operand=None)


@lru_cache
def _cycle_intrinsic_jit(intern_fn: InternFn, node_batch_fn: NodeBatchFn):
    @jax.jit
    def _impl(ledger, frontier_ids):
        return _cycle_intrinsic_impl(
            ledger,
            frontier_ids,
            intern_fn=intern_fn,
            node_batch_fn=node_batch_fn,
        )

    return _impl


def cycle_intrinsic(
    ledger,
    frontier_ids,
    *,
    intern_fn: InternFn = intern_nodes,
    intern_cfg: InternConfig | None = None,
    node_batch_fn: NodeBatchFn = _node_batch,
    host_raise_fn: HostRaiseFn = _host_raise_if_bad,
):
    # BSPᵗ: temporal superstep / barrier semantics.
    if intern_cfg is not None and intern_fn is intern_nodes:
        intern_fn = partial(intern_nodes, cfg=intern_cfg)
    ledger, frontier_ids = _cycle_intrinsic_jit(intern_fn, node_batch_fn)(
        ledger, frontier_ids
    )
    host_raise_fn(ledger, "Ledger capacity exceeded during cycle")
    return ledger, frontier_ids


def cycle_intrinsic_cfg(
    ledger,
    frontier_ids,
    *,
    cfg: IntrinsicConfig = DEFAULT_INTRINSIC_CONFIG,
    intern_cfg: InternConfig | None = None,
    intern_fn: InternFn = intern_nodes,
    node_batch_fn: NodeBatchFn = _node_batch,
    host_raise_fn: HostRaiseFn = _host_raise_if_bad,
):
    """Interface/Control wrapper for intrinsic cycle with DI bundle."""
    if cfg.intern_cfg is not None and intern_cfg is not None:
        raise ValueError("Pass either cfg.intern_cfg or intern_cfg, not both.")
    intern_cfg = intern_cfg if intern_cfg is not None else cfg.intern_cfg
    intern_fn = cfg.intern_fn or intern_fn
    node_batch_fn = cfg.node_batch_fn or node_batch_fn
    host_raise_fn = cfg.host_raise_fn or host_raise_fn
    return cycle_intrinsic(
        ledger,
        frontier_ids,
        intern_fn=intern_fn,
        intern_cfg=intern_cfg,
        node_batch_fn=node_batch_fn,
        host_raise_fn=host_raise_fn,
    )


_cycle_intrinsic_jax = _cycle_intrinsic_impl

__all__ = [
    "_cycle_intrinsic_impl",
    "_cycle_intrinsic_jax",
    "cycle_intrinsic",
    "cycle_intrinsic_cfg",
]
