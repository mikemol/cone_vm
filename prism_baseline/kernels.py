from jax import jit, lax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_vm_core.ontology import OP_ADD, OP_MUL, OP_SUC, OP_ZERO, ZERO_PTR
from prism_vm_core.structures import Manifest


_scatter_drop = _jax_safe.scatter_drop
_scatter_strict = _jax_safe.scatter_strict


@jit
def kernel_add(manifest, ptr):
    ops, a1, a2, count, oom = (
        manifest.opcode,
        manifest.arg1,
        manifest.arg2,
        manifest.active_count,
        manifest.oom,
    )
    cap = ops.shape[0]
    init_x = a1[ptr]
    init_y = a2[ptr]
    init_val = (init_x, init_y, True, ops, a1, count, oom)

    def cond(v):
        return v[2]

    def body(v):
        curr_x, curr_y, active, b_ops, b_a1, b_count, b_oom = v
        op_x = b_ops[curr_x]
        is_suc = (op_x == OP_SUC) & (~b_oom)
        next_x = jnp.where(is_suc, b_a1[curr_x], curr_x)

        def do_spawn(state):
            ops, a1s, count, y_val, oom = state
            ok = (count < cap) & (~oom)
            w_idx = jnp.where(ok, count, cap)
            ops = _scatter_drop(ops, w_idx, OP_SUC, "kernel_add.ops")
            a1s = _scatter_drop(a1s, w_idx, y_val, "kernel_add.a1s")
            next_count = jnp.where(ok, count + 1, count)
            next_y = jnp.where(ok, w_idx, y_val)
            next_oom = oom | (~ok)
            return ops, a1s, next_count, next_y, next_oom

        def no_spawn(state):
            ops, a1s, count, y_val, oom = state
            return ops, a1s, count, y_val, oom

        b_ops, b_a1, next_count, next_y, next_oom = lax.cond(
            is_suc,
            do_spawn,
            no_spawn,
            (b_ops, b_a1, b_count, curr_y, b_oom),
        )
        return (next_x, next_y, is_suc, b_ops, b_a1, next_count, next_oom)

    _, final_y, _, f_ops, f_a1, f_count, f_oom = lax.while_loop(
        cond, body, init_val
    )
    return (
        manifest._replace(opcode=f_ops, arg1=f_a1, active_count=f_count, oom=f_oom),
        final_y,
    )


@jit
def kernel_mul(manifest, ptr):
    ops, a1, a2, count, oom = (
        manifest.opcode,
        manifest.arg1,
        manifest.arg2,
        manifest.active_count,
        manifest.oom,
    )
    cap = ops.shape[0]
    init_x = a1[ptr]
    y = a2[ptr]
    init_acc = jnp.array(ZERO_PTR, dtype=jnp.int32)
    init_val = (init_x, init_acc, ops, a1, a2, count, oom)

    def cond(v):
        curr_x, _, b_ops, _, _, _, b_oom = v
        return (b_ops[curr_x] == OP_SUC) & (~b_oom)

    def body(v):
        curr_x, acc, b_ops, b_a1, b_a2, b_count, b_oom = v
        next_x = b_a1[curr_x]
        ok = (b_count < cap) & (~b_oom)

        def do_add(state):
            b_ops, b_a1, b_a2, b_count, b_oom, acc = state
            add_idx = b_count
            b_ops = _scatter_strict(b_ops, add_idx, OP_ADD, "kernel_mul.b_ops")
            add_a1_raw = y
            add_a2_raw = acc
            add_swap = add_a2_raw < add_a1_raw
            add_a1 = jnp.where(add_swap, add_a2_raw, add_a1_raw)
            add_a2 = jnp.where(add_swap, add_a1_raw, add_a2_raw)
            b_a1 = _scatter_strict(b_a1, add_idx, add_a1, "kernel_mul.b_a1")
            b_a2 = _scatter_strict(b_a2, add_idx, add_a2, "kernel_mul.b_a2")
            b_count = b_count + 1
            add_manifest = Manifest(b_ops, b_a1, b_a2, b_count, b_oom)
            updated_manifest, next_acc = kernel_add(add_manifest, add_idx)
            return (
                updated_manifest.opcode,
                updated_manifest.arg1,
                updated_manifest.arg2,
                updated_manifest.active_count,
                updated_manifest.oom,
                next_acc,
            )

        def no_add(state):
            b_ops, b_a1, b_a2, b_count, b_oom, acc = state
            return b_ops, b_a1, b_a2, b_count, b_oom | (~ok), acc

        b_ops, b_a1, b_a2, b_count, b_oom, next_acc = lax.cond(
            ok,
            do_add,
            no_add,
            (b_ops, b_a1, b_a2, b_count, b_oom, acc),
        )
        return (next_x, next_acc, b_ops, b_a1, b_a2, b_count, b_oom)

    _, final_acc, f_ops, f_a1, f_a2, f_count, f_oom = lax.while_loop(
        cond, body, init_val
    )
    return (
        manifest._replace(
            opcode=f_ops, arg1=f_a1, arg2=f_a2, active_count=f_count, oom=f_oom
        ),
        final_acc,
    )


def _dispatch_identity(args):
    manifest, ptr = args
    return manifest, ptr


def _dispatch_add(args):
    manifest, ptr = args
    return kernel_add(manifest, ptr)


def _dispatch_mul(args):
    manifest, ptr = args
    return kernel_mul(manifest, ptr)


@jit
def optimize_ptr(manifest, ptr):
    ops, a1s, a2s = manifest.opcode, manifest.arg1, manifest.arg2
    op = ops[ptr]
    a1 = a1s[ptr]
    a2 = a2s[ptr]
    op_a1 = ops[a1]
    op_a2 = ops[a2]
    is_add = op == OP_ADD
    is_mul = op == OP_MUL
    is_zero = op_a1 == OP_ZERO
    is_zero_a2 = op_a2 == OP_ZERO
    add_zero_left = is_add & is_zero
    add_zero_right = is_add & is_zero_a2
    mul_zero_left = is_mul & is_zero
    mul_zero_right = is_mul & is_zero_a2
    out_ptr = jnp.where(add_zero_left, a2, ptr)
    out_ptr = jnp.where(add_zero_right, a1, out_ptr)
    out_ptr = jnp.where(mul_zero_left | mul_zero_right, jnp.int32(ZERO_PTR), out_ptr)
    reason = jnp.where(
        add_zero_left | add_zero_right,
        jnp.int32(1),
        jnp.int32(0),
    )
    reason = jnp.where(mul_zero_left | mul_zero_right, jnp.int32(2), reason)
    return out_ptr, reason


@jit
def dispatch_kernel(manifest, ptr):
    opt_ptr, opt_reason = optimize_ptr(manifest, ptr)
    op = manifest.opcode[opt_ptr]
    case_index = jnp.where(op == OP_ADD, 1, jnp.where(op == OP_MUL, 2, 0))
    new_manifest, res_ptr = lax.switch(
        case_index,
        (_dispatch_identity, _dispatch_add, _dispatch_mul),
        (manifest, opt_ptr),
    )
    return new_manifest, res_ptr, opt_reason


__all__ = [
    "kernel_add",
    "kernel_mul",
    "optimize_ptr",
    "dispatch_kernel",
]
