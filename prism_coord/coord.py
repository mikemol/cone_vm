import jax
import jax.numpy as jnp
from jax import lax

from prism_ledger.intern import _coord_norm_id_jax, intern_nodes
from prism_vm_core.domains import _host_int_value
from prism_vm_core.ontology import OP_COORD_ONE, OP_COORD_PAIR, OP_COORD_ZERO
from prism_vm_core.structures import NodeBatch


def _node_batch(op, a1, a2):
    return NodeBatch(op=op, a1=a1, a2=a2)


@jax.jit
def _coord_norm_id_host(ledger, coord_id):
    return _coord_norm_id_jax(ledger, coord_id)


def _coord_leaf_id(ledger, op):
    ids, ledger = intern_nodes(
        ledger,
        _node_batch(
            jnp.array([op], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
        ),
    )
    # SYNC: host reads device id for coord leaf (m1).
    return _host_int_value(ids[0]), ledger


def _coord_promote_leaf(ledger, leaf_id):
    zero_id, ledger = _coord_leaf_id(ledger, OP_COORD_ZERO)
    ids, ledger = intern_nodes(
        ledger,
        _node_batch(
            jnp.array([OP_COORD_PAIR], dtype=jnp.int32),
            jnp.array([leaf_id], dtype=jnp.int32),
            jnp.array([zero_id], dtype=jnp.int32),
        ),
    )
    # SYNC: host reads device id for coord promotion (m1).
    return _host_int_value(ids[0]), ledger


def _coord_cache_init(ledger):
    count = _host_int_value(ledger.count)
    ops = list(jax.device_get(ledger.opcode[:count]))
    a1s = list(jax.device_get(ledger.arg1[:count]))
    a2s = list(jax.device_get(ledger.arg2[:count]))
    return ops, a1s, a2s, count


def _coord_cache_update(cache, idx, op, a1, a2):
    ops, a1s, a2s, count = cache
    if idx == count:
        ops.append(int(op))
        a1s.append(int(a1))
        a2s.append(int(a2))
        count += 1
    return ops, a1s, a2s, count


def _coord_leaf_id_cached(ledger, op, cache, leaf_cache):
    cached = leaf_cache.get(int(op))
    if cached is not None:
        return cached, ledger, cache
    ids, ledger = intern_nodes(
        ledger,
        _node_batch(
            jnp.array([op], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
        ),
    )
    new_id = _host_int_value(ids[0])
    cache = _coord_cache_update(cache, new_id, op, 0, 0)
    leaf_cache[int(op)] = new_id
    return new_id, ledger, cache


def _coord_promote_leaf_cached(ledger, leaf_id, cache, leaf_cache):
    zero_id, ledger, cache = _coord_leaf_id_cached(
        ledger, OP_COORD_ZERO, cache, leaf_cache
    )
    ids, ledger = intern_nodes(
        ledger,
        _node_batch(
            jnp.array([OP_COORD_PAIR], dtype=jnp.int32),
            jnp.array([leaf_id], dtype=jnp.int32),
            jnp.array([zero_id], dtype=jnp.int32),
        ),
    )
    new_id = _host_int_value(ids[0])
    cache = _coord_cache_update(cache, new_id, OP_COORD_PAIR, leaf_id, zero_id)
    return new_id, ledger, cache


def coord_norm(ledger, coord_id):
    # Host-only entrypoint; route through the device oracle for consistency.
    coord_id = jnp.asarray(coord_id, dtype=jnp.int32)
    norm_id = _coord_norm_id_host(ledger, coord_id)
    # SYNC: host reads device scalar for coord normalization (m1).
    return _host_int_value(norm_id), ledger


def coord_xor(ledger, left_id, right_id):
    # CDᵣ + Normalize𝚌
    # COMMUTES: CDₐ ⟂ CDᵣ    [test: tests/test_coord_ops.py::test_coord_norm_commutes_with_xor]
    # Host-only reference path; device hot paths should use jitted batch ops.
    # SYNC: host reads device opcode/args for coord xor (m1).
    # NOTE: per-scalar device reads can be a perf cliff; batch if needed
    # (see IMPLEMENTATION_PLAN.md).
    left_id = int(left_id)
    right_id = int(right_id)
    cache = _coord_cache_init(ledger)
    leaf_cache = {}

    def _get_node(idx, cache):
        ops, a1s, a2s, _ = cache
        return ops[idx], a1s[idx], a2s[idx]

    def _coord_xor_cached(ledger, left_id, right_id, cache, leaf_cache):
        if left_id == right_id:
            zero_id, ledger, cache = _coord_leaf_id_cached(
                ledger, OP_COORD_ZERO, cache, leaf_cache
            )
            return zero_id, ledger, cache, leaf_cache

        left_op, left_a1, left_a2 = _get_node(left_id, cache)
        right_op, right_a1, right_a2 = _get_node(right_id, cache)

        if left_op == OP_COORD_ZERO:
            return right_id, ledger, cache, leaf_cache
        if right_op == OP_COORD_ZERO:
            return left_id, ledger, cache, leaf_cache

        if left_op in (OP_COORD_ZERO, OP_COORD_ONE) and right_op in (
            OP_COORD_ZERO,
            OP_COORD_ONE,
        ):
            if left_op == right_op:
                zero_id, ledger, cache = _coord_leaf_id_cached(
                    ledger, OP_COORD_ZERO, cache, leaf_cache
                )
                return zero_id, ledger, cache, leaf_cache
            one_id, ledger, cache = _coord_leaf_id_cached(
                ledger, OP_COORD_ONE, cache, leaf_cache
            )
            return one_id, ledger, cache, leaf_cache

        if left_op != OP_COORD_PAIR:
            left_id, ledger, cache = _coord_promote_leaf_cached(
                ledger, left_id, cache, leaf_cache
            )
            left_op, left_a1, left_a2 = _get_node(left_id, cache)
        if right_op != OP_COORD_PAIR:
            right_id, ledger, cache = _coord_promote_leaf_cached(
                ledger, right_id, cache, leaf_cache
            )
            right_op, right_a1, right_a2 = _get_node(right_id, cache)

        new_left, ledger, cache, leaf_cache = _coord_xor_cached(
            ledger, left_a1, right_a1, cache, leaf_cache
        )
        new_right, ledger, cache, leaf_cache = _coord_xor_cached(
            ledger, left_a2, right_a2, cache, leaf_cache
        )
        ids, ledger = intern_nodes(
            ledger,
            _node_batch(
                jnp.array([OP_COORD_PAIR], dtype=jnp.int32),
                jnp.array([new_left], dtype=jnp.int32),
                jnp.array([new_right], dtype=jnp.int32),
            ),
        )
        new_id = _host_int_value(ids[0])
        cache = _coord_cache_update(cache, new_id, OP_COORD_PAIR, new_left, new_right)
        return new_id, ledger, cache, leaf_cache

    out_id, ledger, _, _ = _coord_xor_cached(
        ledger, left_id, right_id, cache, leaf_cache
    )
    return out_id, ledger


def cd_lift_binary(ledger, op, left_id, right_id):
    """Host-only CD lift of a binary op across coord pairs.

    If both inputs are OP_COORD_PAIR, apply op componentwise and re-pair.
    Otherwise, fall back to op(left, right).
    """
    left_id = int(left_id)
    right_id = int(right_id)
    left_op = _host_int_value(ledger.opcode[left_id])
    right_op = _host_int_value(ledger.opcode[right_id])
    if left_op == OP_COORD_PAIR and right_op == OP_COORD_PAIR:
        left_a1 = _host_int_value(ledger.arg1[left_id])
        left_a2 = _host_int_value(ledger.arg2[left_id])
        right_a1 = _host_int_value(ledger.arg1[right_id])
        right_a2 = _host_int_value(ledger.arg2[right_id])
        new_left, ledger = cd_lift_binary(ledger, op, left_a1, right_a1)
        new_right, ledger = cd_lift_binary(ledger, op, left_a2, right_a2)
        ids, ledger = intern_nodes(
            ledger,
            _node_batch(
                jnp.array([OP_COORD_PAIR], dtype=jnp.int32),
                jnp.array([new_left], dtype=jnp.int32),
                jnp.array([new_right], dtype=jnp.int32),
            ),
        )
        return _host_int_value(ids[0]), ledger
    ids, ledger = intern_nodes(
        ledger,
        _node_batch(
            jnp.array([op], dtype=jnp.int32),
            jnp.array([left_id], dtype=jnp.int32),
            jnp.array([right_id], dtype=jnp.int32),
        ),
    )
    return _host_int_value(ids[0]), ledger


@jax.jit
def _coord_norm_batch_jax(ledger, coord_ids):
    return jax.vmap(_coord_norm_id_jax, in_axes=(None, 0))(ledger, coord_ids)


def coord_norm_batch(ledger, coord_ids):
    coord_ids = jnp.asarray(coord_ids, dtype=jnp.int32)
    if coord_ids.size == 0:
        return coord_ids, ledger
    norm_ids = _coord_norm_batch_jax(ledger, coord_ids)
    return norm_ids, ledger


def coord_xor_batch(ledger, left_ids, right_ids):
    left_ids = jnp.asarray(left_ids, dtype=jnp.int32)
    right_ids = jnp.asarray(right_ids, dtype=jnp.int32)
    if left_ids.shape != right_ids.shape:
        raise ValueError("coord_xor_batch expects aligned id arrays")
    if left_ids.size == 0:
        return left_ids, ledger
    left_ops = jax.device_get(ledger.opcode[left_ids])
    right_ops = jax.device_get(ledger.opcode[right_ids])
    leaf_mask = (
        ((left_ops == OP_COORD_ZERO) | (left_ops == OP_COORD_ONE))
        & ((right_ops == OP_COORD_ZERO) | (right_ops == OP_COORD_ONE))
    )
    if bool(leaf_mask.all()):
        zero_id, ledger = _coord_leaf_id(ledger, OP_COORD_ZERO)
        one_id, ledger = _coord_leaf_id(ledger, OP_COORD_ONE)
        left_bits = left_ops == OP_COORD_ONE
        right_bits = right_ops == OP_COORD_ONE
        out_bits = left_bits ^ right_bits
        out_ids = jnp.where(out_bits, one_id, zero_id).astype(jnp.int32)
        return out_ids, ledger
    out_ids = []
    for left_id, right_id in zip(left_ids, right_ids):
        out_id, ledger = coord_xor(ledger, int(left_id), int(right_id))
        out_ids.append(out_id)
    return jnp.array(out_ids, dtype=jnp.int32), ledger


__all__ = [
    "coord_norm",
    "coord_norm_batch",
    "coord_xor",
    "coord_xor_batch",
    "cd_lift_binary",
]
