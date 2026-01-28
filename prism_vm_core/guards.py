import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_vm_core.ontology import OP_NULL, OP_ZERO

_TEST_GUARDS = _jax_safe.TEST_GUARDS
_HAS_DEBUG_CALLBACK = _jax_safe.HAS_DEBUG_CALLBACK


def _guards_enabled():
    return _TEST_GUARDS and _HAS_DEBUG_CALLBACK


def _guard_max(value, max_value, label):
    if not _guards_enabled():
        return
    bad = value > max_value

    def _raise(bad_val, val, max_val):
        if bad_val:
            raise RuntimeError(
                f"guard failed: {label} {int(val)} > {int(max_val)}"
            )

    jax.debug.callback(_raise, bad, value, max_value)


def _pop_token(tokens):
    if not tokens:
        raise ValueError("Unexpected end of input")
    return tokens.pop(0)


def _expect_token(tokens, expected):
    token = _pop_token(tokens)
    if token != expected:
        raise ValueError(f"Expected {expected!r}, got {token!r}")
    return token


def _guard_slot0_perm(perm, inv_perm, label):
    if not _guards_enabled():
        return
    p0 = perm[0]
    i0 = inv_perm[0]
    ok = (p0 == 0) & (i0 == 0)

    def _raise(ok_val, p0_val, i0_val):
        if not ok_val:
            raise RuntimeError(
                f"guard failed: {label} perm[0]={int(p0_val)} inv_perm[0]={int(i0_val)}"
            )

    jax.debug.callback(_raise, ok, p0, i0)


def _guard_null_row(opcode, arg1, arg2, label):
    if not _guards_enabled():
        return
    op0 = opcode[0]
    a10 = arg1[0]
    a20 = arg2[0]
    ok = (op0 == OP_NULL) & (a10 == 0) & (a20 == 0)

    def _raise(ok_val, op0_val, a10_val, a20_val):
        if not ok_val:
            raise RuntimeError(
                f"guard failed: {label} op0={int(op0_val)} a1={int(a10_val)} a2={int(a20_val)}"
            )

    jax.debug.callback(_raise, ok, op0, a10, a20)


def _guard_zero_row(opcode, arg1, arg2, label):
    if not _guards_enabled():
        return
    op1 = opcode[1]
    a11 = arg1[1]
    a21 = arg2[1]
    ok = (op1 == OP_ZERO) & (a11 == 0) & (a21 == 0)

    def _raise(ok_val, op1_val, a11_val, a21_val):
        if not ok_val:
            raise RuntimeError(
                f"guard failed: {label} op1={int(op1_val)} a1={int(a11_val)} a2={int(a21_val)}"
            )

    jax.debug.callback(_raise, ok, op1, a11, a21)


def _guard_zero_args(mask, arg1, arg2, label):
    if not _guards_enabled():
        return
    if mask.size == 0:
        return
    bad = jnp.any(mask & ((arg1 != 0) | (arg2 != 0)))

    def _raise(bad_val):
        if bad_val:
            raise RuntimeError(f"guard failed: {label} expected zero args")

    jax.debug.callback(_raise, bad)


def _guard_swizzle_args(arg1, arg2, live, count, label):
    if not _guards_enabled():
        return
    if live.size == 0:
        return
    count_i = jnp.asarray(count, dtype=jnp.int32)
    bad1 = live & (arg1 != 0) & ((arg1 < 0) | (arg1 >= count_i))
    bad2 = live & (arg2 != 0) & ((arg2 < 0) | (arg2 >= count_i))
    bad = jnp.any(bad1 | bad2)

    def _raise(bad_val, count_val):
        if bad_val:
            raise RuntimeError(
                f"guard failed: {label} args out of bounds (count={int(count_val)})"
            )

    jax.debug.callback(_raise, bad, count_i)


__all__ = [
    "_guards_enabled",
    "_guard_max",
    "_pop_token",
    "_expect_token",
    "_guard_slot0_perm",
    "_guard_null_row",
    "_guard_zero_row",
    "_guard_zero_args",
    "_guard_swizzle_args",
]
