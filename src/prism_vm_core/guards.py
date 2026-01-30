from typing import Callable, Optional

import jax
import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe
from prism_core.guards import (
    GuardConfig,
    guard_gather_index_cfg as _guard_gather_index_cfg,
    safe_index_1d_cfg as _safe_index_1d_cfg,
    safe_gather_1d_cfg as _safe_gather_1d_cfg,
    safe_gather_1d_ok_cfg as _safe_gather_1d_ok_cfg,
    make_safe_gather_fn as _make_safe_gather_fn,
    make_safe_gather_ok_fn as _make_safe_gather_ok_fn,
    make_safe_index_fn as _make_safe_index_fn,
    make_safe_gather_value_fn as _make_safe_gather_value_fn,
    make_safe_gather_ok_value_fn as _make_safe_gather_ok_value_fn,
    make_safe_index_value_fn as _make_safe_index_value_fn,
    resolve_safe_gather_fn as _resolve_safe_gather_fn,
    resolve_safe_gather_ok_fn as _resolve_safe_gather_ok_fn,
    resolve_safe_index_fn as _resolve_safe_index_fn,
    resolve_safe_gather_value_fn as _resolve_safe_gather_value_fn,
    resolve_safe_gather_ok_value_fn as _resolve_safe_gather_ok_value_fn,
    resolve_safe_index_value_fn as _resolve_safe_index_value_fn,
)
from prism_vm_core.ontology import OP_NULL, OP_ZERO

_TEST_GUARDS = _jax_safe.TEST_GUARDS
_HAS_DEBUG_CALLBACK = _jax_safe.HAS_DEBUG_CALLBACK


def _guards_enabled():
    return _TEST_GUARDS and _HAS_DEBUG_CALLBACK


DEFAULT_GUARD_CONFIG = GuardConfig()


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


def guards_enabled_cfg(*, cfg: GuardConfig = DEFAULT_GUARD_CONFIG):
    fn = cfg.guards_enabled_fn or _guards_enabled
    return bool(fn())


def guard_max_cfg(value, max_value, label, *, cfg: GuardConfig = DEFAULT_GUARD_CONFIG):
    fn = cfg.guard_max_fn or _guard_max
    return fn(value, max_value, label)


def guard_gather_index_cfg(
    idx,
    size,
    label,
    *,
    guard=None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    return _guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)


def safe_gather_1d_cfg(
    arr,
    idx,
    label="safe_gather_1d",
    *,
    guard=None,
    policy=None,
    return_ok: bool = False,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    return _safe_gather_1d_cfg(
        arr, idx, label, guard=guard, policy=policy, cfg=cfg, return_ok=return_ok
    )


def safe_gather_1d_ok_cfg(
    arr,
    idx,
    label="safe_gather_1d_ok",
    *,
    guard=None,
    policy=None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    return _safe_gather_1d_ok_cfg(
        arr, idx, label, guard=guard, policy=policy, cfg=cfg
    )


def safe_index_1d_cfg(
    idx,
    size,
    label="safe_index_1d",
    *,
    guard=None,
    policy=None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    return _safe_index_1d_cfg(
        idx, size, label, guard=guard, policy=policy, cfg=cfg
    )

def make_safe_gather_fn(*, cfg: GuardConfig = DEFAULT_GUARD_CONFIG, policy=None, safe_gather_fn=None):
    return _make_safe_gather_fn(cfg=cfg, policy=policy, safe_gather_fn=safe_gather_fn)


def make_safe_gather_ok_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    policy=None,
    safe_gather_ok_fn=None,
):
    return _make_safe_gather_ok_fn(
        cfg=cfg, policy=policy, safe_gather_ok_fn=safe_gather_ok_fn
    )


def make_safe_index_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    policy=None,
    safe_index_fn=None,
):
    return _make_safe_index_fn(
        cfg=cfg, policy=policy, safe_index_fn=safe_index_fn
    )


def make_safe_gather_value_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    safe_gather_value_fn=None,
):
    return _make_safe_gather_value_fn(
        cfg=cfg, safe_gather_value_fn=safe_gather_value_fn
    )


def make_safe_gather_ok_value_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    safe_gather_ok_value_fn=None,
):
    return _make_safe_gather_ok_value_fn(
        cfg=cfg, safe_gather_ok_value_fn=safe_gather_ok_value_fn
    )


def make_safe_index_value_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    safe_index_value_fn=None,
):
    return _make_safe_index_value_fn(
        cfg=cfg, safe_index_value_fn=safe_index_value_fn
    )


def resolve_safe_gather_fn(
    *,
    safe_gather_fn=None,
    policy=None,
    guard_cfg: GuardConfig | None = None,
):
    return _resolve_safe_gather_fn(
        safe_gather_fn=safe_gather_fn, policy=policy, guard_cfg=guard_cfg
    )


def resolve_safe_gather_ok_fn(
    *,
    safe_gather_ok_fn=None,
    policy=None,
    guard_cfg: GuardConfig | None = None,
):
    return _resolve_safe_gather_ok_fn(
        safe_gather_ok_fn=safe_gather_ok_fn,
        policy=policy,
        guard_cfg=guard_cfg,
    )


def resolve_safe_index_fn(
    *,
    safe_index_fn=None,
    policy=None,
    guard_cfg: GuardConfig | None = None,
):
    return _resolve_safe_index_fn(
        safe_index_fn=safe_index_fn, policy=policy, guard_cfg=guard_cfg
    )


def resolve_safe_gather_value_fn(
    *,
    safe_gather_value_fn=None,
    guard_cfg: GuardConfig | None = None,
):
    return _resolve_safe_gather_value_fn(
        safe_gather_value_fn=safe_gather_value_fn, guard_cfg=guard_cfg
    )


def resolve_safe_gather_ok_value_fn(
    *,
    safe_gather_ok_value_fn=None,
    guard_cfg: GuardConfig | None = None,
):
    return _resolve_safe_gather_ok_value_fn(
        safe_gather_ok_value_fn=safe_gather_ok_value_fn, guard_cfg=guard_cfg
    )


def resolve_safe_index_value_fn(
    *,
    safe_index_value_fn=None,
    guard_cfg: GuardConfig | None = None,
):
    return _resolve_safe_index_value_fn(
        safe_index_value_fn=safe_index_value_fn, guard_cfg=guard_cfg
    )


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


def guard_slot0_perm_cfg(perm, inv_perm, label, *, cfg: GuardConfig = DEFAULT_GUARD_CONFIG):
    fn = cfg.guard_slot0_perm_fn or _guard_slot0_perm
    return fn(perm, inv_perm, label)


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


def guard_null_row_cfg(opcode, arg1, arg2, label, *, cfg: GuardConfig = DEFAULT_GUARD_CONFIG):
    fn = cfg.guard_null_row_fn or _guard_null_row
    return fn(opcode, arg1, arg2, label)


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


def guard_zero_row_cfg(opcode, arg1, arg2, label, *, cfg: GuardConfig = DEFAULT_GUARD_CONFIG):
    fn = cfg.guard_zero_row_fn or _guard_zero_row
    return fn(opcode, arg1, arg2, label)


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


def guard_zero_args_cfg(mask, arg1, arg2, label, *, cfg: GuardConfig = DEFAULT_GUARD_CONFIG):
    fn = cfg.guard_zero_args_fn or _guard_zero_args
    return fn(mask, arg1, arg2, label)


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


def guard_swizzle_args_cfg(
    arg1, arg2, live, count, label, *, cfg: GuardConfig = DEFAULT_GUARD_CONFIG
):
    fn = cfg.guard_swizzle_args_fn or _guard_swizzle_args
    return fn(arg1, arg2, live, count, label)


__all__ = [
    "GuardConfig",
    "DEFAULT_GUARD_CONFIG",
    "_guards_enabled",
    "_guard_max",
    "_pop_token",
    "_expect_token",
    "_guard_slot0_perm",
    "_guard_null_row",
    "_guard_zero_row",
    "_guard_zero_args",
    "_guard_swizzle_args",
    "guards_enabled_cfg",
    "guard_max_cfg",
    "guard_gather_index_cfg",
    "safe_index_1d_cfg",
    "safe_gather_1d_cfg",
    "safe_gather_1d_ok_cfg",
    "make_safe_gather_fn",
    "make_safe_gather_ok_fn",
    "make_safe_index_fn",
    "make_safe_gather_value_fn",
    "make_safe_gather_ok_value_fn",
    "make_safe_index_value_fn",
    "resolve_safe_gather_fn",
    "resolve_safe_gather_ok_fn",
    "resolve_safe_index_fn",
    "resolve_safe_gather_value_fn",
    "resolve_safe_gather_ok_value_fn",
    "resolve_safe_index_value_fn",
    "guard_slot0_perm_cfg",
    "guard_null_row_cfg",
    "guard_zero_row_cfg",
    "guard_zero_args_cfg",
    "guard_swizzle_args_cfg",
]
