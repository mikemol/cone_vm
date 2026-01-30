from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from prism_core.di import call_with_optional_kwargs, wrap_index_policy, wrap_policy
from prism_core import jax_safe as _jax_safe
from prism_core.protocols import (
    SafeGatherOkValueFn,
    SafeGatherValueFn,
    SafeIndexValueFn,
)


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """Shared guard DI bundle (host-side control surface)."""

    guards_enabled_fn: Optional[Callable[[], bool]] = None
    guard_max_fn: Optional[Callable[..., None]] = None
    guard_gather_index_fn: Optional[Callable[..., None]] = None
    safe_index_fn: Optional[Callable[..., tuple]] = None
    guard_slot0_perm_fn: Optional[Callable[..., None]] = None
    guard_null_row_fn: Optional[Callable[..., None]] = None
    guard_zero_row_fn: Optional[Callable[..., None]] = None
    guard_zero_args_fn: Optional[Callable[..., None]] = None
    guard_swizzle_args_fn: Optional[Callable[..., None]] = None


DEFAULT_GUARD_CONFIG = GuardConfig()


def guard_gather_index_cfg(
    idx,
    size,
    label,
    *,
    guard=None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    fn = cfg.guard_gather_index_fn or _jax_safe.guard_gather_index
    return call_with_optional_kwargs(
        fn, {"guard": guard}, idx, size, label
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
    """Guard-configured wrapper for safe_index_1d (shared)."""
    if cfg.safe_index_fn is not None:
        return call_with_optional_kwargs(
            cfg.safe_index_fn,
            {"guard": guard, "policy": policy},
            idx,
            size,
            label,
        )
    guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)
    return _jax_safe.safe_index_1d(idx, size, label, guard=False, policy=policy)


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
    """Guard-configured wrapper for safe_gather_1d (shared)."""
    size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
    guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)
    return _jax_safe.safe_gather_1d(
        arr, idx, label, guard=False, policy=policy, return_ok=return_ok
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
    """Guard-configured wrapper for safe_gather_1d_ok (shared)."""
    size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
    guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)
    return _jax_safe.safe_gather_1d_ok(
        arr, idx, label, guard=False, policy=policy
    )


def make_safe_gather_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    policy=None,
    safe_gather_fn=None,
):
    """Return a SafeGatherFn wired to the provided GuardConfig."""
    if safe_gather_fn is None:
        safe_gather_fn = _jax_safe.safe_gather_1d

    def _safe_gather(arr, idx, label, *, policy=policy, return_ok: bool = False):
        size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
        guard_gather_index_cfg(idx, size, label, cfg=cfg)
        return call_with_optional_kwargs(
            safe_gather_fn,
            {"guard": False, "policy": policy, "return_ok": return_ok},
            arr,
            idx,
            label,
        )

    return _safe_gather


def make_safe_gather_ok_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    policy=None,
    safe_gather_ok_fn=None,
):
    """Return a SafeGatherOkFn wired to the provided GuardConfig."""
    if safe_gather_ok_fn is None:
        safe_gather_ok_fn = _jax_safe.safe_gather_1d_ok

    def _safe_gather_ok(arr, idx, label, *, policy=policy):
        size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
        guard_gather_index_cfg(idx, size, label, cfg=cfg)
        return call_with_optional_kwargs(
            safe_gather_ok_fn,
            {"guard": False, "policy": policy},
            arr,
            idx,
            label,
        )

    return _safe_gather_ok


def make_safe_index_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    policy=None,
    safe_index_fn=None,
):
    """Return a SafeIndexFn wired to the provided GuardConfig."""
    if safe_index_fn is None:
        safe_index_fn = cfg.safe_index_fn or _jax_safe.safe_index_1d

    def _safe_index(idx, size, label, *, policy=policy):
        guard_gather_index_cfg(idx, size, label, cfg=cfg)
        return call_with_optional_kwargs(
            safe_index_fn,
            {"guard": False, "policy": policy},
            idx,
            size,
            label,
        )

    return _safe_index


def make_safe_gather_value_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    safe_gather_value_fn: SafeGatherValueFn | None = None,
):
    """Return a SafeGatherValueFn wired to the provided GuardConfig."""
    if safe_gather_value_fn is None:
        safe_gather_value_fn = _jax_safe.safe_gather_1d_value

    def _safe_gather(arr, idx, label, *, policy_value):
        size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
        guard_gather_index_cfg(idx, size, label, cfg=cfg)
        return call_with_optional_kwargs(
            safe_gather_value_fn,
            {"guard": False},
            arr,
            idx,
            label,
            policy_value=policy_value,
        )

    return _safe_gather


def make_safe_gather_ok_value_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    safe_gather_ok_value_fn: SafeGatherOkValueFn | None = None,
):
    """Return a SafeGatherOkValueFn wired to the provided GuardConfig."""
    if safe_gather_ok_value_fn is None:
        safe_gather_ok_value_fn = _jax_safe.safe_gather_1d_ok_value

    def _safe_gather_ok(arr, idx, label, *, policy_value):
        size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
        guard_gather_index_cfg(idx, size, label, cfg=cfg)
        return call_with_optional_kwargs(
            safe_gather_ok_value_fn,
            {"guard": False},
            arr,
            idx,
            label,
            policy_value=policy_value,
        )

    return _safe_gather_ok


def make_safe_index_value_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    safe_index_value_fn: SafeIndexValueFn | None = None,
):
    """Return a SafeIndexValueFn wired to the provided GuardConfig."""
    if safe_index_value_fn is None:
        safe_index_value_fn = _jax_safe.safe_index_1d_value

    def _safe_index(idx, size, label, *, policy_value):
        guard_gather_index_cfg(idx, size, label, cfg=cfg)
        return call_with_optional_kwargs(
            safe_index_value_fn,
            {"guard": False},
            idx,
            size,
            label,
            policy_value=policy_value,
        )

    return _safe_index


def resolve_safe_gather_fn(
    *,
    safe_gather_fn=None,
    policy=None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a SafeGatherFn with optional SafetyPolicy + GuardConfig wiring."""
    if safe_gather_fn is None:
        safe_gather_fn = _jax_safe.safe_gather_1d
    if guard_cfg is not None:
        wrapped = make_safe_gather_fn(
            cfg=guard_cfg, policy=policy, safe_gather_fn=safe_gather_fn
        )
    else:
        wrapped = wrap_policy(safe_gather_fn, policy)
    try:
        setattr(wrapped, "_prism_policy_bound", policy is not None)
    except Exception:
        pass
    return wrapped


def resolve_safe_gather_ok_fn(
    *,
    safe_gather_ok_fn=None,
    policy=None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a SafeGatherOkFn with optional SafetyPolicy + GuardConfig wiring."""
    if safe_gather_ok_fn is None:
        safe_gather_ok_fn = _jax_safe.safe_gather_1d_ok
    if guard_cfg is not None:
        wrapped = make_safe_gather_ok_fn(
            cfg=guard_cfg, policy=policy, safe_gather_ok_fn=safe_gather_ok_fn
        )
    else:
        wrapped = wrap_policy(safe_gather_ok_fn, policy)
    try:
        setattr(wrapped, "_prism_policy_bound", policy is not None)
    except Exception:
        pass
    return wrapped


def resolve_safe_index_fn(
    *,
    safe_index_fn=None,
    policy=None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a SafeIndexFn with optional SafetyPolicy + GuardConfig wiring."""
    if safe_index_fn is None:
        safe_index_fn = _jax_safe.safe_index_1d
    if guard_cfg is not None:
        wrapped = make_safe_index_fn(
            cfg=guard_cfg, policy=policy, safe_index_fn=safe_index_fn
        )
    else:
        wrapped = wrap_index_policy(safe_index_fn, policy)
    try:
        setattr(wrapped, "_prism_policy_bound", policy is not None)
    except Exception:
        pass
    return wrapped


def resolve_safe_gather_value_fn(
    *,
    safe_gather_value_fn: SafeGatherValueFn | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a SafeGatherValueFn with optional GuardConfig wiring."""
    if safe_gather_value_fn is None:
        safe_gather_value_fn = _jax_safe.safe_gather_1d_value
    if guard_cfg is not None:
        return make_safe_gather_value_fn(
            cfg=guard_cfg, safe_gather_value_fn=safe_gather_value_fn
        )
    return safe_gather_value_fn


def resolve_safe_gather_ok_value_fn(
    *,
    safe_gather_ok_value_fn: SafeGatherOkValueFn | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a SafeGatherOkValueFn with optional GuardConfig wiring."""
    if safe_gather_ok_value_fn is None:
        safe_gather_ok_value_fn = _jax_safe.safe_gather_1d_ok_value
    if guard_cfg is not None:
        return make_safe_gather_ok_value_fn(
            cfg=guard_cfg, safe_gather_ok_value_fn=safe_gather_ok_value_fn
        )
    return safe_gather_ok_value_fn


def resolve_safe_index_value_fn(
    *,
    safe_index_value_fn: SafeIndexValueFn | None = None,
    guard_cfg: GuardConfig | None = None,
):
    """Return a SafeIndexValueFn with optional GuardConfig wiring."""
    if safe_index_value_fn is None:
        safe_index_value_fn = _jax_safe.safe_index_1d_value
    if guard_cfg is not None:
        return make_safe_index_value_fn(
            cfg=guard_cfg, safe_index_value_fn=safe_index_value_fn
        )
    return safe_index_value_fn


__all__ = [
    "GuardConfig",
    "DEFAULT_GUARD_CONFIG",
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
]
