from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from prism_core import jax_safe as _jax_safe


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
    return fn(idx, size, label, guard=guard)


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
        return cfg.safe_index_fn(idx, size, label, guard=guard, policy=policy)
    guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)
    return _jax_safe.safe_index_1d(idx, size, label, guard=False, policy=policy)


def safe_gather_1d_cfg(
    arr,
    idx,
    label="safe_gather_1d",
    *,
    guard=None,
    policy=None,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
):
    """Guard-configured wrapper for safe_gather_1d (shared)."""
    size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
    guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)
    return _jax_safe.safe_gather_1d(
        arr, idx, label, guard=False, policy=policy
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


def make_safe_index_fn(
    *,
    cfg: GuardConfig = DEFAULT_GUARD_CONFIG,
    policy=None,
):
    """Return a SafeIndexFn wired to the provided GuardConfig."""
    if cfg.safe_index_fn is not None:
        def _safe_index(idx, size, label, *, policy=policy):
            return cfg.safe_index_fn(idx, size, label, guard=None, policy=policy)
    else:
        def _safe_index(idx, size, label, *, policy=policy):
            return safe_index_1d_cfg(idx, size, label, policy=policy, cfg=cfg)

    return _safe_index


__all__ = [
    "GuardConfig",
    "DEFAULT_GUARD_CONFIG",
    "guard_gather_index_cfg",
    "safe_index_1d_cfg",
    "safe_gather_1d_cfg",
    "safe_gather_1d_ok_cfg",
    "make_safe_index_fn",
]
