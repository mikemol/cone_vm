from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from prism_core import jax_safe as _jax_safe


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """Shared guard DI bundle (host-side control surface)."""

    guard_gather_index_fn: Optional[Callable[..., None]] = None


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
    guard_gather_index_cfg(idx, size, label, guard=guard, cfg=cfg)
    return _jax_safe.safe_index_1d(idx, size, label, guard=False, policy=policy)


__all__ = [
    "GuardConfig",
    "DEFAULT_GUARD_CONFIG",
    "guard_gather_index_cfg",
    "safe_index_1d_cfg",
]
