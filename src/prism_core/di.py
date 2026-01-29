from __future__ import annotations

from functools import lru_cache
from typing import Callable, TypeVar

import jax

T = TypeVar("T")


def resolve(value: T | None, default: T) -> T:
    return default if value is None else value


def wrap_policy(safe_gather_fn, policy):
    """Wrap a safe_gather_fn with a fixed SafetyPolicy (if provided)."""
    if policy is None:
        return safe_gather_fn

    def _safe_gather(arr, idx, label):
        return safe_gather_fn(arr, idx, label, policy=policy)

    return _safe_gather


def cached_factory(factory: Callable[..., T]) -> Callable[..., T]:
    return lru_cache(maxsize=None)(factory)


def cached_jit(factory: Callable[..., Callable], *, static_argnames=None, static_argnums=None):
    """Cache a jitted factory keyed by its (hashable) arguments."""

    @lru_cache(maxsize=None)
    def _cached(*args, **kwargs):
        fn = factory(*args, **kwargs)
        return jax.jit(fn, static_argnames=static_argnames, static_argnums=static_argnums)

    return _cached


__all__ = ["resolve", "wrap_policy", "cached_factory", "cached_jit"]
