from __future__ import annotations

from functools import lru_cache
from inspect import Parameter, signature
from typing import Callable, TypeVar

import jax

T = TypeVar("T")


def resolve(value: T | None, default: T) -> T:
    return default if value is None else value


def wrap_policy(safe_gather_fn, policy):
    """Wrap a safe_gather_fn with a fixed SafetyPolicy (if provided)."""
    if policy is None:
        return safe_gather_fn
    wrapped = bind_optional_kwargs(safe_gather_fn, policy=policy)

    def _safe_gather(arr, idx, label, **kwargs):
        return wrapped(arr, idx, label, **kwargs)

    return _safe_gather


def wrap_index_policy(safe_index_fn, policy):
    """Wrap a safe_index_fn with a fixed SafetyPolicy (if provided)."""
    if policy is None:
        return safe_index_fn
    wrapped = bind_optional_kwargs(safe_index_fn, policy=policy)

    def _safe_index(idx, size, label, **kwargs):
        return wrapped(idx, size, label, **kwargs)

    return _safe_index


def cached_factory(factory: Callable[..., T]) -> Callable[..., T]:
    return lru_cache(maxsize=None)(factory)


def cached_jit(factory: Callable[..., Callable], *, static_argnames=None, static_argnums=None):
    """Cache a jitted factory keyed by its (hashable) arguments."""

    @lru_cache(maxsize=None)
    def _cached(*args, **kwargs):
        fn = factory(*args, **kwargs)
        return jax.jit(fn, static_argnames=static_argnames, static_argnums=static_argnums)

    return _cached


def call_with_optional_kw(fn: Callable[..., T], name: str, value, *args, **kwargs) -> T:
    """Call fn with an optional keyword if it is accepted.

    This is a host-side helper to keep DI-compatible call sites tolerant of
    callables that have not yet adopted a new keyword parameter.
    """
    return call_with_optional_kwargs(fn, {name: value}, *args, **kwargs)


def call_with_optional_kwargs(
    fn: Callable[..., T], optional: dict, *args, **kwargs
) -> T:
    """Call fn with optional keyword arguments filtered to accepted names."""
    optional = {k: v for k, v in optional.items() if v is not None}
    if not optional:
        return fn(*args, **kwargs)
    try:
        sig = signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs, **optional)
    if any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs, **optional)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in optional.items() if k in allowed}
    return fn(*args, **kwargs, **filtered)


def bind_optional_kwargs(fn: Callable[..., T], **optional):
    """Return fn partially applied with optional kwargs it accepts."""
    optional = {k: v for k, v in optional.items() if v is not None}
    if not optional:
        return fn
    try:
        sig = signature(fn)
    except (TypeError, ValueError):
        return lambda *args, **kwargs: fn(*args, **kwargs, **optional)
    if any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return lambda *args, **kwargs: fn(*args, **kwargs, **optional)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in optional.items() if k in allowed}
    if not filtered:
        return fn
    return lambda *args, **kwargs: fn(*args, **kwargs, **filtered)


__all__ = [
    "resolve",
    "wrap_policy",
    "wrap_index_policy",
    "cached_factory",
    "cached_jit",
    "call_with_optional_kw",
    "call_with_optional_kwargs",
    "bind_optional_kwargs",
]
