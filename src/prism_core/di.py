from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, signature
from typing import Callable, TypeVar

import jax

from prism_core.errors import PrismPolicyBindingError

# dataflow-bundle: args, kwargs
# dataflow-bundle: args, kwargs, name, value
# dataflow-bundle: arr, idx
# dataflow-bundle: idx, size

T = TypeVar("T")
TCallable = TypeVar("TCallable", bound=Callable[..., object])


@dataclass(frozen=True)
class _CallOptionalKwargsArgs:
    args: tuple
    kwargs: dict


@dataclass(frozen=True)
class _CallOptionalKwArgs:
    name: str
    value: object
    args: tuple
    kwargs: dict


@dataclass(frozen=True)
class _ArrayIndexArgs:
    arr: object
    idx: object


@dataclass(frozen=True)
class _IndexSizeArgs:
    idx: object
    size: object


def resolve(value: T | None, default: T) -> T:
    return default if value is None else value


def wrap_policy(safe_gather_fn: TCallable, policy) -> TCallable:
    """Wrap a safe_gather_fn with a fixed SafetyPolicy (if provided)."""
    if policy is None:
        return safe_gather_fn
    if getattr(safe_gather_fn, "_prism_policy_bound", False):
        raise PrismPolicyBindingError(
            "policy already bound in wrap_policy; remove duplicate",
            context="wrap_policy",
            policy_mode="static",
        )
    wrapped = bind_optional_kwargs(safe_gather_fn, policy=policy)

    def _safe_gather(arr, idx, label, *, guard=None, return_ok=None):
        bundle = _ArrayIndexArgs(arr=arr, idx=idx)
        optional = {"guard": guard}
        if return_ok is not None:
            optional["return_ok"] = return_ok
        return call_with_optional_kwargs(
            wrapped, optional, bundle.arr, bundle.idx, label
        )

    try:
        setattr(_safe_gather, "_prism_policy_bound", True)
    except Exception:
        pass

    return _safe_gather


def wrap_index_policy(safe_index_fn: TCallable, policy) -> TCallable:
    """Wrap a safe_index_fn with a fixed SafetyPolicy (if provided)."""
    if policy is None:
        return safe_index_fn
    if getattr(safe_index_fn, "_prism_policy_bound", False):
        raise PrismPolicyBindingError(
            "policy already bound in wrap_index_policy; remove duplicate",
            context="wrap_index_policy",
            policy_mode="static",
        )
    wrapped = bind_optional_kwargs(safe_index_fn, policy=policy)

    def _safe_index(idx, size, label, *, guard=None):
        bundle = _IndexSizeArgs(idx=idx, size=size)
        return call_with_optional_kwargs(
            wrapped, {"guard": guard}, bundle.idx, bundle.size, label
        )

    try:
        setattr(_safe_index, "_prism_policy_bound", True)
    except Exception:
        pass

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
    bundle = _CallOptionalKwArgs(name=name, value=value, args=args, kwargs=kwargs)
    return call_with_optional_kwargs(
        fn, {bundle.name: bundle.value}, *bundle.args, **bundle.kwargs
    )


def call_with_optional_kwargs(
    fn: Callable[..., T], optional: dict, *args, **kwargs
) -> T:
    """Call fn with optional keyword arguments filtered to accepted names."""
    bundle = _CallOptionalKwargsArgs(args=args, kwargs=kwargs)
    optional = {k: v for k, v in optional.items() if v is not None}
    if not optional:
        return fn(*bundle.args, **bundle.kwargs)
    try:
        sig = signature(fn)
    except (TypeError, ValueError):
        return fn(*bundle.args, **bundle.kwargs, **optional)
    if any(p.kind == Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*bundle.args, **bundle.kwargs, **optional)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in optional.items() if k in allowed}
    return fn(*bundle.args, **bundle.kwargs, **filtered)


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
