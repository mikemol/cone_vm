import os
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from prism_core.safety import (
    DEFAULT_SAFETY_POLICY,
    POLICY_VALUE_CLAMP,
    POLICY_VALUE_CORRUPT,
    POLICY_VALUE_DROP,
    PolicyValue,
    SafetyPolicy,
    SafetyMode,
    oob_mask,
)

# dataflow-bundle: idx, policy_value
# dataflow-bundle: idx, policy_value, size
# dataflow-bundle: idx, size
# dataflow-bundle: max_allowed, max_val, min_val
# dataflow-bundle: max_val, min_val, size_val

TEST_GUARDS = os.environ.get("PRISM_TEST_GUARDS", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SCATTER_GUARD = TEST_GUARDS or os.environ.get(
    "PRISM_SCATTER_GUARD", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
GATHER_GUARD = TEST_GUARDS or os.environ.get(
    "PRISM_GATHER_GUARD", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
HAS_DEBUG_CALLBACK = hasattr(jax, "debug") and hasattr(jax.debug, "callback")


@dataclass(frozen=True)
class _IndexPolicyValue:
    idx: object
    policy_value: PolicyValue


@dataclass(frozen=True)
class _IndexPolicyValueSize:
    idx: object
    policy_value: PolicyValue
    size: object


@dataclass(frozen=True)
class _IndexSizeArgs:
    idx: object
    size: object


@dataclass(frozen=True)
class _BoundsArgs:
    max_val: object
    min_val: object
    size_val: object


def scatter_guard(indices, max_index, label):
    if not SCATTER_GUARD or not HAS_DEBUG_CALLBACK:
        return
    if indices.size == 0:
        return
    min_idx = jnp.min(indices)
    max_idx = jnp.max(indices)
    # Allow sentinel index == max_index for intentional drop semantics.
    bad = (min_idx < 0) | (max_idx > max_index)

    def _raise(bad_val, min_val, max_val, max_allowed):
        if bad_val:
            raise RuntimeError(
                f"scatter index out of bounds in {label} "
                f"(min={int(min_val)}, max={int(max_val)}, max={int(max_allowed)})"
            )

    jax.debug.callback(_raise, bad, min_idx, max_idx, max_index)


def scatter_guard_strict(indices, max_index, label):
    if not SCATTER_GUARD or not HAS_DEBUG_CALLBACK:
        return
    if indices.size == 0:
        return
    min_idx = jnp.min(indices)
    max_idx = jnp.max(indices)
    bad = (min_idx < 0) | (max_idx >= max_index)

    def _raise(bad_val, min_val, max_val, max_allowed):
        if bad_val:
            raise RuntimeError(
                f"scatter index out of bounds in {label} "
                f"(min={int(min_val)}, max={int(max_val)}, max={int(max_allowed)})"
            )

    jax.debug.callback(_raise, bad, min_idx, max_idx, max_index)


def scatter_drop(target, indices, values, label):
    max_index = jnp.asarray(target.shape[0], dtype=jnp.int32)
    scatter_guard(indices, max_index, label)
    # NOTE: drop semantics allow sentinel indices for masked scatters.
    return target.at[indices].set(values, mode="drop")


def scatter_strict(target, indices, values, label):
    max_index = jnp.asarray(target.shape[0], dtype=jnp.int32)
    scatter_guard_strict(indices, max_index, label)
    return target.at[indices].set(values, mode="promise_in_bounds")


def guard_gather_index(idx, size, label, guard=None):
    if guard is None:
        guard = GATHER_GUARD
    if not guard or not HAS_DEBUG_CALLBACK:
        return
    if idx.size == 0:
        return
    min_idx = jnp.min(idx)
    max_idx = jnp.max(idx)
    bad = (min_idx < 0) | (max_idx >= size)

    def _raise(bad_val, min_val, max_val, size_val):
        if bad_val:
            raise RuntimeError(
                "gather index out of bounds in "
                f"{label} (min={int(min_val)}, max={int(max_val)}, size={int(size_val)})"
            )

    bounds = _BoundsArgs(max_val=max_idx, min_val=min_idx, size_val=size)
    jax.debug.callback(
        _raise, bad, bounds.min_val, bounds.max_val, bounds.size_val
    )


# dataflow-bundle: arr, idx, label, policy, return_ok
def safe_gather_1d(
    arr,
    idx,
    label="safe_gather_1d",
    guard=None,
    *,
    policy: SafetyPolicy | None = None,
    return_ok: bool = False,
):
    """Guarded gather with explicit SafetyPolicy handling."""
    if policy is None:
        policy = DEFAULT_SAFETY_POLICY
    size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
    idx_i = jnp.asarray(idx, dtype=jnp.int32)
    guard_gather_index(idx_i, size, label, guard=guard)
    ok = (idx_i >= 0) & (idx_i < size)
    idx_safe = jnp.clip(idx_i, 0, size - 1)
    if policy.mode == SafetyMode.DROP:
        idx_safe = jnp.where(ok, idx_safe, jnp.int32(0))
    values = arr[idx_safe]
    if policy.mode == SafetyMode.DROP:
        values = jnp.where(ok, values, jnp.zeros_like(values))
    if policy.mode == SafetyMode.CLAMP:
        ok = jnp.ones_like(ok, dtype=jnp.bool_)
    if return_ok:
        return values, ok
    return values


# dataflow-bundle: arr, idx, label, policy
def safe_gather_1d_ok(
    arr,
    idx,
    label="safe_gather_1d_ok",
    guard=None,
    *,
    policy: SafetyPolicy | None = None,
):
    """Guarded gather that also returns ok + corruption flag per SafetyPolicy."""
    values, ok = safe_gather_1d(
        arr,
        idx,
        label,
        guard=guard,
        policy=policy,
        return_ok=True,
    )
    if policy is None:
        policy = DEFAULT_SAFETY_POLICY
    corrupt = oob_mask(ok, policy=policy)
    return values, ok, corrupt


def safe_gather_1d_value(
    arr,
    idx,
    label="safe_gather_1d_value",
    guard=None,
    *,
    policy_value: PolicyValue,
    return_ok: bool = False,
):
    """Guarded gather that accepts policy as a JAX value."""
    values, ok = _safe_gather_1d_value_ok(
        arr,
        idx,
        label,
        guard=guard,
        policy_value=policy_value,
    )
    if return_ok:
        return values, ok
    return values


def safe_gather_1d_ok_value(
    arr,
    idx,
    label="safe_gather_1d_ok_value",
    guard=None,
    *,
    policy_value: PolicyValue,
):
    """Guarded gather that returns ok + corruption flag using policy value."""
    values, ok = _safe_gather_1d_value_ok(
        arr,
        idx,
        label,
        guard=guard,
        policy_value=policy_value,
    )
    bundle = _IndexPolicyValue(idx=idx, policy_value=policy_value)
    policy_val = jnp.asarray(bundle.policy_value, dtype=jnp.int32)
    corrupt = jnp.where(policy_val == POLICY_VALUE_CORRUPT, ~ok, False)
    return values, ok, corrupt


def _safe_gather_1d_value_ok(
    arr,
    idx,
    label,
    *,
    guard=None,
    policy_value: PolicyValue,
):
    bundle = _IndexPolicyValue(idx=idx, policy_value=policy_value)
    size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
    idx_i = jnp.asarray(bundle.idx, dtype=jnp.int32)
    policy_val = jnp.asarray(bundle.policy_value, dtype=jnp.int32)
    guard_gather_index(idx_i, size, label, guard=guard)
    ok = (idx_i >= 0) & (idx_i < size)
    idx_safe = jnp.clip(idx_i, 0, size - 1)
    drop_mask = policy_val == POLICY_VALUE_DROP
    clamp_mask = policy_val == POLICY_VALUE_CLAMP
    idx_safe = jnp.where(drop_mask, jnp.where(ok, idx_safe, jnp.int32(0)), idx_safe)
    values = arr[idx_safe]
    values = jnp.where(drop_mask & (~ok), jnp.zeros_like(values), values)
    ok = jnp.where(clamp_mask, jnp.ones_like(ok, dtype=jnp.bool_), ok)
    return values, ok


def safe_index_1d(
    idx,
    size,
    label="safe_index_1d",
    guard=None,
    *,
    policy: SafetyPolicy | None = None,
):
    """Return a policy-aware safe index and in-bounds mask."""
    if policy is None:
        policy = DEFAULT_SAFETY_POLICY
    bundle = _IndexSizeArgs(idx=idx, size=size)
    size_i = jnp.asarray(bundle.size, dtype=jnp.int32)
    idx_i = jnp.asarray(bundle.idx, dtype=jnp.int32)
    guard_gather_index(idx_i, size_i, label, guard=guard)
    ok = (idx_i >= 0) & (idx_i < size_i)
    idx_safe = jnp.clip(idx_i, 0, size_i - 1)
    if policy.mode == SafetyMode.DROP:
        idx_safe = jnp.where(ok, idx_safe, jnp.int32(0))
    if policy.mode == SafetyMode.CLAMP:
        ok = jnp.ones_like(ok, dtype=jnp.bool_)
    return idx_safe, ok


def safe_index_1d_value(
    idx,
    size,
    label="safe_index_1d_value",
    guard=None,
    *,
    policy_value: PolicyValue,
):
    """Return a policy-aware safe index using policy value."""
    bundle = _IndexPolicyValueSize(idx=idx, policy_value=policy_value, size=size)
    size_i = jnp.asarray(bundle.size, dtype=jnp.int32)
    idx_i = jnp.asarray(bundle.idx, dtype=jnp.int32)
    policy_val = jnp.asarray(bundle.policy_value, dtype=jnp.int32)
    guard_gather_index(idx_i, size_i, label, guard=guard)
    ok = (idx_i >= 0) & (idx_i < size_i)
    idx_safe = jnp.clip(idx_i, 0, size_i - 1)
    drop_mask = policy_val == POLICY_VALUE_DROP
    clamp_mask = policy_val == POLICY_VALUE_CLAMP
    idx_safe = jnp.where(drop_mask, jnp.where(ok, idx_safe, jnp.int32(0)), idx_safe)
    ok = jnp.where(clamp_mask, jnp.ones_like(ok, dtype=jnp.bool_), ok)
    return idx_safe, ok


__all__ = [
    "TEST_GUARDS",
    "SCATTER_GUARD",
    "GATHER_GUARD",
    "HAS_DEBUG_CALLBACK",
    "scatter_guard",
    "scatter_guard_strict",
    "scatter_drop",
    "scatter_strict",
    "guard_gather_index",
    "safe_gather_1d",
    "safe_gather_1d_ok",
    "safe_gather_1d_value",
    "safe_gather_1d_ok_value",
    "safe_index_1d",
    "safe_index_1d_value",
]
