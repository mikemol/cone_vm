import os
import jax
import jax.numpy as jnp

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

    jax.debug.callback(_raise, bad, min_idx, max_idx, size)


def safe_gather_1d(arr, idx, label="safe_gather_1d", guard=None):
    # Guarded gather: raise on invalid indices in test mode; always clamp for
    # deterministic OOB behavior across backends.
    size = jnp.asarray(arr.shape[0], dtype=jnp.int32)
    idx_i = jnp.asarray(idx, dtype=jnp.int32)
    guard_gather_index(idx_i, size, label, guard=guard)
    idx_safe = jnp.clip(idx_i, 0, size - 1)
    return arr[idx_safe]


def safe_index_1d(idx, size, label="safe_index_1d", guard=None):
    """Return a clamped index and an in-bounds mask for 1D indexing."""
    size_i = jnp.asarray(size, dtype=jnp.int32)
    idx_i = jnp.asarray(idx, dtype=jnp.int32)
    guard_gather_index(idx_i, size_i, label, guard=guard)
    ok = (idx_i >= 0) & (idx_i < size_i)
    idx_safe = jnp.clip(idx_i, 0, size_i - 1)
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
    "safe_index_1d",
]
