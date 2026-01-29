import jax.numpy as jnp

from prism_core.compact import CompactResult, compact_mask


def _candidate_indices(enabled) -> CompactResult:
    return compact_mask(enabled, index_dtype=jnp.int32, count_dtype=jnp.int32)


def candidate_indices_cfg(enabled, *, candidate_indices_fn=None):
    """Interface/Control wrapper for candidate index selection."""
    if candidate_indices_fn is None:
        candidate_indices_fn = _candidate_indices
    return candidate_indices_fn(enabled)


__all__ = ["_candidate_indices", "candidate_indices_cfg"]
