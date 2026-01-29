from typing import NamedTuple

import jax.numpy as jnp


class CompactResult(NamedTuple):
    idx: jnp.ndarray
    valid: jnp.ndarray
    count: jnp.ndarray


def compact_mask(mask, *, index_dtype=None, count_dtype=None):
    """Compact a boolean mask into indices + valid mask + count.

    Returns:
      CompactResult: (idx, valid, count)
    """
    size = mask.shape[0]
    count = jnp.sum(mask)
    if count_dtype is not None:
        count = count.astype(count_dtype)
    idx = jnp.nonzero(mask, size=size, fill_value=0)[0]
    if index_dtype is not None:
        idx = idx.astype(index_dtype)
    comp_dtype = idx.dtype
    valid = jnp.arange(size, dtype=comp_dtype) < count.astype(comp_dtype)
    return CompactResult(idx=idx, valid=valid, count=count)

def scatter_compacted_ids(
    comp_idx,
    ids_compact,
    count,
    size,
    *,
    scatter_drop_fn,
    index_dtype=jnp.int32,
):
    """Scatter compacted ids back into full-size buffer using drop semantics."""
    valid = jnp.arange(size, dtype=index_dtype) < count
    scatter_idx = jnp.where(valid, comp_idx, index_dtype(size))
    scatter_ids = jnp.where(valid, ids_compact, jnp.int32(0))
    ids_full = jnp.zeros(size, dtype=ids_compact.dtype)
    return scatter_drop_fn(
        ids_full, scatter_idx, scatter_ids, "scatter_compacted_ids"
    )


__all__ = ["CompactResult", "compact_mask", "scatter_compacted_ids"]
