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


__all__ = ["CompactResult", "compact_mask"]
