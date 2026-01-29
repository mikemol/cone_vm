import jax.numpy as jnp


def compact_mask(mask, *, index_dtype=None, count_dtype=None):
    """Compact a boolean mask into indices + valid mask + count.

    Returns:
      idx: indices of True entries (padded to size with 0)
      valid: boolean mask for entries < count
      count: number of True entries
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
    return idx, valid, count


__all__ = ["compact_mask"]
