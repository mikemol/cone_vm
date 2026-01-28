import jax.numpy as jnp


def _candidate_indices(enabled):
    size = enabled.shape[0]
    count = jnp.sum(enabled).astype(jnp.int32)
    idx = jnp.nonzero(enabled, size=size, fill_value=0)[0].astype(jnp.int32)
    valid = jnp.arange(size, dtype=jnp.int32) < count
    return idx, valid, count


__all__ = ["_candidate_indices"]
