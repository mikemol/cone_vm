import jax.numpy as jnp

from prism_core.compact import compact_mask


def _candidate_indices(enabled):
    return compact_mask(enabled, index_dtype=jnp.int32, count_dtype=jnp.int32)


__all__ = ["_candidate_indices"]
