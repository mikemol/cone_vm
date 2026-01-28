import jax.numpy as jnp


def _invert_perm(perm):
    inv = jnp.empty_like(perm)
    return inv.at[perm].set(jnp.arange(perm.shape[0], dtype=perm.dtype))


__all__ = ["_invert_perm"]
