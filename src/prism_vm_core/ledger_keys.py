from __future__ import annotations

import jax.numpy as jnp

from prism_vm_core.constants import MAX_COUNT, MAX_ID


def _pack_key(op, a1, a2):
    # Byte layout: op, a1_hi, a1_lo, a2_hi, a2_lo for lexicographic sort.
    op_u = op.astype(jnp.uint8)
    a1_u = (a1.astype(jnp.uint32) & jnp.uint32(0xFFFF)).astype(jnp.uint16)
    a2_u = (a2.astype(jnp.uint32) & jnp.uint32(0xFFFF)).astype(jnp.uint16)
    a1_hi = (a1_u >> jnp.uint16(8)).astype(jnp.uint8)
    a1_lo = (a1_u & jnp.uint16(0xFF)).astype(jnp.uint8)
    a2_hi = (a2_u >> jnp.uint16(8)).astype(jnp.uint8)
    a2_lo = (a2_u & jnp.uint16(0xFF)).astype(jnp.uint8)
    return op_u, a1_hi, a1_lo, a2_hi, a2_lo


def _checked_pack_key(op, a1, a2, count):
    # Checked pack: enforce bounds before fixed-width encoding to avoid aliasing.
    max_id = jnp.int32(MAX_ID)
    max_count = jnp.int32(MAX_COUNT)
    enabled = op != 0
    op_oob = (op < 0) | (op > jnp.int32(255))
    a1_oob = (a1 < 0) | (a1 > max_id)
    a2_oob = (a2 < 0) | (a2 > max_id)
    count_oob = (count < 0) | (count > max_count)
    bad = count_oob | jnp.any(enabled & (op_oob | a1_oob | a2_oob))
    op_safe = jnp.where(bad, jnp.int32(0), op)
    a1_safe = jnp.where(bad, jnp.int32(0), a1)
    a2_safe = jnp.where(bad, jnp.int32(0), a2)
    return bad, _pack_key(op_safe, a1_safe, a2_safe)


__all__ = ["_pack_key", "_checked_pack_key"]
