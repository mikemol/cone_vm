from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from prism_vm_core.structures import Ledger


@dataclass(frozen=True, slots=True)
class LedgerIndex:
    """Derived index data for fast ledger lookups (data-level cache)."""

    op_start: jnp.ndarray
    op_end: jnp.ndarray


def derive_ledger_index(
    ledger: Ledger,
    *,
    op_buckets_full_range: bool,
) -> LedgerIndex:
    """Derive opcode buckets from the ledger bundle (pure)."""
    count = ledger.count.astype(jnp.int32)
    if op_buckets_full_range:
        op_start = jnp.zeros(256, dtype=jnp.int32)
        op_end = jnp.full((256,), count, dtype=jnp.int32)
    else:
        op_values = jnp.arange(256, dtype=jnp.uint8)
        op_start = jnp.searchsorted(
            ledger.keys_b0_sorted, op_values, side="left"
        ).astype(jnp.int32)
        op_end = jnp.searchsorted(
            ledger.keys_b0_sorted, op_values, side="right"
        ).astype(jnp.int32)
        op_start = jnp.minimum(op_start, count)
        op_end = jnp.minimum(op_end, count)
    return LedgerIndex(op_start=op_start, op_end=op_end)


__all__ = ["LedgerIndex", "derive_ledger_index"]
