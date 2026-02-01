from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax

from prism_vm_core.structures import Ledger


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class LedgerIndex:
    """Derived index data for fast ledger lookups (data-level cache)."""

    op_start: jnp.ndarray
    op_end: jnp.ndarray

    def tree_flatten(self):
        return (self.op_start, self.op_end), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        op_start, op_end = children
        return cls(op_start=op_start, op_end=op_end)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class LedgerState:
    """Ledger plus derived index bundle (canonical interning state)."""

    ledger: Ledger
    index: LedgerIndex
    op_buckets_full_range: bool

    def tree_flatten(self):
        return (self.ledger, self.index), self.op_buckets_full_range

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ledger, index = children
        return cls(ledger=ledger, index=index, op_buckets_full_range=aux_data)

    def __getattr__(self, name: str):
        return getattr(self.ledger, name)

    def _replace(self, **kwargs):
        ledger = kwargs.pop("ledger", self.ledger)
        if kwargs:
            ledger = ledger._replace(**kwargs)
        return LedgerState(
            ledger=ledger,
            index=derive_ledger_index(
                ledger,
                op_buckets_full_range=self.op_buckets_full_range,
            ),
            op_buckets_full_range=self.op_buckets_full_range,
        )


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


def derive_ledger_state(
    ledger: Ledger,
    *,
    op_buckets_full_range: bool,
) -> LedgerState:
    """Derive a LedgerState from a ledger bundle (pure)."""
    return LedgerState(
        ledger=ledger,
        index=derive_ledger_index(
            ledger, op_buckets_full_range=op_buckets_full_range
        ),
        op_buckets_full_range=op_buckets_full_range,
    )


__all__ = ["LedgerIndex", "LedgerState", "derive_ledger_index", "derive_ledger_state"]
