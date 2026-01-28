from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp

from prism_vm_core.constants import LEDGER_CAPACITY, MAX_ROWS
from prism_vm_core.ontology import OP_NULL, OP_ZERO
from prism_vm_core.structures import Arena, Ledger, Manifest


def init_manifest(max_rows: int = MAX_ROWS) -> Manifest:
    return Manifest(
        opcode=jnp.zeros(max_rows, dtype=jnp.int32),
        arg1=jnp.zeros(max_rows, dtype=jnp.int32),
        arg2=jnp.zeros(max_rows, dtype=jnp.int32),
        active_count=jnp.array(1, dtype=jnp.int32),
        oom=jnp.array(False, dtype=jnp.bool_),
    )


def init_arena(
    capacity: int,
    rank_free: int,
    op_zero: int = OP_ZERO,
) -> Arena:
    arena = Arena(
        opcode=jnp.zeros(capacity, dtype=jnp.int32),
        arg1=jnp.zeros(capacity, dtype=jnp.int32),
        arg2=jnp.zeros(capacity, dtype=jnp.int32),
        rank=jnp.full(capacity, rank_free, dtype=jnp.int8),
        count=jnp.array(1, dtype=jnp.int32),
        oom=jnp.array(False, dtype=jnp.bool_),
        servo=jnp.zeros(3, dtype=jnp.uint32),
    )
    arena = arena._replace(
        opcode=arena.opcode.at[1].set(op_zero),
        arg1=arena.arg1.at[1].set(0),
        arg2=arena.arg2.at[1].set(0),
        count=jnp.array(2, dtype=jnp.int32),
    )
    return arena


def init_ledger(
    pack_key: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, ...]],
    capacity: int = LEDGER_CAPACITY,
    op_null: int = OP_NULL,
    op_zero: int = OP_ZERO,
) -> Ledger:
    max_key = jnp.uint8(0xFF)

    opcode = jnp.zeros(capacity, dtype=jnp.int32)
    arg1 = jnp.zeros(capacity, dtype=jnp.int32)
    arg2 = jnp.zeros(capacity, dtype=jnp.int32)

    opcode = opcode.at[1].set(op_zero)

    keys_b0_sorted = jnp.full(capacity, max_key, dtype=jnp.uint8)
    keys_b1_sorted = jnp.full(capacity, max_key, dtype=jnp.uint8)
    keys_b2_sorted = jnp.full(capacity, max_key, dtype=jnp.uint8)
    keys_b3_sorted = jnp.full(capacity, max_key, dtype=jnp.uint8)
    keys_b4_sorted = jnp.full(capacity, max_key, dtype=jnp.uint8)
    ids_sorted = jnp.zeros(capacity, dtype=jnp.int32)

    k0_b0, k0_b1, k0_b2, k0_b3, k0_b4 = pack_key(
        jnp.uint8(op_null), jnp.uint16(0), jnp.uint16(0)
    )
    k1_b0, k1_b1, k1_b2, k1_b3, k1_b4 = pack_key(
        jnp.uint8(op_zero), jnp.uint16(0), jnp.uint16(0)
    )
    keys_b0_sorted = keys_b0_sorted.at[0].set(k0_b0).at[1].set(k1_b0)
    keys_b1_sorted = keys_b1_sorted.at[0].set(k0_b1).at[1].set(k1_b1)
    keys_b2_sorted = keys_b2_sorted.at[0].set(k0_b2).at[1].set(k1_b2)
    keys_b3_sorted = keys_b3_sorted.at[0].set(k0_b3).at[1].set(k1_b3)
    keys_b4_sorted = keys_b4_sorted.at[0].set(k0_b4).at[1].set(k1_b4)
    ids_sorted = ids_sorted.at[0].set(0).at[1].set(1)

    return Ledger(
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
        keys_b0_sorted=keys_b0_sorted,
        keys_b1_sorted=keys_b1_sorted,
        keys_b2_sorted=keys_b2_sorted,
        keys_b3_sorted=keys_b3_sorted,
        keys_b4_sorted=keys_b4_sorted,
        ids_sorted=ids_sorted,
        count=jnp.array(2, dtype=jnp.int32),
        oom=jnp.array(False, dtype=jnp.bool_),
        corrupt=jnp.array(False, dtype=jnp.bool_),
    )


__all__ = ["init_manifest", "init_arena", "init_ledger"]
