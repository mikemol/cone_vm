from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp


class Manifest(NamedTuple):
    opcode: jnp.ndarray
    arg1: jnp.ndarray
    arg2: jnp.ndarray
    active_count: jnp.ndarray
    oom: jnp.ndarray


class Arena(NamedTuple):
    opcode: jnp.ndarray
    arg1: jnp.ndarray
    arg2: jnp.ndarray
    rank: jnp.ndarray
    count: jnp.ndarray
    oom: jnp.ndarray
    servo: jnp.ndarray


class Ledger(NamedTuple):
    opcode: jnp.ndarray
    arg1: jnp.ndarray
    arg2: jnp.ndarray
    keys_b0_sorted: jnp.ndarray
    keys_b1_sorted: jnp.ndarray
    keys_b2_sorted: jnp.ndarray
    keys_b3_sorted: jnp.ndarray
    keys_b4_sorted: jnp.ndarray
    ids_sorted: jnp.ndarray
    count: jnp.ndarray
    oom: jnp.ndarray
    corrupt: jnp.ndarray


class CandidateBuffer(NamedTuple):
    enabled: jnp.ndarray
    opcode: jnp.ndarray
    arg1: jnp.ndarray
    arg2: jnp.ndarray


class NodeBatch(NamedTuple):
    op: jnp.ndarray
    a1: jnp.ndarray
    a2: jnp.ndarray


class Stratum(NamedTuple):
    start: jnp.ndarray
    count: jnp.ndarray


@dataclass(frozen=True)
class StagingContext:
    # Staging context (n, s, t, tile) per in-19.
    n: int
    s: int
    t: int
    tile: int | None = None


def hyperstrata_precedes(s1: int, t1: int, s2: int, t2: int) -> bool:
    return (s1 < s2) or (s1 == s2 and t1 < t2)


def staging_context_forgets_detail(
    fine: StagingContext, coarse: StagingContext
) -> bool:
    # A morphism exists from fine -> coarse if coarse is a coarser view.
    tile_ok = coarse.tile is None or fine.tile == coarse.tile
    return (
        fine.n >= coarse.n
        and fine.s >= coarse.s
        and fine.t >= coarse.t
        and tile_ok
    )
