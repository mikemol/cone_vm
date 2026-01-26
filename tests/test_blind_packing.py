import os

import jax
import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m5


def _make_arena(opcode, arg1, arg2, rank, count, servo_mask):
    servo = jnp.array([servo_mask, 0, 0], dtype=jnp.uint32)
    return pv.Arena(
        opcode=opcode,
        arg1=arg1,
        arg2=arg2,
        rank=rank,
        count=count,
        oom=jnp.array(False, dtype=jnp.bool_),
        servo=servo,
    )


def test_blind_packing(monkeypatch):
    size = 1025
    idx = jnp.arange(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = jnp.zeros(size, dtype=jnp.int32)
    arg2 = jnp.zeros(size, dtype=jnp.int32)

    large_mask = (idx >= 1) & (idx <= 512)
    small_mask = (idx >= 513) & (idx < size)
    arg1 = jnp.where(large_mask, jnp.maximum(idx - 1, 1), arg1)
    arg2 = jnp.where(large_mask, jnp.maximum(idx - 1, 1), arg2)
    arg1 = jnp.where(small_mask, idx, arg1)
    arg2 = jnp.where(small_mask, idx, arg2)

    rank = jnp.where(idx > 0, pv.RANK_HOT, pv.RANK_FREE).astype(jnp.int8)
    count = jnp.array(size, dtype=jnp.int32)
    mask = pv._servo_mask_from_k(jnp.int32(9))
    arena = _make_arena(opcode, arg1, arg2, rank, count, mask)

    morton = pv.op_morton(arena)
    arena_sorted, _ = pv.op_sort_and_swizzle_servo_with_perm(
        arena, morton, arena.servo[0]
    )

    spectrum = pv._blind_spectral_probe(arena_sorted)
    entropy = -jnp.sum(spectrum * jnp.log2(spectrum + 1e-6))
    assert float(entropy) < 1.5

    monkeypatch.setenv("PRISM_DAMAGE_METRICS", "1")
    pv.damage_metrics_reset()
    pv._damage_metrics_update(arena_sorted, tile_size=512)
    metrics = pv.damage_metrics_get()
    assert metrics["damage_rate"] < 0.01
