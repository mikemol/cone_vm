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


def _k_from_mask(mask):
    return int(jax.device_get(pv._servo_mask_to_k(mask)))


def test_lung_capacity_dilate_contract():
    size = 64
    idx = jnp.arange(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = jnp.bitwise_xor(idx, jnp.int32(32))
    arg2 = arg1
    rank = jnp.where(idx > 0, pv.RANK_HOT, pv.RANK_FREE).astype(jnp.int8)
    count = jnp.array(size, dtype=jnp.int32)
    k = 6
    mask = pv._servo_mask_from_k(jnp.int32(k))
    arena = _make_arena(opcode, arg1, arg2, rank, count, mask)

    updated = pv._servo_update(arena)
    k_next = _k_from_mask(updated.servo[0])
    assert k_next == k + 1


def test_lung_capacity_contract():
    size = 64
    idx = jnp.arange(size, dtype=jnp.int32)
    opcode = jnp.zeros(size, dtype=jnp.int32)
    arg1 = idx
    arg2 = arg1
    rank = jnp.where((idx > 0) & (idx <= 8), pv.RANK_HOT, pv.RANK_FREE).astype(
        jnp.int8
    )
    count = jnp.array(size, dtype=jnp.int32)
    k = 6
    mask = pv._servo_mask_from_k(jnp.int32(k))
    arena = _make_arena(opcode, arg1, arg2, rank, count, mask)

    updated = pv._servo_update(arena)
    k_next = _k_from_mask(updated.servo[0])
    assert k_next == k - 1
