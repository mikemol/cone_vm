import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m6


def test_block_local_sort():
    assert hasattr(pv, "op_sort_and_swizzle_blocked"), "op_sort_and_swizzle_blocked missing"
    block_size = 4
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[1].set(pv.OP_ADD)
        .at[2].set(pv.OP_SUC)
        .at[5].set(pv.OP_ADD)
        .at[6].set(pv.OP_SUC),
        arg1=arena.arg1.at[1].set(111).at[2].set(222).at[5].set(555).at[6].set(666),
        rank=arena.rank.at[1].set(pv.RANK_HOT)
        .at[2].set(pv.RANK_COLD)
        .at[5].set(pv.RANK_HOT)
        .at[6].set(pv.RANK_COLD),
        count=jnp.array(7, dtype=jnp.int32),
    )
    sorted_arena = pv.op_sort_and_swizzle_blocked(arena, block_size)
    block0 = sorted_arena.arg1[:block_size]
    block1 = sorted_arena.arg1[block_size : block_size * 2]
    assert not bool(jnp.any(jnp.isin(block0, jnp.array([555, 666], dtype=jnp.int32))))
    assert bool(jnp.any(block1 == 555))
    assert bool(jnp.any(block1 == 666))


def test_single_block_same_as_global():
    assert hasattr(pv, "op_sort_and_swizzle_blocked"), "op_sort_and_swizzle_blocked missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_ADD).at[3].set(pv.OP_SUC).at[4].set(pv.OP_ADD),
        rank=arena.rank.at[2].set(pv.RANK_HOT)
        .at[3].set(pv.RANK_COLD)
        .at[4].set(pv.RANK_HOT),
        count=jnp.array(5, dtype=jnp.int32),
    )
    baseline = pv.op_sort_and_swizzle(arena)
    size = int(arena.rank.shape[0])
    blocked = pv.op_sort_and_swizzle_blocked(arena, size)
    assert bool(jnp.array_equal(blocked.opcode, baseline.opcode))
    assert bool(jnp.array_equal(blocked.arg1, baseline.arg1))
    assert bool(jnp.array_equal(blocked.arg2, baseline.arg2))
    assert bool(jnp.array_equal(blocked.rank, baseline.rank))
    assert int(blocked.count) == int(baseline.count)
