import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m3


def test_block_local_sort():
    assert hasattr(pv, "op_sort_and_swizzle_blocked"), "op_sort_and_swizzle_blocked missing"
    block_size = 3
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[1].set(pv.OP_ADD)
        .at[2].set(pv.OP_SUC)
        .at[4].set(pv.OP_MUL)
        .at[5].set(pv.OP_MUL),
        arg1=arena.arg1.at[1].set(0).at[2].set(0).at[4].set(0).at[5].set(0),
        arg2=arena.arg2.at[1].set(0).at[2].set(0).at[4].set(0).at[5].set(0),
        rank=arena.rank.at[1].set(pv.RANK_HOT)
        .at[2].set(pv.RANK_COLD)
        .at[4].set(pv.RANK_HOT)
        .at[5].set(pv.RANK_COLD),
        count=jnp.array(6, dtype=jnp.int32),
    )
    sorted_arena = pv.op_sort_and_swizzle_blocked(arena, block_size)
    block0 = sorted_arena.opcode[:block_size]
    block1 = sorted_arena.opcode[block_size : block_size * 2]
    assert not bool(jnp.any(block0 == pv.OP_MUL))
    assert bool(jnp.any(block1 == pv.OP_MUL))


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
