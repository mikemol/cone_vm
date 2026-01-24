import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m3


def _tagged_arena(indices, tags, ranks):
    arena = pv.init_arena()
    filtered = [
        (i, tag, rank)
        for i, tag, rank in zip(indices, tags, ranks)
        if i != 0
    ]
    idx = jnp.array([i for i, _, _ in filtered], dtype=jnp.int32)
    tag_arr = jnp.array([tag for _, tag, _ in filtered], dtype=jnp.int32)
    rank_arr = jnp.array([rank for _, _, rank in filtered], dtype=jnp.int8)
    arena = arena._replace(
        opcode=arena.opcode.at[idx].set(tag_arr),
        arg1=arena.arg1.at[idx].set(0),
        arg2=arena.arg2.at[idx].set(0),
        rank=arena.rank.at[idx].set(rank_arr),
        count=jnp.array(max(indices) + 1, dtype=jnp.int32),
    )
    return arena


def test_hierarchy_migrates_within_l1():
    l2_block = 3
    l1_block = 15
    indices = list(range(15))
    tags = [100 + i for i in indices]
    ranks = [pv.RANK_COLD] * 15
    ranks[10] = pv.RANK_HOT
    arena = _tagged_arena(indices, tags, ranks)

    blocked = pv.op_sort_and_swizzle_blocked(arena, l2_block)
    assert int(blocked.opcode[0]) != 110
    assert int(blocked.opcode[9]) == 110

    hierarchical = pv.op_sort_and_swizzle_hierarchical(
        arena, l2_block, l1_block, do_global=False
    )
    assert int(hierarchical.opcode[0]) == pv.OP_NULL
    assert int(hierarchical.opcode[1]) == 110


def test_hierarchy_no_cross_l1_without_global():
    l2_block = 3
    l1_block = 15
    indices = list(range(30))
    tags = [200 + i for i in indices]
    ranks = [pv.RANK_COLD] * 30
    ranks[22] = pv.RANK_HOT
    arena = _tagged_arena(indices, tags, ranks)

    hierarchical = pv.op_sort_and_swizzle_hierarchical(
        arena, l2_block, l1_block, do_global=False
    )
    assert not bool(jnp.any(hierarchical.opcode[:l1_block] == 222))


def test_hierarchy_global_allows_cross_l1():
    l2_block = 3
    l1_block = 15
    indices = list(range(30))
    tags = [200 + i for i in indices]
    ranks = [pv.RANK_COLD] * 30
    ranks[22] = pv.RANK_HOT
    arena = _tagged_arena(indices, tags, ranks)

    hierarchical = pv.op_sort_and_swizzle_hierarchical(
        arena, l2_block, l1_block, do_global=True
    )
    assert bool(jnp.any(hierarchical.opcode[:l1_block] == 222))
