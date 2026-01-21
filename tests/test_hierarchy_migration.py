import jax.numpy as jnp

import prism_vm as pv


def _tagged_arena(indices, tags, ranks):
    arena = pv.init_arena()
    idx = jnp.array(indices, dtype=jnp.int32)
    arena = arena._replace(
        opcode=arena.opcode.at[idx].set(jnp.array(tags, dtype=jnp.int32)),
        arg1=arena.arg1.at[idx].set(0),
        arg2=arena.arg2.at[idx].set(0),
        rank=arena.rank.at[idx].set(jnp.array(ranks, dtype=jnp.int8)),
        count=jnp.array(max(indices) + 1, dtype=jnp.int32),
    )
    return arena


def test_hierarchy_migrates_within_l1():
    l2_block = 4
    l1_block = 8
    indices = list(range(8))
    tags = [100 + i for i in indices]
    ranks = [pv.RANK_COLD] * 8
    ranks[6] = pv.RANK_HOT
    arena = _tagged_arena(indices, tags, ranks)

    blocked = pv.op_sort_and_swizzle_blocked(arena, l2_block)
    assert int(blocked.opcode[0]) != 106
    assert int(blocked.opcode[4]) == 106

    hierarchical = pv.op_sort_and_swizzle_hierarchical(
        arena, l2_block, l1_block, do_global=False
    )
    assert int(hierarchical.opcode[0]) == 106


def test_hierarchy_no_cross_l1_without_global():
    l2_block = 4
    l1_block = 8
    indices = list(range(16))
    tags = [200 + i for i in indices]
    ranks = [pv.RANK_COLD] * 16
    ranks[12] = pv.RANK_HOT
    arena = _tagged_arena(indices, tags, ranks)

    hierarchical = pv.op_sort_and_swizzle_hierarchical(
        arena, l2_block, l1_block, do_global=False
    )
    assert not bool(jnp.any(hierarchical.opcode[:l1_block] == 212))


def test_hierarchy_global_allows_cross_l1():
    l2_block = 4
    l1_block = 8
    indices = list(range(16))
    tags = [200 + i for i in indices]
    ranks = [pv.RANK_COLD] * 16
    ranks[12] = pv.RANK_HOT
    arena = _tagged_arena(indices, tags, ranks)

    hierarchical = pv.op_sort_and_swizzle_hierarchical(
        arena, l2_block, l1_block, do_global=True
    )
    assert bool(jnp.any(hierarchical.opcode[:l1_block] == 212))
