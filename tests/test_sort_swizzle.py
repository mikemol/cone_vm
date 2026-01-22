import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m4


def _arena_with_edges():
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_ADD).at[3].set(pv.OP_SUC),
        arg1=arena.arg1.at[2].set(3).at[3].set(1),
        arg2=arena.arg2.at[2].set(1).at[3].set(0),
        rank=arena.rank.at[1].set(pv.RANK_COLD)
        .at[2].set(pv.RANK_HOT)
        .at[3].set(pv.RANK_COLD),
        count=jnp.array(4, dtype=jnp.int32),
    )
    return arena


def test_swizzle_preserves_edges():
    assert hasattr(pv, "op_sort_and_swizzle"), "op_sort_and_swizzle missing"
    arena = _arena_with_edges()
    sorted_arena = pv.op_sort_and_swizzle(arena)
    assert int(sorted_arena.opcode[0]) == pv.OP_ADD
    assert int(sorted_arena.arg1[0]) == 2
    assert int(sorted_arena.arg2[0]) == 1
    assert int(sorted_arena.count) == int(arena.count)


def test_swizzle_null_pointer_stays_zero():
    assert hasattr(pv, "op_sort_and_swizzle"), "op_sort_and_swizzle missing"
    arena = _arena_with_edges()
    sorted_arena = pv.op_sort_and_swizzle(arena)
    assert int(sorted_arena.opcode[2]) == pv.OP_SUC
    assert int(sorted_arena.arg2[2]) == 0


def test_sort_swizzle_root_remap():
    assert hasattr(pv, "op_sort_and_swizzle_with_perm"), "op_sort_and_swizzle_with_perm missing"
    arena = _arena_with_edges()
    sorted_arena, inv_perm = pv.op_sort_and_swizzle_with_perm(arena)
    root_old = jnp.array(3, dtype=jnp.int32)
    root_new = jnp.where(root_old != 0, inv_perm[root_old], 0)
    assert int(root_new) == 2
    assert int(sorted_arena.opcode[root_new]) == pv.OP_SUC
    assert int(sorted_arena.arg1[root_new]) == 1
