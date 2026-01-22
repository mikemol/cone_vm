import jax.numpy as jnp
import pytest

import prism_vm as pv

pytestmark = pytest.mark.m4


def test_morton_key_stable():
    assert hasattr(pv, "op_sort_and_swizzle_morton"), "op_sort_and_swizzle_morton missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_ADD).at[3].set(pv.OP_SUC),
        rank=arena.rank.at[2].set(pv.RANK_HOT).at[3].set(pv.RANK_HOT),
        count=jnp.array(4, dtype=jnp.int32),
    )
    morton = jnp.zeros_like(arena.rank, dtype=jnp.uint32)
    morton = morton.at[2].set(5).at[3].set(1)
    sorted_arena = pv.op_sort_and_swizzle_morton(arena, morton)
    assert int(sorted_arena.opcode[0]) == pv.OP_SUC
    assert int(sorted_arena.opcode[1]) == pv.OP_ADD


def test_morton_disabled_matches_rank_sort():
    assert hasattr(pv, "op_morton"), "op_morton missing"
    arena = pv.init_arena()
    arena = arena._replace(
        opcode=arena.opcode.at[2].set(pv.OP_ADD).at[3].set(pv.OP_SUC).at[4].set(pv.OP_ADD),
        rank=arena.rank.at[2].set(pv.RANK_HOT)
        .at[3].set(pv.RANK_COLD)
        .at[4].set(pv.RANK_HOT),
        count=jnp.array(5, dtype=jnp.int32),
    )
    baseline = pv.op_sort_and_swizzle(arena)
    _ = pv.op_morton(arena)
    morton = jnp.zeros_like(arena.rank, dtype=jnp.uint32)
    with_morton = pv.op_sort_and_swizzle_morton(arena, morton)
    assert bool(jnp.array_equal(with_morton.opcode, baseline.opcode))
    assert bool(jnp.array_equal(with_morton.arg1, baseline.arg1))
    assert bool(jnp.array_equal(with_morton.arg2, baseline.arg2))
    assert bool(jnp.array_equal(with_morton.rank, baseline.rank))
    assert int(with_morton.count) == int(baseline.count)
